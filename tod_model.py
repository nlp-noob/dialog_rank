import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5Config
import json
from typing import List, Dict, Optional, Union, Tuple
import os

from triplet_loss import TripletLoss


class MLMBiencoder(nn.Module):
    def __init__(self, model_name_or_path: str, tokenizer: AutoTokenizer = None,
            model_args: Dict = {}, cache_dir: Optional[str] = None,
            mlm_probability: Optional[float]=None, mlm: Optional[bool]=True, margin: Optional[float]=1.0, 
            use_triplet_loss: Optional[bool]=True, use_rs_loss: Optional[bool]=True):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        self.config = config
        self.mlm_probability = mlm_probability or 0.15
        self.sim_scale = 20
        self.use_triplet_loss = use_triplet_loss
        self.use_rs_loss = use_rs_loss

        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        self.mlm = mlm
        if self.mlm:
            self.cls = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size, eps=1e-12),
                nn.Linear(hidden_size, vocab_size)
            )
        self.device = torch.device('cpu')
        self.margin = margin

    def to(self, device):
        self.device = device
        return super().to(device)

    def mask_tokens(self, inputs: torch.Tensor, mlm_probability: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        inputs = inputs.to("cpu")
        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        # padding position value = 0
        inputs_pad_pos = (inputs == 0).cpu()
        probability_matrix.masked_fill_(inputs_pad_pos, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        try:
            labels[~masked_indices] = -100  # We only compute loss on masked tokens
        except:
            masked_indices = masked_indices.byte()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        try:
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        except:
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().byte() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        try:
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        except:
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool().byte() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        if inputs.is_cuda:
            indices_random = indices_random.to(self.device)
            random_words = random_words.to(self.device)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def save(self, output_dir):
        self.encoder.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def encoder_forward(
        self,
        input_ids,
        attention_mask,
        labels=None
    ):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False)
        sequence_output = outputs[0]

        loss_mlm = torch.tensor(0.)
        if self.mlm and labels is not None:
            labels = labels.to(self.device)
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            prediction_scores = self.cls(sequence_output)
            loss_mlm = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # pooling to sentence embed
        # the sequence_output is the same as the sequence_output = lm_out.hedden_states[-1] in test_tod_rank.py
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output* input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        sentence_embed = sum_embeddings / sum_mask

        return loss_mlm, sentence_embed

    # model(input_mlm, input_cont, p_resp, n_resp, p_cluster_label_list
    def forward(self, input_mlm, cont, p_resp, n_resp, p_cluster_label_list):
        if self.mlm:
            input_mlm, labels = self.mask_tokens(input_mlm.clone(), self.mlm_probability)
            input_mlm = input_mlm.to(self.device)
            labels = labels.to(self.device)
            loss_mlm, _ = self.encoder_forward(
                    input_ids       = input_mlm,
                    attention_mask  = input_mlm > 0,
                    labels          = labels,
            )
        else:
            loss_mlm = torch.tensor(0.0)
        ###################################################################################
        ## Calculate the RS loss
        cont = cont.to(self.device)
        p_resp = p_resp.to(self.device)

        _, hid_cont = self.encoder_forward(
                input_ids       = cont,
                attention_mask  = cont > 0,
        )

        _, hid_p_resp = self.encoder_forward(
                input_ids       = p_resp,
                attention_mask  = p_resp > 0,
        )
        # the embeddings of the sentences
        hid_cont = torch.nn.functional.normalize(hid_cont, p=2, dim=1)
        hid_p_resp = torch.nn.functional.normalize(hid_p_resp, p=2, dim=1)

        # triplet loss with distance
        # distance_positive = (hid_cont - hid_p_resp).pow(2).sum(1) 
        # distance_negative = (hid_cont - hid_n_resp).pow(2).sum(1) 
        # loss_tri = torch.mean(torch.relu(distance_positive - distance_negative + self.margin))

        # triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)
        # loss_tri = triplet_loss(hid_cont, hid_p_resp, hid_n_resp)

        loss_tri = torch.tensor(0)
        distance_positive = torch.tensor(0)
        distance_negative = torch.tensor(0)

        scores = torch.matmul(hid_cont, hid_p_resp.transpose(1, 0)) * self.sim_scale
        xeloss = nn.CrossEntropyLoss()

        batch_size = cont.shape[0]

        resp_label = torch.arange(batch_size, device=self.device)
        if len(set(p_cluster_label_list)) < batch_size:
            duplicate_set = set()
            for i in range(batch_size):
                tmp = label[i].item()
                if tmp not in duplicate_set:
                    duplicate_set.add(tmp)
                    continue
                # ignore duplicate
                resp_label[i] = -100
        loss_rs = xeloss(scores, resp_label)

        ###################################################################################
        ## Calculate Triplet loss
        _, hid_n_resp = self.encoder_forward(
                input_ids       = n_resp,
                attention_mask  = n_resp > 0,
        )
        hid_p_resp = torch.nn.functional.normalize(hid_p_resp, p=2, dim=1)
        hid_n_resp = torch.nn.functional.normalize(hid_n_resp, p=2, dim=1)
        distance_positive = (hid_cont - hid_p_resp).pow(2).sum(1) 
        distance_negative = (hid_cont - hid_n_resp).pow(2).sum(1) 
        triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)
        loss_tri = triplet_loss(hid_cont, hid_p_resp, hid_n_resp)

        if self.use_triplet_loss and self.use_rs_loss:
            loss = loss_tri + loss_mlm + loss_rs
        elif self.use_triplet_loss and not self.use_rs_loss:
            loss = loss_tri + loss_mlm
        elif self.use_rs_loss and not self.use_triplet_loss:
            loss = loss_mlm + loss_rs

        return loss, (loss_mlm, loss_tri, loss_rs, scores, distance_positive, distance_negative, hid_cont, hid_p_resp)
        

def test():
    import sys
    model_name_or_path = sys.argv[1]
    device = torch.device('cuda:0')
    model = MLMBiencoder(model_name_or_path)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print("Number of parameters:", num_params)

    import pdb;pdb.set_trace()
    model.to(device)
    sample_str = 'hello world, this is a test example for model forward! please check it when failed'
    inputs = model.tokenizer(sample_str, return_tensors="pt")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    output = model.encoder_forward(input_ids, attention_mask)
    print(output[0])
    print([i.shape for i in output])
    output = model.encoder_forward(input_ids, attention_mask, labels=input_ids)
    print(output[0])
    print([i.shape for i in output])

    # output = model(input_ids, input_ids, input_ids, labels=input_ids)
    # print(output[0])
    # print([i.shape for i in output[1]])


if __name__ == '__main__':
    test()

