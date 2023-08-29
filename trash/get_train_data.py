# the get_scored_data --> tag_scored_data.py process is deprecated, use this new version to generate training data.
import argparse
import os
import random
import json
import torch
import numpy as np


from transformers import AutoModel, AutoTokenizer
from msc_dataset import load_datasets, load_msc_bert_test_datasets, load_star_list
from tod_model import MLMBiencoder
from tqdm import tqdm
from rich.console import Console
from dialogbert_tokenizer import get_dialog_tokenizer


console = Console()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class SentenceSplitter():
    def __init__(self, tokenizer):
        special_tokens = tokenizer.all_special_tokens
        special_token_ids = tokenizer.all_special_ids

        # get Advisor tokens
        self.advisor_token_ids = []
        self.cls_token_id = None
        self.usr_token_id = None

        for token, token_id in zip(special_tokens, special_token_ids):
            decode_token = tokenizer.decode(token_id)
            decode_token = decode_token.replace(" ", "")
            print("--"*20)
            print(token_id)
            print(decode_token)
            print(token)
            # check
            if decode_token != token:
                import pdb;pdb.set_trace()
            if "[ADVISOR" in decode_token:
                self.advisor_token_ids.append(token_id)
            if "[CLS]" == decode_token:
                self.cls_token_id = token_id
            if "[USR]" == decode_token:
                self.usr_token_id = token_id

    def get_advisor_sentences_from_cont(self, cont):
        is_advisor = False
        sentences = []
        sentence = []
        for token_id in cont:
            if token_id == self.cls_token_id:
                continue
            elif token_id in self.advisor_token_ids:
                is_advisor =  True
                if len(sentence) > 0:
                    sentence.insert(0, self.cls_token_id)
                    sentences.append(sentence)
                    sentence = [token_id]
                else:
                    sentence = [token_id]
            elif token_id == self.usr_token_id: 
                is_advisor = False
                if len(sentence) > 0:
                    sentence.insert(0, self.cls_token_id)
                    sentences.append(sentence)
                    sentence = []
            elif is_advisor:
                sentence.append(token_id)
    
        if len(sentence) > 0:
            sentence.insert(0, self.cls_token_id)
            sentences.append(sentence)
    
        return sentences


def get_cluster_info(dataset):
    data_size = len(dataset)
    cluster_label_to_index_list = {}
    index_to_cluster_label = {}
    console.print("Collating the cluster_info")
    for index in tqdm(range(data_size)):
        _, _, _, cluster_label = dataset[index]
        if cluster_label not in cluster_label_to_index_list:
            cluster_label_to_index_list[cluster_label] = []
        cluster_label_to_index_list[cluster_label].append(index)
        index_to_cluster_label[index] = cluster_label
    return cluster_label_to_index_list, index_to_cluster_label

# see the sentence appeared in the context as the negative sample
# notice that the p_index is empty here(self)
def tag_data_dup_sentence(args, dataset):
    cluster_label_to_index_list, index_to_cluster_label = get_cluster_info(dataset)
    tokenizer = get_dialog_tokenizer("distilbert", "distilbert-base-uncased")
    sentence_splitter = SentenceSplitter(tokenizer)
    tagged_data = {
        "cont": [],
        "resp": [],
        "mlm": [],
        "p_index": [],
        "n_index": [],
        "dup_n_id": [],
        "info": {
            "cluster_label_to_index_list": cluster_label_to_index_list,
            "index_to_cluster_label": index_to_cluster_label,
        },
    }
    data_size = len(dataset)
    text_to_resp_index = {}
    console.print("Collating the resp searching dict...")

    for index in tqdm(range(data_size)):
        _, resp, _, _ = dataset[index]
        # print("--"*20)
        # print(tokenizer.decode(resp))
        # print(tokenizer.decode(resp[2:]))
        # input()
        text_to_resp_index[tokenizer.decode(resp[2:])] = index


    console.print("Collating the n_index_list", style="green")
    for index in tqdm(range(data_size)):
        cont, resp, mlm, cluster_label = dataset[index]
        tagged_data["cont"].append(cont)
        tagged_data["resp"].append(resp)
        tagged_data["mlm"].append(mlm)
        tagged_data["p_index"].append([index])
        advisor_sentences_in_cont = sentence_splitter.get_advisor_sentences_from_cont(cont)
        n_index_list = []
        dup_id_list = []
        for advisor_sentence in advisor_sentences_in_cont:
            if len(advisor_sentence) < 5:
                continue
            text = tokenizer.decode(advisor_sentence[2:])
            if text in text_to_resp_index:
                n_index_list.append(text_to_resp_index[text])
                print("**"*20)
                print(text)
                print(tokenizer.decode(resp[2:]))

        tagged_data["n_index"].append(n_index_list)
        tagged_data["dup_n_id"].append(dup_id_list)

    return tagged_data

# use model prediction to tag the data
def tag_data_model(args, dataset):
    tokenizer = get_dialog_tokenizer("distilbert", "distilbert-base-uncased")
    cluster_label_to_index_list, index_to_cluster_label = get_cluster_info(dataset)
    
    tokenizer = get_dialog_tokenizer("distilbert", "distilbert-base-uncased")

    sentence_splitter = SentenceSplitter(tokenizer)
    model = MLMBiencoder(args.model_path, tokenizer, mlm_probability=0.15, mlm=False)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    data_size = len(dataset)

    list_resp_norm = []

    console.print("Collating the resp norm matrix...", style="green")

    tagged_data = {
        "cont": [],
        "resp": [],
        "mlm": [],
        "p_index": [],
        "n_index": [],
        "dup_n_id": [],
        "p_cluster_label_list": [],
        "n_cluster_label_list": [],
        "info": {
            "cluster_label_to_index_list": cluster_label_to_index_list,
            "index_to_cluster_label": index_to_cluster_label,
        },
    }

    set_seed(42)

    for index in tqdm(range(data_size)):
        cont, resp, mlm, cluster_label = dataset[index]
        cont_id = cont.copy()
        resp_id = resp.copy()
        tagged_data["cont"].append(cont_id)
        tagged_data["resp"].append(resp_id)
        tagged_data["mlm"].append(mlm)
        resp = torch.tensor(resp).to(device).unsqueeze(0)
        with torch.no_grad():
            _, hid_resp = model.encoder_forward(
                    input_ids = resp,
                    attention_mask = resp > 0,
                    )
            hid_resp_norm = torch.nn.functional.normalize(hid_resp, p=2, dim=1)
        list_resp_norm.append(hid_resp_norm)
    cur_resp_norm = torch.cat(list_resp_norm, dim=0)

    console.print("tagging the data", style="green")


    split_cnt = 0
    for index in tqdm(range(data_size)):
        cont, resp, mlm, cluster_label = dataset[index]

        advisor_sentences_in_cont = sentence_splitter.get_advisor_sentences_from_cont(cont_id)

        with torch.no_grad():
            cont = torch.tensor(cont).to(device).unsqueeze(0)
            _, hid_cont = model.encoder_forward(
                    input_ids = cont,
                    attention_mask = cont > 0,
                    )
            hid_cont_norm = torch.nn.functional.normalize(hid_cont, p=2, dim=1)
            score = torch.matmul(hid_cont_norm, cur_resp_norm.transpose(1, 0))
            rank_value_tensor, rank_tensor = score.topk(dim=-1, k=len(score[0]))
            rank_value_list = rank_value_tensor[0].tolist()
            rank_list = rank_tensor[0].tolist()
            rank_cluster_label_list = [index_to_cluster_label[index] for index in rank_list]

        other_cluster_resp_index_list = []
        other_p_cluster_label_list = []
        added_cluster_label_list = [cluster_label]
        # get other cluster resp index
        for rank_cluster_label, rank_resp_index in zip(rank_cluster_label_list, rank_list):
            if len(added_cluster_label_list) >= (args.other_cluster_topk + 1):
                break
            if rank_cluster_label in added_cluster_label_list:
                continue
            else:
                other_cluster_resp_index_list.append(rank_resp_index)
                added_cluster_label_list.append(rank_cluster_label)
                other_p_cluster_label_list.append(rank_cluster_label)

        # get negative cluster resp index
        added_cluster_label_list = [cluster_label]
        rank_negative_resp_index_list = []
        other_n_cluster_label_list = []
        for i in range(len(rank_list)):
            rank_cluster_label = rank_cluster_label_list[-(i+1)]
            rank_resp_index = rank_list[-(i+1)]
            if len(added_cluster_label_list) >= (args.other_cluster_topk + 1):
                break
            if rank_cluster_label in added_cluster_label_list:
                continue
            else:
                rank_negative_resp_index_list.append(rank_resp_index)
                added_cluster_label_list.append(rank_cluster_label)
                other_n_cluster_label_list.append(rank_cluster_label)

        # get medium cluster resp index
        added_cluster_label_list = [cluster_label]
        rank_medium_resp_index_list = []
        middle_index = int(len(rank_list)/2)
        for i in range(middle_index):
            rank_cluster_label = rank_cluster_label_list[middle_index + i]
            rank_resp_index = rank_list[middle_index + 1]
            if len(added_cluster_label_list) >= (args.other_cluster_topk + 1):
                break
            if rank_cluster_label in added_cluster_label_list:
                continue
            else:
                rank_medium_resp_index_list.append(rank_resp_index)
                added_cluster_label_list.append(rank_cluster_label)

        tagged_data["p_index"].append(other_cluster_resp_index_list)
        tagged_data["n_index"].append(rank_negative_resp_index_list)
        tagged_data["dup_n_id"].append(advisor_sentences_in_cont)
        tagged_data["p_cluster_label_list"].append(other_p_cluster_label_list)
        tagged_data["n_cluster_label_list"].append(other_n_cluster_label_list)
    return tagged_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path to the model. default: ./exp.distbert.b16/chkps_epoch-99/", type=str, default="./exp.distbert.b16/chkps_epoch-99/")
    args = parser.parse_args()
    parser.add_argument('--order_file', default='./data/orders.star1-2.1.json', type=str, help="The order to be tag")
    parser.add_argument('--fp16', action="store_true", default=False, help="user fp16")
    parser.add_argument('--device', type=str, default='cuda:0', help="default: cuda:0")
    parser.add_argument('--max_sess_size', help='gen msc segmentation with max session size, default=15', type=int, default=15)
    parser.add_argument('--min_sess_size', help='gen msc segmentation with min session size, default=5', type=int, default=5)
    parser.add_argument('--first_orders', help='Only take the first few orders between the same user and advisor, default=5', type=int, default=5)
    parser.add_argument('--p_score_thresh', help="the threshold of the possitive sample.", type=float, default=0.6)
    parser.add_argument('--other_cluster_topk', help='the number to get other cluster resp index(top score in cluster) list, default=10', type=int, default=15)
    parser.add_argument('--train_out_path', help='Path to out put the n_p_index data', type=str, default="./data/tri_train/train_v8.json")
    parser.add_argument('--valid_out_path', help='Path to out put the n_p_index data', type=str, default="./data/tri_train/valid_v8.json")
    args = parser.parse_args()
    tokenizer = get_dialog_tokenizer("distilbert", "distilbert-base-uncased")

    # just the whole dataset
    console.print("Loading the dataset...", style="green")

    # train_dataset, val_dataset, star_list = load_datasets(
    # dataset = load_msc_bert_test_datasets(
    #         args.order_file,
    #         first_orders=args.first_orders,
    #         max_sess_size=args.max_sess_size,
    #         min_sess_size=args.min_sess_size,
    #         tokenizer=tokenizer,
    #         data_type='msc_mlm_biencoder_cat',
    # )



    pre_star_list = load_star_list("data/star.list.220802.1-2")

    train_dataset, val_dataset, star_list = load_datasets(
            args.order_file,
            first_orders=args.first_orders,
            max_sess_size=args.max_sess_size,
            min_sess_size=args.min_sess_size,
            tokenizer=tokenizer,
            data_type='msc_mlm_biencoder_cat',
            star_list=pre_star_list,
    )
    import pdb;pdb.set_trace()

    # dataset = val_dataset
    tagged_train_data = tag_data_model(args, train_dataset)
    tagged_val_data = tag_data_model(args, val_dataset)
    # tagged_train_data = tag_data_dup_sentence(args, train_dataset)
    # tagged_val_data = tag_data_dup_sentence(args, val_dataset)

    with open(args.train_out_path, "w") as fout:
        json.dump(tagged_train_data, fout)
    with open(args.valid_out_path, "w") as fout:
        json.dump(tagged_val_data, fout)


if __name__ == "__main__":
    main()
