import argparse
import os
import random
import json
import torch
import numpy as np


from transformers import AutoModel, AutoTokenizer
from msc_dataset import load_datasets, load_msc_bert_test_datasets
from tod_model import MLMBiencoder
from tqdm import tqdm
from rich.console import Console


console = Console()

class SentenceSplitter():
    def __init__(self, tokenizer):
        special_tokens = tokenizer.all_special_tokens
        special_token_ids = tokenizer.all_special_ids

        # get Advisor tokens
        self.advisor_token_ids = []
        self.cls_token_id = None
        self.usr_token_id = None
        self.tokenizer = tokenizer

        for token, token_id in zip(special_tokens, special_token_ids):
            decode_token = tokenizer.decode(token_id)
            # check
            if decode_token != token:
                import pdb;pdb.set_trace()
            if "[ADVISOR" in decode_token:
                self.advisor_token_ids.append(token_id)
            if "[CLS]" == decode_token:
                self.cls_token_id = token_id
            if "[USR]" == decode_token:
                self.usr_token_id = token_id

    def split_sentence(self, cont):
        sentences = []
        sentence = []
        is_advisor = False
        for token_id in cont:
            if token_id == self.cls_token_id:
                continue
            elif token_id in self.advisor_token_ids:
                if len(sentence) > 0:
                    if is_advisor:
                        sentences.append([self.tokenizer.decode(sentence), "advi"])
                    else:
                        sentences.append([self.tokenizer.decode(sentence), "user"])
                    sentence = []
                else:
                    sentence = []
                is_advisor = True
            elif token_id == self.usr_token_id:
                if len(sentence) > 0:
                    if is_advisor:
                        sentences.append([self.tokenizer.decode(sentence), "advi"])
                    else:
                        sentences.append([self.tokenizer.decode(sentence), "user"])
                    sentence = []
                is_advisor = False
            else:
                sentence.append(token_id)
    
        if len(sentence) > 0:
            if is_advisor:
                sentences.append([self.tokenizer.decode(sentence), "advi"])
            else:
                sentences.append([self.tokenizer.decode(sentence), "user"])
    
        return sentences

# {
#       "id": 3,
#             "data": {
#                     "dialogue": [
#                               {
#                                           "text": "Hello",
#                                                   "author": "Speaker 1"
#                                                             },
#                           {
#                                       "text": "Hi",
#                                               "author": "Speaker 2"
#                                                         }
#                         ],
#                             "reply": [
#                                       {
#                                                   "text": "haha",
#                                                           "author": "Speaker 2"
#                                                                     }
#                             ]
#                                   },
#               "annotations": [],
#                 "predictions": []
# }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", help="path to the training data to convert.", type=str, default="./data/tri_train/train_v3_small.json")
    parser.add_argument("--model_path", help="path to the model. default: ./exp.distbert.b16/chkps_epoch-99/", type=str, default="./exp.distbert.b16/chkps_epoch-99/")
    parser.add_argument("--out_dir", help="path to the output directory", type=str, default="./label_studio/train_v3/")
    args = parser.parse_args()
    jf = open(args.train_data_path, "r")
    data = json.load(jf)
    jf.close()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    sentence_splitter = SentenceSplitter(tokenizer)

    id_cnt = 0

    file_name = "task0001.json"
    for i in tqdm(range(len(data["cont"]))):
        cont_id = data["cont"][i]
        sentences = sentence_splitter.split_sentence(cont_id)
        dialogue = [{"text": item[0], "author": item[1]} for item in sentences]
        out_data_list = []
        for p_index in data["other_cluster_resp_index_list"][i]:
            out_data = {}
            out_data["dialogue"] = dialogue      
            reply_sentences = sentence_splitter.split_sentence(data["resp"][p_index])
            reply = [{"text": item[0], "author": item[1]} for item in reply_sentences]
            out_data["reply"] = reply
            out_data_list.append(out_data)
        for p_index in data["rank_negative_resp_index_list"][i]:
            out_data = {}
            out_data["dialogue"] = dialogue      
            reply_sentences = sentence_splitter.split_sentence(data["resp"][p_index])
            reply = [{"text": item[0], "author": item[1]} for item in reply_sentences]
            out_data["reply"] = reply
            out_data_list.append(out_data)
        if i == 10:
            break
    with open(args.out_dir + file_name, "w") as fout:
        json.dump(out_data_list, fout)


if __name__ == "__main__":
    main()
