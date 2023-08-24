import argparse
import glob
import logging
import os
import pickle
import random
import json

import numpy as np
from tqdm import tqdm
import torch

from transformers import  AutoConfig, AutoModel, AutoTokenizer
from msc_dataset import load_datasets, load_msc_bert_test_datasets
from torch.utils.tensorboard import SummaryWriter
from tod_model import MLMBiencoder
from msc_dataset import MSCTokenizer
from dialogbert_tokenizer import get_dialog_tokenizer

from tod_model import MLMBiencoder



def get_resp_list(data_dir):
    data_len = len(os.listdir(data_dir))
    resp_set = set([])
    print("collating resp_list....")
    for i in tqdm(range(data_len)):
        with open(data_dir + "flatten_{}.json".format(i + 1), "r") as jf:
            data_item = json.load(jf)
            for resp_item in data_item["resp"]:
                resp_set.add(resp_item[0])
            resp_set.add(data_item["origin_resp"])
    return list(resp_set)


def get_msc_list(data_dir, sample_len, window_size):
    msc_list = []
    data_len = len(os.listdir(data_dir))
    data_index_list = list(range(data_len))[-sample_len:]
    print("collating msc_list....")
    for data_index in tqdm(data_index_list):
        with open(data_dir + "flatten_{}.json".format(data_index), "r") as jf:
            data_item = json.load(jf)
            raw_msc = data_item["context"]
            msc = []
            for character_type, msg_text in raw_msc:
                if character_type == "user-msg":
                    from_user = True
                else:
                    from_user = False
                msc.append([from_user, msg_text])
            msc_list.append(msc)
    return msc_list


def get_scored(args):
    resp_list = get_resp_list(args.data_dir)
    resp_list = resp_list[:100]
    msc_list = get_msc_list(args.data_dir, args.sample_len, args.window_size)

    tokenizer = get_dialog_tokenizer("distilbert", "distilbert-base-uncased")
    msc_tokenizer = MSCTokenizer([], tokenizer)
    model = MLMBiencoder(args.model_path, tokenizer, mlm_probability=0.15, mlm=False)
    device = args.device
    model.to(device)
    model.eval()

    result_dict = {"data": [], "resp_list": resp_list}

    list_resp_norm = []

    print("Collating the resp norm matrix...")

    for resp in tqdm(resp_list):
        resp_ids = msc_tokenizer.gen_resp_ids(resp, star=None)
        resp_tensor = torch.tensor(resp_ids).to(device).unsqueeze(0)
        with torch.no_grad():
            _, hid_resp = model.encoder_forward(
                    input_ids = resp_tensor,
                    attention_mask = resp_tensor > 0,
                    )
            hid_resp_norm = torch.nn.functional.normalize(hid_resp, p=2, dim=1)
        list_resp_norm.append(hid_resp_norm)
    cur_resp_norm = torch.cat(list_resp_norm, dim=0)

    print("doing retrieval....")
    for msc in tqdm(msc_list):
        cont_ids = msc_tokenizer.gen_context_ids(msc, window_size=args.window_size, star=None)
        with torch.no_grad():
            cont_tensor = torch.tensor(cont_ids).to(device).unsqueeze(0)
            _, hid_cont = model.encoder_forward(
                    input_ids = cont_tensor,
                    attention_mask = cont_tensor > 0,
                    )
            hid_cont_norm = torch.nn.functional.normalize(hid_cont, p=2, dim=1)
            score = torch.matmul(hid_cont_norm, cur_resp_norm.transpose(1, 0))
            score_list = score.tolist()[0]
            round_score_list = [round(score, 4) for score in score_list]

            result_item = {"msc": msc, "score_list": round_score_list}
            result_dict["data"].append(result_item)
    return result_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./exp.distbert.b16.tri.tagged/chkps_epoch-20/', type=str, help="The path to the model.")
    parser.add_argument("--data_dir", default="./data/tag_data/splits/flatten/")
    parser.add_argument("--cluster_info_path", default="", type=str, help="./data/tag_data/resp_to_cluster_idx_v9.json")
    parser.add_argument("--sample_len", default=500, type=int, help="")
    parser.add_argument("--window_size", default=10, type=int, help="")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument('--gpus', default='1', type=str)
    parser.add_argument("--out_dir", default="./data/tag_data/split_demo_v1_best_star/")

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    result_dict = get_scored(args)

    with open(args.out_dir + "resp_list.json", "w") as fout:
        json.dump(result_dict["resp_list"], fout)

    for i, data_item in enumerate(result_dict["data"]):
        with open(args.out_dir + "split_{}.json".format(i), "w") as fout:
            json.dump(data_item, fout)

        
if __name__ == "__main__":
    main()
