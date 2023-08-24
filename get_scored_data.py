# deprecated use get_train_data.py
import argparse
import glob
import logging
import os
import pickle
import random

import numpy as np
from tqdm import tqdm
import torch

from transformers import  AutoConfig, AutoModel, AutoTokenizer
from msc_dataset import load_datasets, load_msc_bert_test_datasets
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from tod_model import MLMBiencoder


console = Console()


def get_n_p_list(
        rank_list,
        rank_value_list,
        cluster_label,
        resp_index_to_cluster_label,
        cluster_label_to_resp_index):
    p_cluster = set([])
    n_cluster = set([])
    for resp_index, (rank, rank_value) in enumerate(zip(rank_list, rank_value_list)):
        if rank_value > 0.55:
            p_cluster.add(resp_index_to_cluster_label[resp_index])
        elif rank_value < -0.1:
            n_cluster.add(resp_index_to_cluster_label[resp_index])
    return list(p_cluster), list(n_cluster)


def get_scored_data(args, dataset):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = MLMBiencoder(args.model_path, tokenizer, mlm_probability=0.15, mlm=False)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    data_size = len(dataset)

    list_resp_norm = []
    
    scored_data = {}

    console.print("Collating the resp norm matrix...", style="green")
    for i in tqdm(range(data_size)):
        _, resp, _, _ = dataset[i]
        resp = torch.tensor(resp).to(device).unsqueeze(0)

        with torch.no_grad():
            _, hid_resp = model.encoder_forward(
                    input_ids = resp,
                    attention_mask = resp > 0,
                    )
            hid_resp_norm = torch.nn.functional.normalize(hid_resp, p=2, dim=1)
        list_resp_norm.append(hid_resp_norm)
    cur_resp_norm = torch.cat(list_resp_norm, dim=0)

    console.print("calculating scores and acc...", style="green")
    # scores = []

    cluster_label_to_resp_index = {}
    resp_index_to_cluster_label = {}
    console.print("Getting cluster_label_to_resp_index ...", style="green")
    for i in tqdm(range(data_size)):
        _, _, _, cluster_label = dataset[i]
        if cluster_label not in cluster_label_to_resp_index:
            cluster_label_to_resp_index[cluster_label] = []
        cluster_label_to_resp_index[cluster_label].append(i)
        resp_index_to_cluster_label[i] = cluster_label

    for i in tqdm(range(data_size)):
        cont, resp, mlm, cluster_label = dataset[i]

        scored_data[i] = {}
        scored_data[i]["cont"] = cont
        scored_data[i]["resp"] = resp
        scored_data[i]["cluster_label"] = cluster_label

        with torch.no_grad():
            cont = torch.tensor(cont).to(device).unsqueeze(0)
            _, hid_cont = model.encoder_forward(
                    input_ids = cont,
                    attention_mask = cont > 0,
                    )
            hid_cont_norm = torch.nn.functional.normalize(hid_cont, p=2, dim=1)
            score = torch.matmul(hid_cont_norm, cur_resp_norm.transpose(1, 0))
            # scores.append(score.tolist()[0])
            score_list = score.tolist()[0]
            rank_value_list, rank_list = score.topk(dim=-1, k=len(score[0]))
            rank_value_list = rank_value_list[0].tolist()
            rank_list = rank_list[0].tolist()
            p_cluster_list, n_cluster_list = get_n_p_list(rank_list, rank_value_list, cluster_label, resp_index_to_cluster_label, cluster_label_to_resp_index) 
            scored_data[i]["p_cluster_list"] = p_cluster_list
            scored_data[i]["n_cluster_list"] = n_cluster_list
            # scored_data[i]["score"] = score.tolist()[0]

    return scored_data 

def get_tagged_data(args, scored_data, scores):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    device = torch.device(args.device)
    console.print("Calculating topk values...", style="green")
    topk_values, topk_indexes = torch.tensor(scores).to(device).topk(dim=-1, k=args.topk)
    minus_scores = torch.tensor(scores) * (-1)
    smallk_values, smallk_indexes = torch.tensor(minus_scores).topk(dim=-1, k=args.topk)
    smallk_values = smallk_values * (-1)

    console.print("Generating the N and P data...", style="green")

    data_size = len(scored_data)

    for i in tqdm(range(data_size)):
        topk_idx_list = topk_indexes[i].tolist()
        smallk_idx_list = smallk_indexes[i].tolist()
        for topk_idx, smallk_idx in zip(topk_idx_list, smallk_idx_list):
            if topk_idx != i:
                cont = scored_data[i]["cont"]
                resp = scored_data[topk_idx]["resp"]
                print("--"*20)
                print(tokenizer.decode(cont))
                print(tokenizer.decode(resp))
                input()
            if smallk_idx != i:
                scored_data[i]["N"].append(smallk_idx)


def write_data(out_path, scored_data):
    out_dict = {
        "cont": [],
        "resp": [],
        "cluster_label": [],
        "p_cluster_list": [],
        "n_cluster_list": [],
    }
    for i in range(len(scored_data)):
        out_dict["cont"].append(scored_data[i]["cont"])
        out_dict["resp"].append(scored_data[i]["resp"])
        out_dict["cluster_label"].append(scored_data[i]["cluster_label"])
        out_dict["n_cluster_list"].append(scored_data[i]["n_cluster_list"])
        out_dict["p_cluster_list"].append(scored_data[i]["p_cluster_list"])

    import json
    json_str = json.dumps(out_dict, indent=2)
    fout = open(out_path, "w")
    fout.write(json_str)
    fout.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./exp.distbert.b16/chkps_epoch-99', type=str, help="The path to the model.")
    parser.add_argument('--order_file', default='./data/orders.star1-2.1.json', type=str, help="The order to be tag")
    parser.add_argument('--fp16', action="store_true", default=False, help="user fp16")
    parser.add_argument('--device', type=str, default='cuda:0', help="default: cuda:0")
    parser.add_argument('--max_sess_size', help='gen msc segmentation with max session size, default=15', type=int, default=15)
    parser.add_argument('--min_sess_size', help='gen msc segmentation with min session size, default=5', type=int, default=5)
    parser.add_argument('--first_orders', help='Only take the first few orders between the same user and advisor, default=5', type=int, default=5)
    parser.add_argument('--topk', help="the topk and the smallk to be tagged.", type=str, default=10)
    parser.add_argument('--p_score_thresh', help="the threshold of the possitive sample.", type=float, default=0.6)
    parser.add_argument('--scored_data_out_path', help="The output path of the scored_data.", type=str, default="./tmp/scored_data.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # just the whole dataset
    console.print("Loading the dataset...", style="green")
    # train_dataset, val_dataset, star_list = load_datasets(
    dataset = load_msc_bert_test_datasets(
            args.order_file,
            first_orders=args.first_orders,
            max_sess_size=args.max_sess_size,
            min_sess_size=args.min_sess_size,
            tokenizer=tokenizer,
            data_type='msc_mlm_biencoder_cat',
    )
    # dataset = val_dataset

    import json
    scored_data = get_scored_data(args, dataset)
    import pdb;pdb.set_trace()
    write_data(args.scored_data_out_path, scored_data)

    # get_tagged_data(args, scored_data, scores)

if __name__ == "__main__":
    main()
