from dialogbert_tokenizer import get_dialog_tokenizer
from msc_dataset import load_datasets, load_star_list, load_tri_dataset
from torch.utils.data.distributed import DistributedSampler
from utils import misc, xlog
from typing import Tuple, List, Dict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset

import torch
import random
import numpy as np


def batch_pad_collector_tri(batch: List[Tuple[List[int], List[int], Tuple[List[int], List[int], List[int], int, int]]], pad_val=0):
    # val_dataset.append((
    #             tri_data["cont_id"][index],
    #             tri_data["resp_id"][index],
    #             (
    #                 tri_data["cluster_label"][index],
    #                 tri_data["mlm_id"][index],
    #                 tri_data["p_index"][index],
    #                 tri_data["n_index"][index],
    #                 tri_data["index"][index],
    #                 tri_data["p_cluter_label_list"][index],
    #                 tri_data["n_cluter_label_list"][index],
    #             )))
    items = [[], [], [], [], [], [], []]
    batch_size = len(batch)
    for i in range(batch_size):
        # cont, resp, mlm, cluster_label, p_index, n_index, index
        items[0].append(torch.tensor(batch[i][0])) # cont
        items[1].append(torch.tensor(batch[i][1])) # resp
        items[2].append(torch.tensor(batch[i][2][1])) # mlm
        items[3].append(torch.tensor(batch[i][2][0])) # cluster label
        items[4].append(batch[i][2][2]) # p_index
        items[5].append(batch[i][2][3]) # n_index
        items[6].append(batch[i][2][4]) # index

    items[:3] = [pad_sequence(item, batch_first=True, padding_value=pad_val) for item in items[:3]]
    items[3] = torch.tensor(items[3])
    return items
    
def batch_pad_collector(batch: List[Tuple[List[int], List[int], int]], pad_val=0):
    item_size = len(batch[0])

    items = [[torch.tensor(item[i]) for item in batch] \
                for i in range(item_size)]

    try:
        items[:-1] = [pad_sequence(item, batch_first=True, padding_value=pad_val) for item in items[:-1]]
        items[-1] = torch.tensor(items[-1])
    except:
        import pdb; pdb.set_trace()

    return items


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    tokenizer = get_dialog_tokenizer("distilbert", "distilbert-base-uncased")
    pre_star_list = load_star_list("data/star.list.220802.1-2")

    set_seed(42)

    train_dataset_origin, val_dataset_origin, star_list = load_datasets(
            "./data/orders.star1-2.1.1.json",
            first_orders=5,
            max_sess_size=15,
            min_sess_size=5,
            tokenizer=tokenizer,
            data_type='msc_mlm_biencoder_cat',
            star_list=pre_star_list,
    )

    train_dataset  = load_tri_dataset(
            "data/tri_train/train_v6_small.json"
    )

    val_dataset  = load_tri_dataset(
            "data/tri_train/valid_v6_small.json"
    )
    train_data_size = len(train_dataset)

    diff_cnt = 0

    trn_loader_origin = DataLoader(
            dataset     = train_dataset_origin,
            sampler     = DistributedSampler(train_dataset_origin, num_replicas=1, rank=misc.get_rank(), shuffle=True),
            # sampler     = SequentialSampler(train_dataset_origin),
            batch_size  = 16,
            # num_workers = 4,
            drop_last   = True,
            collate_fn  = batch_pad_collector
            )
    trn_loader = DataLoader(
            dataset     = train_dataset,
            sampler     = DistributedSampler(train_dataset, num_replicas=1, rank=misc.get_rank(), shuffle=True),
            # sampler     = SequentialSampler(train_dataset),
            batch_size  = 16,
            # num_workers = 4,
            drop_last   = True,
            collate_fn  = batch_pad_collector_tri
            )

    for i in range(train_data_size):
        cont_id, resp_id, (cluster_label, mlm_id, _, _, _) =  train_dataset[i]
        cont_id_origin, resp_id_origin, mlm_id_origin, cluster_label_origin = train_dataset_origin[i]
        # print("--"*20)
        # print("cont")
        # print(cont_id)
        # print(cont_id_origin)
        # print(tokenizer.decode(cont_id))
        # print(tokenizer.decode(cont_id_origin))
        # print("--"*20)
        # print("resp")
        # print(resp_id)
        # print(resp_id_origin)
        # print(tokenizer.decode(resp_id))
        # print(tokenizer.decode(resp_id_origin))
        # input()
        if cluster_label != cluster_label_origin:
            import pdb;pdb.set_trace()

        if cont_id != cont_id_origin or resp_id != resp_id_origin or mlm_id != mlm_id_origin:
            diff_cnt += 1
    print("train diff rate {}".format(diff_cnt / train_data_size))

    diff_cnt = 0

    valid_data_size = len(val_dataset)
    for i in range(valid_data_size):
        cont_id, resp_id, (_, mlm_id, _, _, _) =  val_dataset[i]
        cont_id_origin, resp_id_origin, mlm_id_origin, _ = val_dataset_origin[i]
        if cont_id != cont_id_origin or resp_id != resp_id_origin or mlm_id != mlm_id_origin:
            diff_cnt += 1
    print("valid diff rate {}".format(diff_cnt / valid_data_size))

    for batch_origin, batch in zip(trn_loader_origin, trn_loader):
        cont_id_origin, resp_id_origin, mlm_id_origin, cluster_label_origin = batch_origin
        cont_id, resp_id, mlm_id, cluster_label, p_index, n_index, index = batch
        for i in range(len(cont_id)):
            if cont_id_origin[i].shape != cont_id[i].shape:
                print("--"*20)
                print(cont_id[i])
                print(resp_id[i])
                print(cont_id_origin[i])
                print(resp_id_origin[i])


if __name__ == "__main__":
    main()
