# test the samples sampled by the sampler
import argparse
import torch

from utils.n_p_sampler import RankingNagativeTripletSampler
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset
from msc_dataset import load_datasets, load_star_list, load_tri_dataset
from dialogbert_tokenizer import get_dialog_tokenizer
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, List, Dict
from rich.console import Console


console = Console()


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
    #                 tri_data["p_cluster_label_list"][index],
    #                 tri_data["n_cluster_label_list"][index],
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="The path to the data.", type=str, default="./data/tri_train/train_v6_small.json")
    parser.add_argument("--batch_size", help="batch size of the data.", type=int, default=16)
    args = parser.parse_args()

    dataset = load_tri_dataset(args.data_path)
    collator = batch_pad_collector_tri

    sampler = RankingNagativeTripletSampler(
            dataset    = dataset,
            batch_size = args.batch_size,
            n_size     = 16,
            shuffle    = True,
            seed       = 42,
            drop_last  = True,
            )

    dataloader = DataLoader(
            dataset     = dataset,
            # sampler     = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=misc.get_rank(), shuffle=True),
            sampler     = sampler,
            batch_size  = args.batch_size,
            num_workers = 4,
            drop_last   = True,
            collate_fn  = collator
            )
    
    have_n_p_cnt_list = []

    for batch in dataloader:
        have_n_p_cnt = 0
        cont_id, resp_id, mlm_id, cluster_label, p_index, n_index, index = batch
        for item_p_index_list, item_n_index_list in zip(p_index, n_index):
            p_cnt = 0
            n_cnt = 0
            for item_p_index in item_p_index_list:
                if item_p_index in index:
                    p_cnt += 1
            for item_n_index in item_n_index_list:
                if item_n_index in index:
                    n_cnt += 1
            have_n_p_cnt += min([n_cnt, p_cnt])
        have_n_p_cnt_list.append(have_n_p_cnt)

    over_cnt = 0
    for cnt in have_n_p_cnt_list:
        if cnt > 4: 
            over_cnt += 1
    console.print(over_cnt/len(have_n_p_cnt_list))


if __name__ == "__main__":
    main()
