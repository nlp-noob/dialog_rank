import torch

from torch.utils.data import Sampler, DataLoader
from torch.utils.data.distributed  import DistributedSampler
from msc_dataset import load_tri_dataset
from train_biencoder_with_mlm_tri_sampler import batch_pad_collector_tri
from typing import Tuple, List, Dict
from torch.nn.utils.rnn import pad_sequence


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


class RankingNagativeTripletSampler(Sampler):
    def __init__(self, dataset, batch_size: int, n_size: int, shuffle: bool, seed: int = 0, drop_last: bool = False) -> None:
        self.dataset = dataset
        self.data_size = len(dataset)
        self.batch_size = batch_size
        self.n_size = n_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last
        self.num_batch = int(len(self.dataset) / batch_size)
        self.full_size = self.num_batch * self.batch_size

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if self.drop_last:
            indices = indices[:self.full_size]
        elif full_size != self.data_size:
            pad_full_size = self.full_size + self.batch_size
            pad_indices = indices[:(pad_full_size - self.data_size)]
            indices.extend(pad_indices)

        # the head indices
        batched_indices_list = []  
        unused_indices_cnt = {}
        for indices_index, indice in enumerate(indices):
            if indices_index % self.batch_size == 0:
                batched_indices_list.append([indice])
            else: 
                if indice not in unused_indices_cnt:
                    unused_indices_cnt[indice] = 0
                unused_indices_cnt[indice] += 1

        for batch_index in range(len(batched_indices_list)):
            n_cnt = 0
            batch_indices = batched_indices_list[batch_index]
            while(len(batch_indices) < self.batch_size):
                n_indices = self.dataset[batch_indices[-1]][2][3]

                have_n_indice = False
                for n_indice in n_indices:
                    if n_indice in unused_indices_cnt and unused_indices_cnt[n_indice] > 0:
                        batch_indices.append(n_indice)
                        unused_indices_cnt[n_indice] = unused_indices_cnt[n_indice] - 1
                        have_n_indice = True
                        break

                if not have_n_indice:
                    for indice in unused_indices_cnt:
                        if indice in unused_indices_cnt and unused_indices_cnt[indice] > 0:
                            batch_indices.append(indice)
                            unused_indices_cnt[indice] = unused_indices_cnt[indice] - 1
                            break
            batched_indices_list[batch_index] = batch_indices
        indices = []
        for batch_indices in batched_indices_list:
            indices.extend(batch_indices)
        return iter(indices)


    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def test():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", help="The path to the data to test the sampler.", type=str, default="./data/tri_train/train_v4_small.json")
    parser.add_argument("--batch_size", help="The batch size of the sampler and the dataloader.", type=int, default=16)
    parser.add_argument("--n_size", help="The est n size in a batch.", type=int, default=3)
    args = parser.parse_args()

    train_dataset = load_tri_dataset(
            args.test_data_path,
            )

    sampler = RankingNagativeTripletSampler(
            dataset    = train_dataset,
            batch_size = args.batch_size,
            n_size     = args.n_size,
            shuffle    = True,
            seed       = 42,
            drop_last  = True,
            )

    train_dataloader = DataLoader(
            dataset    = train_dataset,
            sampler    = sampler,
            batch_size = args.batch_size,
            collate_fn = batch_pad_collector_tri,
            )

    train_dataloader.sampler.set_epoch(0)
    for batch in train_dataloader:
        print(batch)
        input()


if __name__ == "__main__":
    test()
