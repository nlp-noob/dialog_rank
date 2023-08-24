import sys
import os
import random
import argparse
import json
import numpy as np
import time
import re
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util

from msc_dataset import load_msc

sys.path.insert(0, '..')
from deduplicate_text import cluster_texts


class Cluster(object):
    def __init__(self, model_name_or_path, batch_size=16, limit=60000):
        self.model = SentenceTransformer(model_name_or_path)
        self.batch_size = batch_size
        self.limit=limit

    def _cluster(self, corpus, threshold=0.6, min_community_size=3, init_max_size=1000, limit=False):
        if limit:
            corpus = corpus[:self.limit]

        corpus_embeddings = self.model.encode(corpus, batch_size=self.batch_size, show_progress_bar=True, convert_to_tensor=True)

        if init_max_size > len(corpus) // 2:
            init_max_size = len(corpus) // 2
        return util.community_detection(corpus_embeddings, threshold=threshold, min_community_size=min_community_size, batch_size=init_max_size)

    def cluster(self, corpus, threshold=0.6, min_community_size=3, init_max_size=1000):
        if self.limit < 1:
            return self._cluster(corpus, threshold, min_community_size, init_max_size)

        if len(corpus) <= self.limit:
            return self._cluster(corpus, threshold, min_community_size, init_max_size)

        # sub batch
        batch_idxs  = list(range(0, len(corpus), self.limit))
        batch_clusters = []
        cluster_merge_corpus = []
        cluster_merge_corpus_idx = []
        for bi in batch_idxs:
            batch_corpus = corpus[bi: bi+self.limit]
            batch_rst = self._cluster(batch_corpus, threshold, min_community_size, init_max_size)
            batch_clusters.append(batch_rst)

            for cluster in batch_rst:
                txts = [batch_corpus[i] for i in cluster]
                txts_idx = [bi + i for i in cluster]
                txts_size = [len(i) for i in txts]
                mid_i = np.argsort(txts_size)[len(txts_size) // 2]
                cluster_merge_corpus.append(txts[mid_i])
                cluster_merge_corpus_idx.append(txts_idx[mid_i])

        merge_clusters = self._cluster(cluster_merge_corpus, threshold, min_community_size, init_max_size)
        # merge all the clusters
        merge_idx_dict = {}
        for idx, merge_cluster in enumerate(merge_clusters):
            global_idxs = [cluster_merge_corpus_idx[i] for i in merge_cluster]
            for gid in global_idxs:
                merge_idx_dict[gid] = idx

        clusters = []
        merged_sets = {}
        merge_global_idx = 0
        for bi, batch_cluster in zip(batch_idxs, batch_clusters):
            for cluster in batch_cluster:
                global_idxs = [bi + i for i in cluster]
                merge_idx = cluster_merge_corpus_idx[merge_global_idx]
                merge_global_idx += 1

                if merge_idx not in merge_idx_dict:
                    # not merged
                    clusters.append(global_idxs)
                    continue

                merged_cluster_idx = merge_idx_dict[merge_idx]
                merged_sets.setdefault(merged_cluster_idx, []).extend(global_idxs)

        # append merged sets
        for merged_set in merged_sets.values():
            clusters.append(merged_set)

        return clusters


def load_advisor_msgs(order_file,
        msc_cnt_thresh=5,
        first_orders=5,
        first_msc_thresh=0):
    mscs, stars, infos = load_msc(order_file, msc_cnt_thresh, first_orders, first_msc_thresh)

    advisor_msgs = []
    for msc_idx, msc in enumerate(mscs):
        advisor_msgs.extend([(msc_idx, i, msg.message) for i, msg in enumerate(msc) if not msg.sender])
    return advisor_msgs, mscs, stars


def print_cluster(groups):
    sorted_groups = sorted(groups, key=lambda x: -len(x))
    for msgs in sorted_groups:
        print('=' * 10 + f'{len(msgs):04d}' + '=' * 10)
        print(('\n' + '--'*20 +'\n').join([i[2] for i in sorted(msgs, key=lambda x: x[2])]))


def cluster_advisor_msgs(engine, msgs, deduplicate_thresh=0.85, sim_thresh=0.6):
    # msgs: [(msc_idx, msg_idx, txt)]
    txt_key = lambda x: x[2]
    hamming_clusters = cluster_texts(msgs, deduplicate_thresh, text_key=txt_key)

    if False:
        # debug
        print('**' * 20 + 'hamming deduplicate cluster' + '**' * 20)
        print_cluster(hamming_clusters)

    dedup_txts, dedup_idxs = [], []
    dedup_group_idxs = list(range(len(hamming_clusters)))
    for cluster in hamming_clusters:
        # choose median length txt
        txt_size = [len(txt_key(i)) for i in cluster]
        si = np.argsort(txt_size)
        median_idx = si[len(cluster)//2]
        txt = txt_key(cluster[median_idx])
        dedup_txts.append(txt)
        dedup_idxs.append(median_idx)
        
    # shuffle dedup lists
    tmp = list(zip(dedup_txts, dedup_idxs, dedup_group_idxs))
    random.shuffle(tmp)
    dedup_txts, dedup_idxs, dedup_group_idxs = zip(*tmp)
    sim_clusters = engine.cluster(dedup_txts, threshold=sim_thresh)

    if False:
        print('**' * 20 + 'sim cluster' + '**' * 20)
        sim_groups = []
        for cluster in sim_clusters:
            cur_group = []
            for i in cluster:
                gidx = dedup_group_idxs[i]
                midx = dedup_idxs[i]
                msg = hamming_clusters[gidx][midx]
                cur_group.append(msg)
            sim_groups.append(cur_group)
        print_cluster(sim_groups)

    # merge deduplicate cluster and sim clusters
    final_groups = []
    gidx_set = set()
    for cluster in sim_clusters:
        cur_group = []
        for i in cluster:
            gidx = dedup_group_idxs[i]
            gidx_set.add(gidx)
            cur_group.extend(hamming_clusters[gidx])
        final_groups.append(cur_group)
    for gidx, hamming_group in enumerate(hamming_clusters):
        if gidx in gidx_set:
            continue
        final_groups.append(hamming_group)

    if True:
        print_cluster(final_groups)

    return final_groups


def main():
    parser = argparse.ArgumentParser(description='semantic question search')
    parser.add_argument('-s', '--source', help='the source log json path', type=str, required=True)
    parser.add_argument('-m', '--model', help='the sentence-transformer model path', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--sim_thresh', help='cluster siminarity threshold', type=float, default=0.6)
    parser.add_argument('--hamming_thresh', help='hamming siminarity threshold', type=float, default=0.85)
    parser.add_argument('-l', '--limit', help='limit the cluster embeddings', type=int, default=60000)
    parser.add_argument('--gpus', default='0', type=str)
    args = parser.parse_args()
    cluster_result_path = args.source[:-4] + "cluster.json"
    cluster_label_path = args.source[:-4] + "label.json"
    print("The cluster result will be saved in {}".format(cluster_result_path))
    print("The cluster label will be saved in {}".format(cluster_label_path))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    engine = Cluster(args.model, limit=args.limit)

    msgs, mscs, stars = load_advisor_msgs(args.source)
    clusters = cluster_advisor_msgs(engine, msgs, args.hamming_thresh, args.sim_thresh)

    with open(cluster_result_path, 'w+') as fout:
        json.dump(clusters, fout)

    # transfer cluster label [[(msc_idx, cluster_label)], ...]
    cluster_label = [[] for _ in range(len(mscs))]
    for label, group in enumerate(clusters):
        for mscs_idx, msc_idx, txt in group:
            cluster_label[mscs_idx].append([msc_idx, label])

    for labels in cluster_label:
        labels.sort(key=lambda x: x[0])

    with open(cluster_label_path, 'w+') as fout:
        json.dump(cluster_label, fout)


if __name__ == '__main__':
    main()

