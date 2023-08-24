import argparse
import json

from deduplicate_text import levenshtein_dist_rate

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="", type=str, default="data/tag_data/resp_to_cluster_idx_v9.json")
    parser.add_argument("--max_len_bias", help="", type=int, default=5)
    parser.add_argument("--leven_thresh", help="", type=float, default=0.84)
    parser.add_argument("--out_path", help="", type=str, default="./data/tag_data/resp_to_cluster_idx.merge.json")
    args = parser.parse_args()

    with open(args.data_path, "r") as jf:
        resp_to_cluster_idx = json.load(jf)

    cluster_idx_to_resp_list = {}

    merge_cluster_idx_list = []

    for resp, cluster_idx in resp_to_cluster_idx.items():
        if cluster_idx not in cluster_idx_to_resp_list:
            cluster_idx_to_resp_list[cluster_idx] = []
        cluster_idx_to_resp_list[cluster_idx].append(resp)

    cluster_idx_to_resp_list_items = list(cluster_idx_to_resp_list.items())

    for item_idx_a, (cluster_idx_a, resp_list_a) in enumerate(tqdm(cluster_idx_to_resp_list_items)):
        resp_list_a = sorted(resp_list_a, key=lambda x:len(x))

        for item_idx_b in range(item_idx_a, len(cluster_idx_to_resp_list_items)):
            cluster_idx_b, resp_list_b = cluster_idx_to_resp_list_items[item_idx_b]
            if cluster_idx_a == cluster_idx_b:
                continue
            resp_list_b = sorted(resp_list_b, key=lambda x:len(x))

        found_dup = False
        for a_resp in resp_list_a:
            for b_resp in resp_list_b:
                leven_dist = levenshtein_dist_rate(a_resp, b_resp)
                if leven_dist > args.leven_thresh:
                    print("##"*30)
                    print(cluster_idx_a)
                    print(resp_list_a[:5])
                    print("--"*20)
                    print(cluster_idx_b)
                    print(resp_list_b[:5])
                    merge_cluster_idx_list.append([cluster_idx_a, cluster_idx_b])
                    found_dup = True
                    break
            if found_dup:
                break

    for cluster_idx_pair in merge_cluster_idx_list:
        cluster_idx_to_resp_list[cluster_idx_pair[0]].extend(cluster_idx_to_resp_list[cluster_idx_pair[1]])
        cluster_idx_to_resp_list.pop(cluster_idx_pair[1])

    merge_resp_to_cluster_idx = {}
    for idx, (cluster_idx, resp_list) in cluster_idx_to_resp_list.items():
        for resp in resp_list:
            merge_resp_to_cluster_idx[resp] = idx

    with open(args.out_path, "w") as fout:
        json.dump(merge_resp_to_cluster_idx, fout)


if __name__ == "__main__":
    main()

