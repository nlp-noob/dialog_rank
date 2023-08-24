import json
import argparse

from sentence_transformers import SentenceTransformer, util
from deduplicate_text import levenshtein_dist_rate
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resp_cluster_info_path", help="", type=str, default="./data/tag_data/resp_to_cluster_idx_v9.json")
    parser.add_argument("--ritual_sentences_path", help="", type=str, default="./data/ritual_sentences.json")
    parser.add_argument("--leven_thresh", help="", type=float, default=0.85)
    parser.add_argument("--sim_thresh", help="", type=float, default=0.75)
    parser.add_argument("--out_path", help="", type=str, default="./data/resp_to_ritual.json")
    args = parser.parse_args()

    ritual_cluster_set = set([])
    resp_to_ritual = {}

    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open(args.resp_cluster_info_path, "r") as jf:
        resp_to_cluster_idx = json.load(jf)

    with open(args.ritual_sentences_path, "r") as jf:
        ritual_sentences = json.load(jf)

    resp_list = list(resp_to_cluster_idx.keys())
    # resp_list = resp_list[:500]

    print("sentence len: {}".format(len(resp_to_cluster_idx)))
    print("embedding....")
    resp_embed = model.encode(resp_list, show_progress_bar=True, convert_to_tensor=True)
    ritual_embed = model.encode(ritual_sentences, show_progress_bar=True, convert_to_tensor=True)

    # debug
    print("computing scores...")
    scores = util.cos_sim(resp_embed, ritual_embed)
    # for resp_idx, resp in enumerate(resp_list):
    #     for ritual_idx, ritual_sentence in enumerate(ritual_sentences):
    #         embed1 = model.encode(resp, convert_to_tensor=True)
    #         embed2 = model.encode(ritual_sentence, convert_to_tensor=True)
    #         score = util.cos_sim(embed1, embed2).tolist()[0][0]
    #         if score != scores[resp_idx][ritual_idx].item():
    #             print(scores[resp_idx][ritual_idx].item())
    #             print(score)
    #             input()

    print("analysing scores...")
    for resp_idx, resp in enumerate(tqdm(resp_list)):
        cluster_idx = resp_to_cluster_idx[resp]
        if cluster_idx in ritual_cluster_set:
            continue
        for ritual_idx, ritual_sentence in enumerate(ritual_sentences):
            leven_dist = levenshtein_dist_rate(resp, ritual_sentence)
            sim_score = scores[resp_idx][ritual_idx].item()
            if leven_dist > args.leven_thresh or sim_score > args.sim_thresh:
                ritual_cluster_set.add(cluster_idx)

    ritual_cnt = 0

    for resp, cluster_idx in resp_to_cluster_idx.items():
        if cluster_idx in ritual_cluster_set:
            resp_to_ritual[resp] = 1
            ritual_cnt += 1
        else:
            resp_to_ritual[resp] = 0

    print("ritual sentence ratio: {}".format(ritual_cnt / len(resp_to_ritual)))

    with open(args.out_path, "w") as fout:
        json.dump(resp_to_ritual, fout)

    
if __name__ == "__main__":
    main()

