import json
import argparse
import os

from tqdm import tqdm
from rich.console import Console


console = Console()


def get_resp_with_star_info(args, star_info):
    data_len = len(os.listdir(args.raw_data_dir))
    best_star_cnt = 0
    resp_to_star_info = {}
    for i in tqdm(range(data_len)):
        data_index = i + 1
        with open("./data/tag_data/splits/flatten/flatten_{}.json".format(data_index), "r") as jf:
            is_best = False
            data = json.load(jf)
            star_id = data["star_id"]
            if star_id in star_info:
                is_best = True
                best_star_cnt += 1
            context = data["context"]
            context.append(["advisor-msg", data["origin_resp"]])
            msc = []
            for msg_character, msg_text in context:
                if msg_character == "user-msg":
                    continue
                if msg_text not in resp_to_star_info:
                    resp_to_star_info[msg_text] = {}
                if is_best:
                    resp_to_star_info[msg_text][star_id] = 1
                else:
                    resp_to_star_info[msg_text][star_id] = 0
    print("best star ratio: {}".format(best_star_cnt / data_len))
    return resp_to_star_info
 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--star_info_path", help="", type=str, default="./data/best_star_id_to_name.json")
    parser.add_argument("--raw_data_dir", help="", type=str, default="./data/tag_data/splits/flatten/")
    parser.add_argument("--cluster_path", help="", type=str, default="data/tag_data/resp_to_cluster_idx_v9.json")
    parser.add_argument("--out_path", help="", type=str, default="./data/tag_data/resp_to_star_info.json")
    args = parser.parse_args()

    with open(args.star_info_path, "r") as jf:
        star_info = json.load(jf)

    with open(args.cluster_path, "r") as jf:
        resp_to_cluster_idx = json.load(jf)

    resp_to_star_info = get_resp_with_star_info(args, star_info)

    # check
    for resp in tqdm(resp_to_cluster_idx):
        if resp not in resp_to_star_info:
            raise(ValueError, "not matched")

    with open(args.out_path, "w") as fout:
        json.dump(resp_to_star_info, fout)


if __name__ == "__main__":
    main()
