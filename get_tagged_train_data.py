import json
import argparse

from tqdm import tqdm
from msc_dataset import MSCTokenizer
from dialogbert_tokenizer import get_dialog_tokenizer
from rich.console import Console

console = Console()


def format_tagged_data(tagged_data):
    formatted_data = []
    for data_item in tagged_data:
        context = data_item.pop("context")
        msc = []
        for character, msg_text in context:
            if character == "user-msg":
                from_user = True
            else:
                from_user = False
            msc.append([from_user, msg_text])
        data_item["msc"] = msc
        formatted_data.append(data_item)
    return formatted_data


def _gen_dataset(tagged_data, msc_tokenizer, window_size, resp_to_cluster_idx, resp_to_star_info=None, use_best_stars=False):
    dataset = []
    for data_item in tqdm(tagged_data):
        dataset_item = {"cont_id": None, "p_resp_id": [], "n_resp_id": [], "mlm_id": None, "p_resp_cluster_idx": [], "n_resp_cluster_idx": []}
        msc = data_item["msc"]
        origin_resp = data_item["origin_resp"]
        resp_list = data_item["resp"]
        cont_id = msc_tokenizer.gen_context_ids(msc, window_size=window_size, star=None)
        dataset_item["cont_id"] = cont_id
        origin_resp_id = msc_tokenizer.gen_resp_ids(origin_resp, star=None)
        cluster_idx = resp_to_cluster_idx[origin_resp]
        dataset_item["p_resp_id"].append(origin_resp_id)
        dataset_item["p_resp_cluster_idx"].append(cluster_idx)
        concat_id = cont_id + origin_resp_id[1:]
        dataset_item["mlm_id"] = concat_id[:msc_tokenizer.max_token_size]
        
        for resp, resp_tag in resp_list:
            if resp_tag == "p" and use_best_stars:
                if resp not in resp_to_star_info:
                    resp_tag = "n"
                else:
                    star_info_dict = resp_to_star_info[resp]
                    if 1 in [star_item[1] for star_item in star_info_dict.items()]:
                        resp_tag = "p"
                    else:
                        resp_tag = "n"
                
            cluster_idx = resp_to_cluster_idx[resp]
            if resp_tag == "p":
                resp_id = msc_tokenizer.gen_resp_ids(resp, star=None)
                dataset_item["p_resp_id"].append(resp_id)
                dataset_item["p_resp_cluster_idx"].append(cluster_idx)
            elif resp_tag == "n":
                resp_id = msc_tokenizer.gen_resp_ids(resp, star=None)
                dataset_item["n_resp_id"].append(resp_id)
                dataset_item["n_resp_cluster_idx"].append(cluster_idx)
        dataset.append(dataset_item)
    return dataset
        

def gen_train_dataset(tagged_data, msc_tokenizer, valid_ratio, window_size, resp_to_cluster_idx, resp_to_star_info=None, use_best_stars=False):
    valid_len = int(len(tagged_data) * valid_ratio)
    train_tagged_data = tagged_data[valid_len:]
    valid_tagged_data = tagged_data[:valid_len]
    print("Collating train data....")
    train_dataset = _gen_dataset(
            train_tagged_data,
            msc_tokenizer,
            window_size,
            resp_to_cluster_idx,
            resp_to_star_info=resp_to_star_info,
            use_best_stars=use_best_stars
            )
    print("Collating valid data....")
    valid_dataset = _gen_dataset(
            valid_tagged_data,
            msc_tokenizer,
            window_size,
            resp_to_cluster_idx,
            resp_to_star_info=resp_to_star_info,
            use_best_stars=use_best_stars
            )
    return train_dataset, valid_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tagged_data_path", help="", type=str, default="./data/tag_data/tagged_data.json")
    parser.add_argument("--window_size", help="", type=int, default=10)
    parser.add_argument("--valid_ratio", help="", type=float, default=0.1)
    parser.add_argument("--train_out_path", help="", type=str, default="./data/tagged_train/train_v1.json")
    parser.add_argument("--valid_out_path", help="", type=str, default="./data/tagged_train/valid_v1.json")
    parser.add_argument("--resp_cluster_info_path", help="", type=str, default="data/tag_data/resp_to_cluster_idx_v9.json")
    parser.add_argument("--resp_star_info_path", help="", type=str, default="./data/tag_data/resp_to_star_info.json")
    parser.add_argument("--use_best_stars", required=True, action="store_true")
    args = parser.parse_args()

    with open(args.tagged_data_path, "r") as jf:
        tagged_data = json.load(jf)

    with open(args.resp_cluster_info_path, "r") as jf:
        resp_to_cluster_idx = json.load(jf)

    with open(args.resp_star_info_path, "r") as jf:
        resp_to_star_info = json.load(jf)

    train_out_path = args.train_out_path
    valid_out_path = args.valid_out_path

    if args.use_best_stars:
        train_out_path = train_out_path[:-4] + "best_star.json"
        valid_out_path = valid_out_path[:-4] + "best_star.json"

    tagged_data = format_tagged_data(tagged_data)
    tokenizer = get_dialog_tokenizer("distilbert", "distilbert-base-uncased")
    msc_tokenizer = MSCTokenizer([], tokenizer)

    train_dataset, valid_dataset = gen_train_dataset(
            tagged_data,
            msc_tokenizer,
            args.valid_ratio,
            args.window_size,
            resp_to_cluster_idx,
            resp_to_star_info=resp_to_star_info,
            use_best_stars=args.use_best_stars
            )

    console.print("train out path: {}".format(train_out_path), style="green")
    console.print("valid out path: {}".format(valid_out_path), style="green")
    input()

    with open(train_out_path, "w") as fout:
        json.dump(train_dataset, fout)
    with open(valid_out_path, "w") as fout:
        json.dump(valid_dataset, fout)


if __name__ == "__main__":
    main()

