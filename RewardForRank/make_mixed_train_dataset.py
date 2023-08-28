import argparse
import json

from utils.text_process
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--select_data_path", help="", type=str, default="./data/tag_data_with_info_15000.json")
    parser.add_argument("--select_data_path", help="", type=str, default="./data/tag_data_with_info_15000.json")
    parser.add_argument("--order_data_path", help="", type=str, default="./data/order_data/order15000.json")
    parser.add_argument("--val_ratio", help="", type=float, default=0.05)
    parser.add_argument("--split_ratio", help="", type=float, default=1)
    parser.add_argument("--neg_cnt", help="", type=int, default=3)
    parser.add_argument("--window_size", help="", type=int, default=5)
    parser.add_argument("--dev_out_path", help="", type=str, default="./data/train_data/dedup_valid_v1.full.json")
    parser.add_argument("--train_out_path", help="", type=str, default="./data/train_data/dedup_train_v1.full.json")
    args = parser.parse_args()

    with open(args.cluster_data_path, "r") as jf:
        pass


if __name__ == "__main__":
    main()
