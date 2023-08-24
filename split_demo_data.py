import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="", default="./data/tag_data/rank_demo_data_wan.json")
    parser.add_argument("--split_data_dir", help="", default="./data/tag_data/split_demo_data_wan/")
    args = parser.parse_args()
    with open(args.data_path, "r") as jf:
        data = json.load(jf)

    with open(args.split_data_dir + "resp_list.json", "w") as fout:
        json.dump(data["resp_list"], fout)

    for i, data_item in enumerate(data["data"]):
        with open(args.split_data_dir + "split_{}.json".format(i), "w") as fout:
            json.dump(data_item, fout)


if __name__ == "__main__":
    main()
