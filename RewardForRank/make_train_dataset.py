import json
import argparse

from utils.text_process import get_context


def context_to_msc(context):
    msc = []
    for character, msg_text in context:
        if character == "user-msg":
            from_user = True
        else:
            from_user = False
        msc.append([from_user, msg_text])
    return msc


def _get_dataset(data, window_size):
    dataset = {
        "sentence1": [],
        "sentence2": [],
        "label": [],
        "idx": [],
        "group_id": [],
        # "full_context": []
    }
    # 'sentence1', 'sentence2', 'label', 'idx', 'group_id', 'full_context'
    for group_id, data_item in enumerate(data):
        context = data_item["context"]
        msc = context_to_msc(context)
        full_context = msc
        sentence1 = get_context(msc, len(msc), window_size)

        all_label = [label for _, label in data_item["resp"]]

        if not "p" in all_label:
            continue

        for resp_text, label in data_item["resp"]: 
            if label == "p":
                label = 1
            else:
                label = 0
            dataset["sentence1"].append(sentence1)
            dataset["sentence2"].append(resp_text)
            dataset["label"].append(label)
            dataset["idx"].append(len(dataset["idx"]))
            dataset["group_id"].append(group_id)
            # dataset["full_context"].append(full_context)
    return dataset


def get_dataset(tagged_data, valid_ratio, window_size):
    valid_len = int(len(tagged_data) * valid_ratio)
    valid_data = tagged_data[:valid_len]
    train_data = tagged_data[valid_len:]
    valid_dataset = _get_dataset(valid_data, window_size)
    train_dataset = _get_dataset(train_data, window_size)
    return train_dataset, valid_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tagged_data_path", help="", type=str, default="../data/tagged_data.no_skipped.json")
    parser.add_argument("--resp_cluster_path", help="use for balance", type=str, default="./data/")
    parser.add_argument("--train_data_out_path", help="", type=str, default="./data/train_v1.json")
    parser.add_argument("--valid_data_out_path", help="", type=str, default="./data/valid_v1.json")
    parser.add_argument("--window_size", help="", type=int, default=10)
    parser.add_argument("--valid_ratio", help="", type=float, default=0.1)
    args = parser.parse_args()

    with open(args.tagged_data_path, "r") as jf:
        tagged_data = json.load(jf)

    train_dataset, valid_dataset = get_dataset(tagged_data, valid_ratio=args.valid_ratio, window_size=args.window_size)

    with open(args.train_data_out_path + ".win{}".format(args.window_size), "w") as fout:
        json.dump(train_dataset, fout)

    with open(args.valid_data_out_path + ".win{}".format(args.window_size), "w") as fout:
        json.dump(valid_dataset, fout)


if __name__ == "__main__":
    main()
