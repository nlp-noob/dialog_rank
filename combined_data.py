import argparse
import json

#     tagged_data = {
#         "cont": [],
#         "resp": [],
#         "mlm": [],
#         "p_index": [],
#         "n_index": [],
#         "dup_n_id": [],
#         "info": {
#             "cluster_label_to_index_list": cluster_label_to_index_list,
#             "index_to_cluster_label": index_to_cluster_label,
#         },
#     }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dup_dataset_path", help="The path to the dup dataset, ", default="./data/tri_train/train_v7_dup_n_small.json")
    parser.add_argument("--sim_dataset_path", help="The path to the dup dataset, ", default="./data/tri_train/train_v6_small.json")
    args = parser.parse_args()

    with open(args.dup_dataset_path, "r") as jf:
        dup_data = json.load(jf)

    with open(args.sim_dataset_path, "r") as jf:
        sim_data = json.load(jf)

    combined_data = {
        "cont": [],
        "resp": [],
        "mlm": [],
        "p_index": [],
        "n_index": [],
        "dup_n_id": [],
        "info": {
            "cluster_label_to_index_list": cluster_label_to_index_list,
            "index_to_cluster_label": index_to_cluster_label,
        }
    }


if __name__ == "__main__":
    main()
