# deprecated use get_train_data.py
import json
import argparse
import random

from transformers import  AutoTokenizer
from rich.console import Console
from tqdm import tqdm

console = Console()
random.seed(12835)


def tag_with_score(scored_data, model_path): 
    # 'cont', 'resp', 'cluster_label', 'p_cluster_list', 'n_cluster_list'

    # Get cluster label
    index_to_cluster = {}
    cluster_to_index_list = {}
    tagged_data = {"cont_id": [], "p_resp_id": [], "n_resp_id": [], "mlm_id": [], "cluster_label": [], "is_origin": []}

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for index in range(len(scored_data["cont"])):
        cluster_label = scored_data["cluster_label"][index]
        index_to_cluster[index] = cluster_label
        if cluster_label not in cluster_to_index_list:
            cluster_to_index_list[cluster_label] = []
        cluster_to_index_list[cluster_label].append(index)
        
    abandon_cnt = 0
    console.print("collating the training data.")
    for index in tqdm(range(len(scored_data["cont"]))):
        cont_id = scored_data["cont"][index]
        resp_id = scored_data["resp"][index]
        cluster_label = scored_data["cluster_label"][index]
        p_cluster_set = set(scored_data["p_cluster_list"][index])
        n_cluster_set = set(scored_data["n_cluster_list"][index])
        conflic_set = p_cluster_set & n_cluster_set
        pure_p_cluster_set = p_cluster_set - conflic_set
        pure_n_cluster_set = n_cluster_set - conflic_set

        # console.print("--"*20, style="yellow")
        # console.print("The lenth of the p list {}".format(len(p_cluster_set)))
        # console.print("The lenth of the n list {}".format(len(n_cluster_set)))
        # console.print("The lenth of the confic set {}".format(len(conflic_set)))
        # console.print("The lenth of pure_p_cluster_set: {}".format(len(pure_p_cluster_set)))
        # console.print("The lenth of pure_n_cluster_set: {}".format(len(pure_n_cluster_set)))
        
        p_set_to_select = None

        if len(pure_p_cluster_set) == 0 and len(p_cluster_set) == 0:
            abandon_cnt += 1
            continue
        elif len(pure_p_cluster_set) == 0 and len(p_cluster_set) != 0:
            p_set_to_select = p_cluster_set
        else:
            p_set_to_select = pure_p_cluster_set

        random_cluster = random.choice(list(p_set_to_select))
        resp_id_list = cluster_to_index_list[random_cluster]
        random_p_resp_id = scored_data["resp"][random.choice(resp_id_list)]

        random_cluster_n1 = random.choice(list(n_cluster_set))
        resp_id_list = cluster_to_index_list[random_cluster_n1]
        random_n_resp_id1 = scored_data["resp"][random.choice(resp_id_list)]

        n_cluster_set.remove(random_cluster_n1)

        random_cluster_n2 = random.choice(list(n_cluster_set))
        resp_id_list = cluster_to_index_list[random_cluster_n2]
        random_n_resp_id2 = scored_data["resp"][random.choice(resp_id_list)]

        tagged_data["p_resp_id"].append(resp_id)
        tagged_data["n_resp_id"].append(random_n_resp_id1)
        tagged_data["cont_id"].append(cont_id)
        tagged_data["cluster_label"].append(cluster_label)
        tagged_data["is_origin"].append(1)
        mlm = []
        mlm.extend(cont_id)
        mlm.extend(resp_id[1:])
        tagged_data["mlm_id"].append(mlm)

        # tagged_data["p_resp_id"].append(random_p_resp_id)
        # tagged_data["n_resp_id"].append(random_n_resp_id2)
        # tagged_data["cont_id"].append(cont_id)
        # tagged_data["cluster_label"].append(cluster_label)
        # tagged_data["is_origin"].append(0)
        # mlm = []
        # mlm.extend(cont_id)
        # mlm.extend(random_p_resp_id[1:])
        # tagged_data["mlm_id"].append(mlm)

    console.print("The abandon rate is: {}".format(abandon_cnt / len(scored_data["cont"])))

    return tagged_data
 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_data_path", help="path to the scored_data json file. default: ./tmp/scored_data.json", type=str, default="./tmp/scored_data.json")
    parser.add_argument("--model_path", help="path to the model. default: ./exp.distbert.b16/chkps_epoch-99/", type=str, default="./exp.distbert.b16/chkps_epoch-99/")
    parser.add_argument("--out_path", help="output of the training data. default: ./data/tri_train/train.json", type=str, default="./data/tri_train/train_v2.json")
    args = parser.parse_args()

    jf = open(args.scored_data_path, "r")
    scored_data = json.load(jf)
    jf.close()

    tagged_data = tag_with_score(scored_data, args.model_path)

    json_str = json.dumps(tagged_data, indent=2)
    fout = open(args.out_path, "w")
    fout.write(json_str)
    fout.close()


if __name__ == "__main__":
    main()

