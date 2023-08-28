import os
import json
import argparse

from rich.console import Console
from tqdm import tqdm


console = Console()


def load_json(json_path):
    jf = open(json_path, "r")
    data = json.load(jf)
    jf.close()
    return data


def load_test_sentence(args):
    file_name_list = os.listdir(args.test_src_dir)
    file_path_list = [args.test_src_dir + file_name for file_name in file_name_list]

    sentences = []

    for file_path in file_path_list:
        f = open(file_path, "r")
        data = f.readlines()
        f.close()

        if data:
            del(data[0])

        for line in data:
            if "{\"\"detail\"\"" in line:
                continue
            sentence_pieces = line.split(",")[4:]
            sentences.append(",".join(sentence_pieces)[:-1])

    return sentences


def filter_sentences(train_data, sentences):
    new_sentences = []
    console.print("Running the filtering process...")
    for sentence in tqdm(sentences):
        sentence_in_train = False
        for train_line in train_data:
            if sentence == train_line[0]:
                sentence_in_train = True
        if not sentence_in_train:
            new_sentences.append(sentence)
    console.print("The filtered sentence is {} --> {}".format(len(sentences), len(new_sentences)))
    console.print("The filtered ratio is {}".format((len(sentences) - len(new_sentences)) / len(sentences)))
    return new_sentences
    

def tag_data(sentences, tag_out_path, tagged_sentences):
    for sentence_index, sentence in enumerate(sentences):
        if sentence_index <= (len(tagged_sentences) - 1):
            continue
        console.print("##"*20, style="yellow")
        console.print("already tagged {} sentences...".format(len(tagged_sentences)))
        console.print("{} sentences remaining to be tagged".format(len(sentences) - len(tagged_sentences)))
        console.print("The sentence is: {}".format(sentence))
        console.print("--"*20, style="yellow")
        console.print("Please choose 0: flirt or 1: no_flirt 2: out", style="green")
        while(True):
            user_input = input()
            if not user_input.isdigit():
                console.print("Wrong input!!!", data)
                continue
            else:
                input_int = int(user_input)
                if input_int not in [0, 1, 2]:
                    console.print("Wrong input!!!", data)
                    continue
                else:
                    label_id = input_int
                    break
        tagged_sentences.append((sentence, label_id))
        if len(tagged_sentences) % 10 == 0:
            json_str = json.dumps(tagged_sentences, indent=2)
            fout = open(tag_out_path, "w")
            fout.write(json_str)
            fout.close()
    return tagged_sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_src_dir", help="", type=str, default="./data/test_src/raw/")
    parser.add_argument("--train_data_path", help="test for data in the training make errors!", type=str, default="./data/train/train.json")
    parser.add_argument("--filter_out", help="", type=str, default="./data/test_src/filtered_src.json")
    parser.add_argument("--tag_out_path", help="", type=str, default="./data/test_src/tagged_data.json")
    parser.add_argument("--run_filter", help="If run filter, use test_src_dir. else use filtered out", action=argparse.BooleanOptionalAction, required=True)
    args = parser.parse_args()

    if args.run_filter:
        train_data = load_json(args.train_data_path)
        sentences = load_test_sentence(args)
        sentences = filter_sentences(train_data, sentences)
        json_str = json.dumps(sentences, )
        fout = open(args.filter_out, "w")
        fout.write(json_str)
        fout.close()
    else:
        jf = open(args.filter_out, "r")
        sentences = json.load(jf)
        jf.close()

    if os.path.exists(args.tag_out_path):
        jf = open(args.tag_out_path, "r")
        tagged_sentences = json.load(jf)
        jf.close()
    else:
        tagged_sentences = []

    tag_data(sentences, args.tag_out_path, tagged_sentences)


if __name__ == "__main__":
    main()

