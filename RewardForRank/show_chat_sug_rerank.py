import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_sug_data_path", help="", type=str, default="./data/test_data_2000.json")
    args = parser.parse_args()

    with open(args.chat_sug_data_path, "r") as jf:
        test_data = json.load(jf)

from utils.detect_model import RewardPredict
    
    
    import pdb;pdb.set_trace()


if __name__ == "__main__":
    main()
