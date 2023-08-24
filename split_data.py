import json
import argparse

from rich.console import Console


console = Console()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order_path", help="the path to the small order data.", type=str, default="./data/orders.star1-2.1.json")
    parser.add_argument("--out_path", help="output path of the splitted file.", type=str, default="./data/ordersstar1-2.1.1.json")
    args = parser.parse_args()

    with open(args.order_path, "r") as jf:
        order_data = json.load(jf)

    order_key_list = list(order_data.keys())
    split_order_key_list = order_key_list[:int(len(order_key_list) * 0.1)]
    splitted_order = {}

    for key in split_order_key_list:
        splitted_order[key] = order_data[key]

    with open(args.out_path, "w") as fout:
        json.dump(splitted_order, fout)


if __name__ == "__main__":
    main()


