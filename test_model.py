import argparse
from tod_model import MLMBiencoder


test_cont = "I want to honestly know if I\u2019ll hear from\n Someone I broke up with a month ago I miss him a lot"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="", type=str, default="./exp.distbert.b16/checkpoint-36000/")

    args = parser.parse_args()

    model = MLMBiencoder(args.model_path, mlm=True)



if __name__ == "__main__":
    main()
