import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from model2 import Model


def get_parser():
    parser = argparse.ArgumentParser(
        description="Tensorflow implementation of a dual-task consistency "
                    "semi-supervised medical image segmentation. "
                    "See https://ojs.aaai.org/index.php/AAAI/article/view/17066 "
                    "for more details."
    )
    parser.add_argument(
        "-p",
        "--phase",
        dest="phase",
        type=str,
        choices=["train", "test"],
        help="Training phase"
    )
    parser.add_argument(
        "--config_json",
        dest="config_json",
        type=str,
        default="configs/config.json",
        help="JSON file for model configuration"
    )
    args = parser.parse_args()
    return args


def main(args):
    # load config file
    with open(os.path.join(os.path.dirname(os.getcwd()), args.config_json), "r") as config_json:
        config = json.load(config_json)

    # set seeds
    seed = config["Seed"]
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # tensorflow gpu options
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # run model
    model = Model(config)
    if args.phase == "train":
        model.train()
    elif args.phase == "test":
        model.test()
    else:
        sys.exit("Invalid training phase.")


if __name__ == '__main__':
    args = get_parser()
    main(args)
