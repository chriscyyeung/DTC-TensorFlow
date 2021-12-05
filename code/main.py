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
    parser.add_argument(
        "--debug_mode",
        type=bool,
        choices=[True, False],
        help="Enable TensorFlow debugger V2"
    )
    parser.add_argument(
        "--dump_dir",
        type=str,
        default="/tmp/tfdbg2_logdir"
    )
    parser.add_argument(
        "--dump_tensor_debug_mode",
        type=str,
        default="FULL_HEALTH"
    )
    parser.add_argument(
        "--dump_circular_buffer_size",
        type=int,
        default=-1
    )
    return parser.parse_args()


def main(FLAGS):
    # debugging
    if FLAGS.debug_mode:
        tf.debugging.experimental.enable_dump_debug_info(
            FLAGS.dump_dir,
            tensor_debug_mode=FLAGS.dump_tensor_debug_mode,
            circular_buffer_size=FLAGS.dump_circular_buffer_size
        )
    tf.debugging.enable_check_numerics()

    # load config file
    with open(os.path.join(os.path.dirname(os.getcwd()), FLAGS.config_json), "r") as config_json:
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
    if FLAGS.phase == "train":
        model.train()
    elif FLAGS.phase == "test":
        model.test()
    else:
        sys.exit("Invalid training phase.")


if __name__ == '__main__':
    FLAGS = get_parser()
    main(FLAGS)
