import os
import json
import numpy as np
import tensorflow as tf
from model import Model


def main():
    # load config file
    with open(os.path.join(os.path.dirname(os.getcwd()), "configs/config.json"), "r") as config_json:
        config = json.load(config_json)

    # set seeds
    seed = config["Seed"]
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # run model
    model = Model(config)
    model.train()


if __name__ == '__main__':
    main()
