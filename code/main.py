import os
import json
import yaml
import datetime
import tensorflow as tf

import LA_dataset
from losses import *
from metrics import *
from vnet import VNet


class Model:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.train_iterator = None
        self.test_iterator = None

    def read_config(self):
        print(f"{datetime.datetime.now()}: Reading configuration file...")

        self.data_dir = os.path.join(os.path.dirname(os.getcwd()), self.config["TrainingSettings"]["DataDirectory"])

        self.epochs = self.config["TrainingSettings"]["Epochs"]
        self.initial_learning_rate = self.config["TrainingSettings"]["InitialLearningRate"]
        self.learning_rate_decay = self.config["TrainingSettings"]["LearningRateDecay"]
        self.batch_size = self.config["TrainingSettings"]["BatchSize"]
        self.dropout_rate = self.config["TrainingSettings"]["DropoutRate"]
        self.training_pipeline = self.config["TrainingSettings"]["Pipeline"]

        print(f"{datetime.datetime.now()}: Reading configuration file complete.")

    def get_dataset_iterator(self, data_dir, transforms, train=True):
        Dataset = LA_dataset.LAHeart(data_dir=data_dir, transforms=transforms, train=train)
        dataset = Dataset.get_dataset()
        dataset = dataset.shuffle(len(dataset))
        dataset = dataset.batch(self.batch_size)
        return dataset

    # TODO
    @tf.function
    def train_step(self, model, next_element):
        with tf.GradientTape() as tape:
            predictions = model(next_element["image"])

    def train(self):
        # read config file
        self.read_config()

        # get images/labels and apply data augmentation
        with tf.device("/cpu:0"):
            # load pipeline from yaml
            with open(os.path.join(os.path.dirname(os.getcwd()), self.training_pipeline), "r") as f:
                pipeline = yaml.load(f, Loader=yaml.FullLoader)

            # get list of transforms
            train_transforms = []
            test_transforms = []

            if pipeline["preprocess"]["train"] is not None:
                for transform in pipeline["preprocess"]["train"]:
                    try:
                        tfm_class = getattr(LA_dataset, transform["name"])(*[], **transform["variables"])
                    except KeyError:
                        tfm_class = getattr(LA_dataset, transform["name"])()
                    train_transforms.append(tfm_class)

            if pipeline["preprocess"]["test"] is not None:
                for transform in pipeline["preprocess"]["test"]:
                    try:
                        tfm_class = getattr(LA_dataset, transform["name"])(*[], **transform["variables"])
                    except KeyError:
                        tfm_class = getattr(LA_dataset, transform["name"])()
                    test_transforms.append(tfm_class)

            # generate tensorflow datasets
            self.train_iterator = self.get_dataset_iterator(self.data_dir, transforms=train_transforms)
            self.test_iterator = self.get_dataset_iterator(self.data_dir, transforms=test_transforms, train=False)

            # TODO: instantiate network, losses, metrics, optimzers

    def test(self):
        pass


if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(os.getcwd()), "configs/config.json"), "r") as config_json:
        config = json.load(config_json)
    model = Model(config)
    model.train()
