import datetime
import numpy as np
import tensorflow as tf

from losses import *
from vnet import VNet


class Model:
    def __init__(self, config):
        self.config = config
        self.model = None

    def read_config(self):
        print(f"{datetime.datetime.now()}: Reading configuration file...")

        self.data_dir = self.config["TrainingSettings"]["DataDirectory"]

        self.epochs = self.config["TrainingSettings"]["Epochs"]
        self.initial_learning_rate = self.config["TrainingSettings"]["InitialLearningRate"]
        self.learning_rate_decay = self.config["TrainingSettings"]["LearningRateDecay"]
        self.batch_size = self.config["TrainingSettings"]["BatchSize"]
        self.dropout = self.config["TrainingSettings"]["Dropout"]

        print(f"{datetime.datetime.now()}: Reading configuration file complete.")

    def get_dataset_iterator(self, data_dir, transforms, train=True):
        pass
