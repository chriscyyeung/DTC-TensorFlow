import os
import json
import yaml
import tqdm
import datetime
import tensorflow as tf
import tensorflow_addons as tfa

import LA_dataset
from losses import DTCLoss
from vnet import VNet


class Model:
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.network = None
        self.loss_fn = None
        self.optimizer = None
        self.current_iter = 0

    def read_config(self):
        print(f"{datetime.datetime.now()}: Reading configuration file...")

        self.data_dir = os.path.join(os.path.dirname(os.getcwd()), self.config["TrainingSettings"]["DataDirectory"])

        # model settings
        self.input_shape = self.config["TrainingSettings"]["InputShape"]
        self.epochs = self.config["TrainingSettings"]["Epochs"]
        self.batch_size = self.config["TrainingSettings"]["BatchSize"]
        self.dropout_rate = self.config["TrainingSettings"]["DropoutRate"]
        self.training_pipeline = self.config["TrainingSettings"]["Pipeline"]

        # optimizer settings
        self.initial_learning_rate = self.config["TrainingSettings"]["Optimizer"]["InitialLearningRate"]
        self.learning_rate_decay = self.config["TrainingSettings"]["Optimizer"]["LearningRateDecay"]
        self.lr_decay_interval = self.config["TrainingSettings"]["Optimizer"]["LearningRateDecayInterval"]
        self.weight_decay = self.config["TrainingSettings"]["Optimizer"]["WeightDecay"]
        self.momentum = self.config["TrainingSettings"]["Optimizer"]["Momentum"]

        # loss settings
        self.sigmoid_k = self.config["TrainingSettings"]["Loss"]["K"]
        self.beta = self.config["TrainingSettings"]["Loss"]["Beta"]
        self.consistency = self.config["TrainingSettings"]["Loss"]["Consistency"]
        self.consistency_rampup = self.config["TrainingSettings"]["Loss"]["ConsistencyRampup"]
        self.consistency_interval = self.config["TrainingSettings"]["Loss"]["ConsistencyInterval"]

        print(f"{datetime.datetime.now()}: Reading configuration file complete.")

    def get_dataset_iterator(self, data_dir, transforms, train=True):
        Dataset = LA_dataset.LAHeart(data_dir=data_dir, transforms=transforms, train=train)
        dataset = Dataset.get_dataset()
        dataset = dataset.shuffle(len(dataset))
        dataset = dataset.batch(self.batch_size)
        return dataset

    @tf.function
    def train_step(self, next_element, epoch):
        label = next_element[1]
        with tf.GradientTape() as tape:
            # get predictions
            pred_tanh, pred = self.network(next_element[0])
            # calculate loss
            self.loss_fn.set_true(label[:, :, :, :, 0])
            self.loss_fn.set_epoch(epoch)
            loss = self.loss_fn(pred[:, :, :, :, 0], pred_tanh[:, :, :, :, 0])
        grads = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_weights))
        return loss

    def train(self):
        # read config file
        self.read_config()

        # get images/labels and apply data augmentation
        with tf.device("/cpu:0"):
            print(f"{datetime.datetime.now()}: Loading images and applying transformations...")
            # load pipeline from yaml
            with open(os.path.join(os.path.dirname(os.getcwd()), self.training_pipeline), "r") as f:
                pipeline = yaml.load(f, Loader=yaml.FullLoader)

            # get list of transforms
            train_transforms = []

            if pipeline["preprocess"]["train"] is not None:
                for transform in pipeline["preprocess"]["train"]:
                    try:
                        tfm_class = getattr(LA_dataset, transform["name"])(*[], **transform["variables"])
                    except KeyError:
                        tfm_class = getattr(LA_dataset, transform["name"])()
                    train_transforms.append(tfm_class)

            # generate tensorflow datasets
            self.train_iterator = self.get_dataset_iterator(self.data_dir, transforms=train_transforms)

            print(f"{datetime.datetime.now()}: Image loading and transformation complete.")

        # instantiate VNet model, loss function, optimizer
        self.network = VNet(self.input_shape)
        self.loss_fn = DTCLoss(
            self.sigmoid_k,
            self.beta,
            self.consistency,
            self.consistency_rampup,
            self.consistency_interval
        )
        self.optimizer = tfa.optimizers.SGDW(
            self.weight_decay,
            self.initial_learning_rate,
            self.momentum
        )

        # train the network
        max_epochs = self.epochs // len(self.train_iterator)  # number of passes through entire dataset
        print(f"{datetime.datetime.now()}: Beginning training...")
        for epoch in tqdm.tqdm(range(max_epochs + 1)):
            for sampled_batch in tqdm.tqdm(self.train_iterator):
                print(f"{datetime.datetime.now()}: Starting epoch {self.current_iter + 1}...")
                loss = self.train_step(sampled_batch, self.current_iter)
                self.current_iter += 1

                # log loss every 100 iterations
                if self.current_iter % 100 == 0:
                    print(f"{datetime.datetime.now()}: Model loss at epoch {self.current_iter}: {loss:.4f}")

                # adjust learning rate
                if self.current_iter % self.lr_decay_interval == 0:
                    new_lr = self.initial_learning_rate * self.learning_rate_decay ** \
                             (self.current_iter // self.lr_decay_interval)
                    self.optimizer.lr.assign(new_lr)
                    print(f"{datetime.datetime.now()}: Learning rate decayed to {new_lr}")

    # TODO
    def test(self):
        # get images/labels and apply data augmentation
        with tf.device("/cpu:0"):
            print(f"{datetime.datetime.now()}: Loading images and applying transformations...")
            # load pipeline from yaml
            with open(os.path.join(os.path.dirname(os.getcwd()), self.training_pipeline), "r") as f:
                pipeline = yaml.load(f, Loader=yaml.FullLoader)

            # get list of transforms
            test_transforms = []
            if pipeline["preprocess"]["test"] is not None:
                for transform in pipeline["preprocess"]["test"]:
                    try:
                        tfm_class = getattr(LA_dataset, transform["name"])(*[], **transform["variables"])
                    except KeyError:
                        tfm_class = getattr(LA_dataset, transform["name"])()
                    test_transforms.append(tfm_class)

            # generate tensorflow datasets
            self.test_iterator = self.get_dataset_iterator(self.data_dir, transforms=test_transforms, train=False)

            print(f"{datetime.datetime.now()}: Image loading and transformation complete.")


if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(os.getcwd()), "configs/config.json"), "r") as config_json:
        config = json.load(config_json)
    model = Model(config)
    model.train()
