import os
import json
import yaml
import math
import tqdm
import datetime
import nibabel as nib
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from pathlib import Path
from medpy import metric
from skimage.measure import label

import generator
from losses import DTCLoss
from vnet import VNet


class Model:
    def __init__(self, config):
        self.config = config
        self.network = None
        self.loss_fn = None
        self.optimizer = None
        self.current_iter = 0

    def read_config(self):
        print(f"{datetime.datetime.now()}: Reading configuration file...")

        # TODO: only works if run from the code directory
        # I/O settings
        self.data_dir = os.path.join(os.path.dirname(os.getcwd()), self.config["DataDirectory"])
        self.model_save_dir = os.path.join(os.path.dirname(os.getcwd()), self.config["ModelSaveDir"])
        self.test_save_path = os.path.join(os.path.dirname(os.getcwd()), self.config["PredictionSavePath"])

        # model settings
        self.input_shape = self.config["TrainingSettings"]["InputShape"]
        self.iterations = self.config["TrainingSettings"]["Iterations"]
        self.batch_size = self.config["TrainingSettings"]["BatchSize"]
        self.labeled_bs = self.config["TrainingSettings"]["LabeledBatchSize"]
        self.num_labeled = self.config["TrainingSettings"]["NumberLabeledImages"]
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

        # testing settings
        self.stride_xy = self.config["TestingSettings"]["XYStride"]
        self.stride_z = self.config["TestingSettings"]["ZStride"]

        print(f"{datetime.datetime.now()}: Reading configuration file complete.")

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
            if pipeline["preprocess"]["train"] is not None:
                for transform in pipeline["preprocess"]["train"]:
                    try:
                        tfm_class = getattr(generator, transform["name"])(*[], **transform["variables"])
                    except KeyError:
                        tfm_class = getattr(generator, transform["name"])()
                    train_transforms.append(tfm_class)

        # instantiate dataset iterator
        train_generator = generator.DataGenerator(
            data_dir=self.data_dir,
            transforms=train_transforms,
            image_dims=self.input_shape[0:3],
            batch_size=self.batch_size,
            labeled_bs=self.labeled_bs,
            num_labeled=self.num_labeled
        )

        # instantiate VNet model, loss function, optimizer, LR decay
        self.network = VNet(self.input_shape)
        self.loss_fn = DTCLoss(
            self.sigmoid_k,
            self.beta,
            self.consistency,
            self.consistency_rampup,
            self.consistency_interval,
            self.labeled_bs
        )
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_learning_rate,
            decay_steps=self.lr_decay_interval,
            decay_rate=self.learning_rate_decay,
            staircase=True
        )
        self.optimizer = tfa.optimizers.SGDW(
            self.weight_decay,
            lr_schedule,
            self.momentum
        )

        # train model
        self.network.compile(
            optimizer=self.optimizer, loss=self.loss_fn)
        epoch_size = len(train_generator)
        epochs = self.iterations // epoch_size + 1
        train_log = self.network.fit(
            train_generator,
            epochs=epochs,
            steps_per_epoch=epoch_size,
            callbacks=[EpochCallback(self.loss_fn), MaxIterationsCallback(self.iterations)]
        )

        # save model
        if not os.path.isdir(self.model_save_dir):
            Path(self.model_save_dir).mkdir(exist_ok=True)
        complete_model_save_path = os.path.join(self.model_save_dir, f"DTC_{self.num_labeled}_labels")
        self.network.save(complete_model_save_path)
        print(f"{datetime.datetime.now()}: Trained model saved to {complete_model_save_path}.")

    def test(self):
        # read config file
        self.read_config()

        # load saved model
        complete_model_save_path = os.path.join(self.model_save_dir, f"DTC_{self.num_labeled}_labels")
        self.network = tf.keras.models.load_model(complete_model_save_path)
        print(f"{datetime.datetime.now()}: Model loaded from {complete_model_save_path}.")

        # TODO: add code to test_all_cases to load images
        avg_metric = self.test_all_cases(
            self.test_iterator(),
            stride_xy=self.stride_xy,
            stride_z=self.stride_z
        )
        print(f"{datetime.datetime.now()}: Average metric is {avg_metric}.")

    def test_all_cases(self, test_iterator, num_classes=1, patch_size=(112, 112, 80), stride_xy=18,
                       stride_z=4, save_result=True, metric_detail=True, nms=False):
        total_metric = 0.0
        for idx, sample in tqdm.tqdm(enumerate(test_iterator)):
            image, label = sample[0], sample[1]
            pred = self.test_single_case(image, stride_xy, stride_z, patch_size, num_classes=num_classes)

            if nms:
                pred = self.getLargestCC(pred)

            if np.sum(pred) == 0:
                single_metric = (0, 0, 0, 0)
            else:
                single_metric = self.calculate_metric_per_case(pred, label[:].numpy())

            if metric_detail:
                print(f"Image number: {idx}\n, "
                      f"\tDice Coefficient: {single_metric[0]:.5f}\n, "
                      f"\tJaccard: {single_metric[1]:.5f}\n, "
                      f"\t95% Hausdorff Distance (HD): {single_metric[2]:.5f}\n, "
                      f"\tAverage Surface Distance (ASD): {single_metric[3]:.5f}\n, ")

            total_metric += np.asarray(single_metric)

            # save predictions as images
            if save_result:
                if not os.path.isdir(self.test_save_path):
                    Path(self.test_save_path).mkdir(exist_ok=True)

                nib.save(nib.Nifti1Image(pred.astype(np.float32), np.eye(4)),
                         f"{self.test_save_path}/{idx}_pred.nii.gz")
                nib.save(nib.Nifti1Image(image[:].numpy().astype(np.float32), np.eye(4)),
                         f"{self.test_save_path}/{idx}_image.nii.gz")
                nib.save(nib.Nifti1Image(label[:].numpy().astype(np.float32), np.eye(4)),
                         f"{self.test_save_path}/{idx}_gt.nii.gz")

                print(f"{datetime.datetime.now()}: Saved prediction to {self.test_save_path}.")

        avg_metric = total_metric / len(test_iterator)
        return avg_metric

    def test_single_case(self, image, stride_xy, stride_z, patch_size, num_classes=1):
        w, h, d = image.shape
        sx = math.ceil((w - patch_size[0]) / stride_xy) + 1
        sy = math.ceil((h - patch_size[1]) / stride_xy) + 1
        sz = math.ceil((d - patch_size[2]) / stride_z) + 1

        score_map = np.zeros(image.shape + (num_classes, )).astype(np.float32)
        cnt = np.zeros(image.shape).astype(np.float32)

        for x in range(0, sx):
            xs = min(stride_xy * x, w - patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y, h - patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_z * z, d - patch_size[2])
                    test_patch = image[xs:xs + patch_size[0],
                                       ys:ys + patch_size[1],
                                       zs:zs + patch_size[2]]
                    test_patch = np.expand_dims(np.expand_dims(
                        test_patch, axis=0), axis=-1).astype(np.float32)
                    test_patch = tf.convert_to_tensor(test_patch)

                    y1_tanh, y1 = self.network(test_patch)
                    y = tf.math.sigmoid(y1)

                    y = y[0, ...].numpy()
                    score_map[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2], :] \
                        = score_map[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2], :] + y
                    cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                        = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1

        # average logits generated from predictions
        score_map = score_map / np.expand_dims(cnt, axis=-1)
        # convert to binary mask
        label_map = (score_map[..., 0] > 0.5).astype(np.int)
        return label_map

    @staticmethod
    def getLargestCC(seg):
        labels = label(seg)
        assert (labels.max() != 0)  # assume at least 1 CC
        largest_CC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return largest_CC

    @staticmethod
    def calculate_metric_per_case(pred, gt):
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        hd = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, jc, hd, asd


class EpochCallback(tf.keras.callbacks.Callback):
    """Class to get the current epoch during training."""
    def __init__(self, loss):
        self.loss = loss
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        if self.epoch < 1:
            self.epoch = epoch  # starts at 0

    def on_train_batch_begin(self, batch, logs=None):
        self.epoch += 1
        self.loss.set_epoch(self.epoch)


class MaxIterationsCallback(tf.keras.callbacks.Callback):
    """Class to stop training when max iterations are reached."""
    def __init__(self, max_iters):
        self.max_iters = max_iters
        self.batch = None

    def on_batch_end(self, batch, logs=None):
        self.batch = batch + 1

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) * self.batch >= self.max_iters:
            self.model.stop_training = True


if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(os.getcwd()), "configs/config.json"), "r") as config_json:
        config = json.load(config_json)
    model = Model(config)
    # model.train()
    model.test()
