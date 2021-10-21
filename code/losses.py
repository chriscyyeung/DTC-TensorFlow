import numpy as np
import tensorflow as tf
from util import compute_lsf_gt, sigmoid_rampup

# To-do: Add comments


class DTCLoss(tf.keras.losses.Loss):
    def __init__(self, k, beta, consistency, consistency_rampup, consistency_interval, labeled_bs):
        super().__init__()

        self.k = k
        self.beta = beta
        self.consistency = consistency
        self.consistency_rampup = consistency_rampup
        self.consistency_interval = consistency_interval
        self.labeled_bs = labeled_bs
        self.mse = tf.keras.losses.MeanSquaredError()
        self.epoch = 0
        self.true = None

    @staticmethod
    def dice_loss(pred, true):
        true = tf.cast(true, tf.float32)

        smooth = 1e-5
        intersect = tf.math.reduce_sum(pred * true)

        gt_sum = tf.math.reduce_sum(true * true)
        predicted_sum = tf.math.reduce_sum(pred * pred)

        loss = (2 * intersect + smooth) / (gt_sum + predicted_sum + smooth)
        loss = 1 - loss
        return loss

    def get_current_consistency_weight(self, epoch):
        return self.consistency * sigmoid_rampup(epoch, self.consistency_rampup)

    def set_true(self, true):
        self.true = true

    def set_epoch(self, epoch):
        self.epoch = epoch

    def call(self, pred, pred_tanh):
        pred = tf.keras.activations.sigmoid(pred)  # convert to logits

        # labeled predictions
        pred_labeled = pred[:self.labeled_bs]
        pred_tanh_labeled = pred_tanh[:self.labeled_bs]
        true_labeled = self.true[:self.labeled_bs]

        # supervised loss (labeled images)
        true_lsf = tf.py_function(compute_lsf_gt, [self.true[:], true_labeled.shape], tf.float32)
        loss_lsf = self.mse(pred_tanh_labeled, true_lsf)
        loss_seg_dice = self.dice_loss(pred_labeled, true_labeled == 1)
        supervised_loss = loss_seg_dice + self.beta * loss_lsf

        # unsupervised loss (no labels)
        lsf_to_mask = tf.keras.activations.sigmoid(-self.k * pred_tanh)
        consistency_loss = tf.math.reduce_mean((lsf_to_mask - pred) ** 2)
        consistency_weight = self.get_current_consistency_weight(self.epoch // self.consistency_interval)

        # overall DTC loss
        return supervised_loss + consistency_weight * consistency_loss


def dice_loss1(predicted, target_gt):

    target_gt = target_gt.float()

    smooth = 1e-5
    intersect = tf.math.reduce_sum(predicted * target_gt)

    gt_sum = tf.math.reduce_sum(target_gt)
    predicted_sum = tf.math.reduce_sum(predicted)

    loss = (2 * intersect + smooth) / (gt_sum + predicted_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p):

    num_class = 2
    num_class = tf.convert_to_tensor(num_class)

    ent = -np.mean(np.sum(p * np.log(p + 1e-6)) / np.log(num_class)).cuda()

    return ent


def entropy_minimization(p):

    ent = -np.mean(np.sum(p * np.log(p + 1e-6)))

    return ent


def softmax_mse_loss(input_logits, gt_logits, sigmoid=False):

    assert input_logits.size() == gt_logits.size()

    if sigmoid:
        input_softmax = tf.keras.activations.sigmoid(input_logits)
        gt_softmax = tf.keras.activations.sigmoid(gt_logits)

    else:
        input_softmax = tf.nn.softmax(input_logits)
        gt_softmax = tf.nn.softmax(gt_logits)

    mse_loss = (input_softmax - gt_softmax) ** 2
    return mse_loss


def softmax_dice_loss(input_logits, gt_logits):

    assert input_logits.size() == gt_logits.size()

    input_softmax = tf.nn.softmax(input_logits)
    gt_softmax = tf.nn.softmax(gt_logits)

    n = input_logits.shape[1]
    dice = 0

    for i in range(n):
        dice += dice_loss1(input_softmax[:, i], gt_softmax[:, i])

    mean_dice = dice / n
    return mean_dice


def softmax_kl_loss(input_logits, gt_logits, sigmoid=False):

    assert input_logits.size() == gt_logits.size()

    if sigmoid:
        input_log_softmax = np.log(tf.keras.activations.sigmoid(input_logits))
        gt_softmax = tf.keras.activations.sigmoid(gt_logits)

    else:
        input_log_softmax = np.log(tf.nn.softmax(input_logits))
        gt_softmax = tf.nn.softmax(gt_logits)

    kl = tf.keras.losses.KLDivergence()
    kl_div = kl(gt_softmax, input_log_softmax)
    return kl_div
