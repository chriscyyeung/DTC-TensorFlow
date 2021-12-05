import numpy as np
import tensorflow as tf
from util import compute_lsf_gt, sigmoid_rampup

# To-do: Add comments


class DTCLoss:
    def __init__(self, k, beta, consistency, consistency_rampup, consistency_interval, labeled_bs):
        self.k = k
        self.beta = beta
        self.consistency = consistency
        self.consistency_rampup = consistency_rampup
        self.consistency_interval = consistency_interval
        self.labeled_bs = labeled_bs
        self.mse = tf.keras.losses.MeanSquaredError()
        self.epoch = 0

    def get_current_consistency_weight(self, epoch):
        return self.consistency * sigmoid_rampup(epoch, self.consistency_rampup)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __call__(self, y_true, y_pred, y_pred_tanh):
        y_pred = y_pred[..., 0]
        y_pred_tanh = y_pred_tanh[..., 0]
        pred_soft = tf.keras.activations.sigmoid(y_pred)  # convert to logits

        # labeled predictions
        pred_labeled = y_pred[:self.labeled_bs]
        pred_soft_labeled = pred_soft[:self.labeled_bs]
        pred_tanh_labeled = y_pred_tanh[:self.labeled_bs]
        true_labeled = y_true[:self.labeled_bs]

        # supervised loss (labeled images)
        true_lsf = tf.py_function(compute_lsf_gt, [y_true[:], tf.shape(pred_labeled)], tf.float32)
        loss_lsf = self.mse(true_lsf, pred_tanh_labeled)
        loss_seg_dice = self.dice_loss(true_labeled == 1, pred_soft_labeled)
        supervised_loss = loss_seg_dice + self.beta * loss_lsf

        # unsupervised loss (no labels)
        lsf_to_mask = tf.keras.activations.sigmoid(-self.k * y_pred_tanh)
        consistency_loss = tf.math.reduce_mean((lsf_to_mask - pred_soft) ** 2)
        consistency_weight = tf.py_function(
            self.get_current_consistency_weight,
            [self.epoch // self.consistency_interval],
            tf.float32
        )
        # overall DTC loss
        return supervised_loss + consistency_weight * consistency_loss

    @staticmethod
    def dice_loss(true, pred):
        true = tf.cast(true, tf.float32)

        smooth = 1e-5
        intersect = tf.math.reduce_sum(pred * true)

        gt_sum = tf.math.reduce_sum(true * true)
        predicted_sum = tf.math.reduce_sum(pred * pred)

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
