import tensorflow as tf
import numpy as np



def dice_loss(predicted, target_gt):
    target_gt = target_gt.float()

    smooth = 1e-5
    intersect = tf.math.reduce_sum(predicted * target_gt)

    gt_sum = tf.math.reduce_sum(target_gt * target_gt)
    predicted_sum = tf.math.reduce_sum(predicted * predicted)

    loss = (2 * intersect + smooth) / (gt_sum + predicted_sum + smooth)
    loss = 1 - loss
    return loss



def entropy_loss():
    pass


def softmax_mse_loss(input_logits, gt_logits, sigmoid=False):

    assert input_logits.size() == gt_logits.size()

    if sigmoid:
        input_softmax = tf.keras.activations.sigmoid(input_logits)
        gt_softmax = tf.keras.activations.sigmoid(gt_logits)

    else:
        input_softmax = tf.nn.softmax(input_logits)
        gt_softmax = tf.nn.softmax(gt_logits)

    mse_loss = (input_softmax - gt_softmax)**2
    return mse_loss
