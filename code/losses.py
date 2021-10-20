import tensorflow as tf
import numpy as np

# Note: still a work in progress


def dice_loss(predicted, target_gt):
    target_gt = target_gt.float()

    smooth = 1e-5
    intersect = tf.math.reduce_sum(predicted * target_gt)

    gt_sum = tf.math.reduce_sum(target_gt * target_gt)
    predicted_sum = tf.math.reduce_sum(predicted * predicted)

    loss = (2 * intersect + smooth) / (gt_sum + predicted_sum + smooth)
    loss = 1 - loss
    return loss


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

    p = tf.convert_to_tensor(p)
    num_class = 2
    num_class = tf.convert_to_tensor(num_class)

    ent = -np.mean(np.sum(p * np.log(p + 1e-6)) / np.log(num_class).cuda())

    return ent


def entropy_minimization(p):

    ent = -np.mean(np.sum(p * np.log(p + 1e-6)))

    return ent


def entropy_map(p):

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


def softmax_dice_loss(input_logits, gt_logits):

    assert input_logits.size() == gt_logits.size()

    input_softmax = tf.nn.softmax(input_logits)
    gt_softmax = tf.nn.softmax(gt_logits)

    n = input_logits.shape[1]
    dice = 0

    for i in range(0, n):
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






