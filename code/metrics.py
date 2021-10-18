import tensorflow as tf
from medpy import metrics as mt

def calculate_eval_metric(predicted, target_gt):

    asd = mt.binary.asd(predicted, target_gt)
    jaccard = mt.binary.jc(predicted,target_gt)
    dc = mt.binary.dc(predicted, target_gt)
    hd = mt.binary.hd(predicted, target_gt)

    return asd, jaccard, dc, hd

def dice_coef(predicted, gt):

    pred_f = tf.reshape(tf.dtypes.cast(predicted,tf.float32), [-1])
    gt_f = tf.reshape(tf.dtypes.cast(gt, tf.float32), [-1])

    smooth = 1

    intersect = tf.math.reduce_sum(pred_f * gt_f)
    pred_sum = tf.math.reduce_sum(pred_f)
    gt_sum = tf.math.reduce_sum(gt_f)

    dice = (2 * intersect + smooth) / (gt_sum + pred_sum + smooth)

    return dice



