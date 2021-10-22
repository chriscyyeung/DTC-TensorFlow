import numpy as np
import tensorflow as tf
from skimage import segmentation as skimage_seg
from scipy.ndimage import distance_transform_edt as distance


def compute_lsf_gt(label, output_shape):
    """Computes the signed distance function of a binary image label.
        sdf(x) = 0 (x in segmentation boundary),
                 -inf|x-y| (x in segmentation),
                 +inf|x-y| (x outside of segmentation)

    :param label: the segmentation mask with shape (batch size, x, y, z)
    :param output_shape: a tuple or list of shape (batch size, x, y, z)
    :return: a Tensor of the [-1, 1] normalized signed distance function
             of the segmentation mask
    """
    label = label.numpy()
    normalized_sdf = np.zeros(output_shape)
    for batch in range(output_shape[0]):
        pos_mask = label[batch].astype(np.bool)
        if pos_mask.any():
            neg_mask = ~pos_mask
            pos_dis = distance(pos_mask)
            neg_dis = distance(neg_mask)
            boundary = skimage_seg.find_boundaries(pos_mask, mode="inner").astype(np.uint8)
            sdf = ((neg_dis - np.min(neg_dis)) / (np.max(neg_dis) - np.min(neg_dis)) -
                   (pos_dis - np.min(pos_dis)) / (np.max(pos_dis) - np.min(pos_dis)))
            sdf[boundary == 1] = 0
            normalized_sdf[batch] = sdf
    return tf.convert_to_tensor(normalized_sdf, tf.float32)


def sigmoid_rampup(epoch, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242 to
    control the balance between the supervised loss and the unsupervised
    consistency loss. Uses a time-dependent Gaussian function as follows:
        gamma(t) = e^(-5(1-(t/tmax))^2)

    :param epoch: an int denoting the current training step
    :param rampup_length: a float representing the rampup length
    :return: a float representing gamma at the current training step
    """
    if rampup_length == 0:
        return 1.0
    current = np.clip(epoch.numpy(), 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


if __name__ == '__main__':
    for i in range(6001):
        print(sigmoid_rampup(i // 150, 40.0))
