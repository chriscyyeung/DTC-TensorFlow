import numpy as np
from skimage import segmentation as skimage_seg
from scipy.ndimage import distance_transform_edt as distance


def compute_lsf_gt(label, output_shape):
    """Computes the signed distance function of a binary image label.
    sdf(x) = 0 (x in segmentation boundary),
             -inf|x-y| (x in segmentation),
             +inf|x-y| (x outside of segmentation)

    :param label: the segmentation mask with shape (batch size, x, y, z)
    :param output_shape: a tuple or list of shape (batch size, x, y, z)
    :return: the [-1, 1] normalized signed distance function of the
             segmentation mask
    """
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
    return normalized_sdf
