import os
import h5py
import numpy as np
import tensorflow as tf


class LAHeart:
    """Loads the preprocessed left atrial segmentation challenge images and
    labels for input into the network. Applies on-the-fly data augmentation
    methods during the training phase.
    """
    def __init__(self, data_dir, transforms=None, train=False):
        self.data_dir = data_dir
        self.transforms = transforms
        self.train = train

        if self.train:
            image_list_file = data_dir + "/train.list"
        else:
            image_list_file = data_dir + "/test.list"
        with open(image_list_file, "r") as f:
            self.image_dir_list = f.readlines()
        self.image_dir_list = [image_dir.strip() for image_dir in self.image_dir_list]

        self.num_images = len(self.image_dir_list)
        self.dataset = None

    def get_dataset(self):
        """Returns a TensorFlow Dataset object containing the training
        or testing images and labels.

        :return: a tf.data.Dataset of the training/testing data
        """
        dataset = tf.data.Dataset.from_tensor_slices(self.image_dir_list)
        dataset = dataset.map(lambda case: tuple(tf.py_function(
            self.input_parser, [case], [tf.float32, tf.uint8])),
                              num_parallel_calls=1)
        self.dataset = dataset
        return self.dataset

    def input_parser(self, case):
        """Parses the images and labels from their directory filenames to
        their TensorFlow tensors.

        :param case: a Tensor representing the name of the directory
                     containing the mri_norm.h5 file
        :return: a (image, label) tuple of the image data in case
        """
        case = case.numpy().decode("utf-8")
        image_path = os.path.join(self.data_dir, case)

        # read image and label
        h5f = h5py.File(image_path + "/mri_norm2.h5", "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}

        # apply transforms
        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample)

        return sample["image"], sample["label"]


class RandomCrop:
    """Randomly crops the image to the specified output_size. The output
    size can be a tuple or an integer (for a cubic crop).
    """
    def __init__(self, output_size, seed):
        self.name = "RandomCrop"
        self.seed = seed

        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        image = tf.image.random_crop(image, self.output_size, self.seed)
        label = tf.image.random_crop(label, self.output_size, self.seed)

        return {"image": image, "label": label}


class RandomRotation:
    """Randomly rotates the image 0, 90, 180, or 270 degrees."""
    def __init__(self):
        self.name = "RandomRotation"

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        num_flips = np.random.randint(4)
        image = tf.image.rot90(image, num_flips)
        label = tf.image.rot90(label, num_flips)

        return {"image": image, "label": label}


class RandomFlip:
    """Randomly flips the image in a sample along its x or y axis."""
    def __init__(self, seed):
        self.name = "RandomFlip"
        self.seed = seed

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        axis = np.random.randint(2)
        if axis:  # flip along y axis
            image = tf.image.random_flip_left_right(image, self.seed)
            label = tf.image.random_flip_left_right(label, self.seed)
        else:  # flip along x axis
            image = tf.image.random_flip_up_down(image, self.seed)
            label = tf.image.random_flip_up_down(label, self.seed)

        return {"image": image, "label": label}


# TODO: not sure how the image dimensions will fit in with network
class ToTensor:
    """Converts the 3D image arrays in a sample to Tensors."""
    def __init__(self):
        self.name = "ToTensor"

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = image[:, :, :, np.newaxis]
        label = label[:, :, :, np.newaxis]

        return {"image": image, "label": label}


if __name__ == '__main__':
    import glob
    data_dir = glob.glob("../data/*")[0]
    dataset = LAHeart(data_dir, train=True)
    data = dataset.get_dataset()
    print(len(data))
    print(data.element_spec)
    for elem in data:
        print(elem)
