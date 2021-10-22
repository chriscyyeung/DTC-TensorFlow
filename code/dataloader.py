import os
import h5py
import itertools
import numpy as np
import tensorflow as tf


class LAHeart:
    """Loads the preprocessed left atrial segmentation challenge images and
    labels for input into the network. Applies on-the-fly data augmentation
    methods during the training phase.
    """
    def __init__(
            self,
            data_dir,
            transforms=None,
            train=False,
            batch_size=4,
            labeled_bs=2,
            num_labeled=16
    ):
        self.data_dir = data_dir
        self.transforms = transforms
        self.train = train
        self.batch_size = batch_size
        self.labeled_bs = labeled_bs
        self.unlabeled_bs = self.batch_size - self.labeled_bs
        self.num_labeled = num_labeled

        # list of images in training and testing sets
        if self.train:
            image_list_file = data_dir + "/train.list"
        else:
            image_list_file = data_dir + "/test.list"
        with open(image_list_file, "r") as f:
            self.image_dir_list = f.readlines()
        self.image_dir_list = [image_dir.strip() for image_dir in self.image_dir_list]
        self.total_num_images = len(self.image_dir_list)

        # indices of images used with or without labels for training
        if self.train:
            self.primary_indices = list(range(self.num_labeled))  # with label
            self.secondary_indices = list(range(self.num_labeled, self.total_num_images))

        self.dataset = None

    def get_dataset(self, image_dir_list):
        """Returns a TensorFlow Dataset object containing the training
        or testing images and labels.

        :return: a tf.data.Dataset of the training/testing data
        """
        dataset = tf.data.Dataset.from_tensor_slices(image_dir_list)
        dataset = dataset.map(lambda case: tuple(tf.py_function(
            self.input_parser, [case], [tf.float32, tf.uint8])),
                              num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

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

    def get_sample_indices(self):
        primary_iter = self.iterate_once(self.primary_indices)
        secondary_iter = self.iterate_eternally(self.secondary_indices)
        # create generator that gets indices of labeled and unlabeled images
        batch_iter = (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in
            zip(self.grouper(primary_iter, self.labeled_bs),
                self.grouper(secondary_iter, self.unlabeled_bs))
        )
        return [batch_idx for batch_idx in batch_iter]

    def __len__(self):
        if self.train:
            # number of batches per one run though dataset
            return int(self.num_labeled / self.labeled_bs)
        else:
            # use all for testing
            return self.total_num_images

    def __call__(self):
        if self.train:
            # randomly get filenames of 2 labeled and 2 unlabeled images
            batch_idx_list = self.get_sample_indices()

            # create first batch
            batch_dir_list = [self.image_dir_list[idx] for idx in batch_idx_list[0]]
            self.dataset = self.get_dataset(batch_dir_list)

            # concatenate remaining batches to first
            for i in range(1, len(batch_idx_list)):
                batch_idx = batch_idx_list[i]
                batch_dir_list = [self.image_dir_list[idx] for idx in batch_idx]
                next_batch = self.get_dataset(batch_dir_list)
                self.dataset = self.dataset.concatenate(next_batch)

            return self.dataset.batch(self.batch_size)
        # return all images for testing
        else:
            return self.get_dataset(self.image_dir_list)

    @staticmethod
    def iterate_once(indices):
        return np.random.permutation(indices)

    @staticmethod
    def iterate_eternally(indices):
        def infinite_shuffles():
            while True:
                yield np.random.permutation(indices)
        return itertools.chain.from_iterable(infinite_shuffles())

    @staticmethod
    def grouper(iterable, n):
        """Collect data into fixed-length chunks or blocks
            grouper("ABCDEFG", 3) --> ABC DEF
        """
        args = [iter(iterable)] * n
        return zip(*args)


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
    np.random.seed(2021)
    data_dir = glob.glob("../data/*")[0]
    dataset = LAHeart(data_dir, train=True)
    print(dataset.get_sample_indices())
    print(dataset().element_spec)
    print(len(dataset))
