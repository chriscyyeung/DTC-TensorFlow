import os
import h5py
import itertools
import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    """Loads the preprocessed left atrial segmentation challenge images and
    labels for input into the network. Applies on-the-fly data augmentation
    methods during the training phase.
    """
    def __init__(
            self,
            data_dir,
            transforms=None,
            image_dims=(112, 112, 80),
            n_classes=2,
            batch_size=4,
            labeled_bs=2,
            num_labeled=16
    ):
        self.data_dir = data_dir
        self.transforms = transforms
        self.image_dims = image_dims
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.labeled_bs = labeled_bs
        self.unlabeled_bs = self.batch_size - self.labeled_bs
        self.num_labeled = num_labeled

        # list of images
        image_list_file = data_dir + "/train.list"
        with open(image_list_file, "r") as f:
            self.image_dir_list = f.readlines()
        self.image_dir_list = [image_dir.strip() for image_dir in self.image_dir_list]
        self.total_num_images = len(self.image_dir_list)

        # indices of images used with or without labels for training
        self.primary_indices = list(range(self.num_labeled))  # with label
        self.secondary_indices = list(range(self.num_labeled, self.total_num_images))

        self.indexes = self.get_sample_indices()

    def input_parser(self, case):
        """Parses the images and labels from their directory filenames to
        their TensorFlow tensors.

        :param case: a Tensor representing the name of the directory
                     containing the mri_norm.h5 file
        :return: a (image, label) tuple of the image data in case
        """
        # case = case.numpy().decode("utf-8")
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
        # return list of all indexes in current epoch
        return [idx for batch in batch_iter for idx in batch]

    def on_epoch_end(self):
        self.indexes = self.get_sample_indices()

    def __len__(self):
        # one epoch is one iteration through primary indices
        return self.num_labeled // self.labeled_bs

    def __getitem__(self, index):
        # generate one batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        print(indexes)
        image_dir_temp = [self.image_dir_list[idx] for idx in indexes]

        # create emtpy arrays for all images in batch
        X = np.zeros((0, *self.image_dims))
        y = np.zeros((0, *self.image_dims))

        # add images/labels to array
        for image_dir in image_dir_temp:
            image, label = self.input_parser(image_dir)
            X = np.concatenate((X, np.expand_dims(image.numpy(), axis=0)))
            y = np.concatenate((y, np.expand_dims(label.numpy(), axis=0)))

        X = X[:, :, :, :, np.newaxis]

        return X, y

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
    def __init__(self, output_size):
        self.name = "RandomCrop"

        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        image = tf.image.random_crop(image, self.output_size)
        label = tf.image.random_crop(label, self.output_size)

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
    def __init__(self):
        self.name = "RandomFlip"

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        axis = np.random.randint(2)
        if axis:  # flip along y axis
            image = tf.image.random_flip_left_right(image)
            label = tf.image.random_flip_left_right(label)
        else:  # flip along x axis
            image = tf.image.random_flip_up_down(image)
            label = tf.image.random_flip_up_down(label)

        return {"image": image, "label": label}


if __name__ == '__main__':
    import glob
    import yaml
    np.random.seed(2021)
    data_dir = glob.glob("../data/*")[0]

    with open(os.path.join(os.path.dirname(os.getcwd()), "configs/pipeline.yaml"), "r") as f:
        pipeline = yaml.load(f, Loader=yaml.FullLoader)
    # get list of transforms
    train_transforms = []
    if pipeline["preprocess"]["train"] is not None:
        for transform in pipeline["preprocess"]["train"]:
            try:
                tfm_class = locals()[transform["name"]](*[], **transform["variables"])
            except KeyError:
                tfm_class = locals()[transform["name"]]()
            train_transforms.append(tfm_class)

    dataset = DataGenerator(data_dir, transforms=train_transforms)
    image, label = dataset[0]
    print(image.shape, label.shape)

    print(dataset.indexes)
    for i in range(len(dataset)):
        image, label = dataset[i]
        print(image.shape, label.shape)

    # import nibabel as nib
    # nib.save(nib.Nifti1Image(image[0, ..., 0].astype(np.float32), np.eye(4)), "test_image_0.nii.gz")
    # nib.save(nib.Nifti1Image(label[0].astype(np.float32), np.eye(4)), "test_label_0.nii.gz")
    # nib.save(nib.Nifti1Image(image[1, ..., 0].astype(np.float32), np.eye(4)), "test_image_1.nii.gz")
    # nib.save(nib.Nifti1Image(label[1].astype(np.float32), np.eye(4)), "test_label_1.nii.gz")

    # from model2 import VNet
    # network = VNet((112, 112, 80, 1), 0.0001)
    # out_seg, out_tanh = network(image)
    # print(out_seg.shape, out_tanh.shape)
