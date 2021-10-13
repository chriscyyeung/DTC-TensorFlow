import os
import h5py
import tensorflow as tf


class LAHeart:
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

        self.num_images = len(self.image_dir_list)
        self.dataset = None

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.image_dir_list)
        dataset = dataset.map(lambda case: tuple(tf.py_function(
            self.input_parser, [case], [tf.float32, tf.uint8])),
                              num_parallel_calls=1)
        self.dataset = dataset
        return self.dataset

    def input_parser(self, case):
        case = case.decode("utf-8")
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


if __name__ == '__main__':
    import glob
    data_dir = glob.glob("../data/*")[0]
    dataset = LAHeart(data_dir, train=True)
    data = dataset.get_dataset()
    print(len(data))
