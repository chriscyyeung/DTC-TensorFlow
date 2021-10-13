import os
import tensorflow as tf


class LAHeart:
    def __init__(self, data_dir, transforms=None, train=False):
        self.data_dir = data_dir
        self.transforms = transforms
        self.train = train

        with open(data_dir, "r") as f:
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

        image_paths = []
        for i in range(self.num_images):
            image_paths.append(os.path.join(self.data_dir, case, self.image_dir_list[i]))
