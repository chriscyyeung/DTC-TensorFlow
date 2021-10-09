import re
import glob
import random


def split_train_test():
    """Randomly samples 80% of the data files into the training set and 20%
    into the test set. Writes the results into separate list files.

    :return: None
    """
    files = glob.glob("../data/2018_LA_Seg_Challenge/*/")
    files = [re.search("\\\\(.*?)\\\\", path).group(1) for path in files]
    train = random.sample(files, int(len(files) * 0.8))
    test = [folder for folder in files if folder not in train]

    # write to files
    with open("../data/2018_LA_Seg_Challenge/train.list", "w") as f:
        for folder in train:
            f.write(f"{folder}\n")

    with open("../data/2018_LA_Seg_Challenge/test.list", "w") as f:
        for folder in test:
            f.write(f"{folder}\n")


if __name__ == "__main__":
    split_train_test()
