import enum
import subprocess
import pathlib

import checksumdir
import numpy as np
import keras

import dataset_norb


class Dataset(enum.Enum):
    MNIST = 1
    CIFAR10 = 2
    NORB = 3

def get_dataset(dataset: Dataset):
    if dataset == Dataset.MNIST:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_shape = tuple(list(x_train.shape[1:]) + [1])
        x_train = x_train.reshape(-1, *x_shape)
        x_test = x_test.reshape(-1, *x_shape)
    elif dataset == Dataset.CIFAR10:
        (x_train, y_train), (x_test, y_test) = \
                keras.datasets.cifar10.load_data()
    elif dataset == Dataset.NORB:
        (x_train, y_train), (x_test, y_test) = load_norb()
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
    return (x_train, y_train), (x_test, y_test)


def load_norb(validation_set=False):
    old_dims = (96, 96, 1)
    data_slice = 1  # Slice amount for sampling training and test data
    feat_slice = 1  # Slice amount for downsampling features
    # If True, uses jittered test data else use small norb.
    jittered = False
    n_classes = 5
    validation_set = False
    base_data_dir = "norb_data"

    dims = [int(old_dims[0] / feat_slice),
            int(old_dims[1] / feat_slice),
            1]  # downsample!
    small_norb_data_dir = base_data_dir + "/small-norb/"
    jittered_norb_data_dir = base_data_dir + "/jittered/"
    train_data_dir = small_norb_data_dir
    if jittered:
        test_data_dir = jittered_norb_data_dir
    else:
        test_data_dir = small_norb_data_dir
    small_norb_data_dir_path = pathlib.Path(small_norb_data_dir)
    jittered_norb_data_dir_path = pathlib.Path(jittered_norb_data_dir)
    if not small_norb_data_dir_path.exists():
        # Attempt to download files using shell script
        small_norb_download_script_name = "./scripts/download-small.sh"
        subprocess.call([small_norb_download_script_name])
    if not jittered_norb_data_dir_path.exists() and jittered:
        # Attempt to download files using shell script
        jittered_norb_download_script_name = "./scripts/download-jittered-norb.sh"
        subprocess.call([jittered_norb_download_script_name])
    small_norb_sha1 = checksumdir.dirhash(small_norb_data_dir_path.as_posix(),
                                          "sha1")
    while small_norb_sha1 != "491e5ce8bf79fbb750784f4cafd69648e3257e1d":
        print("small_norb_sha1 is incorrect... re-downloading")
        # Attempt to download files using shell script
        small_norb_download_script_name = "./scripts/download-small.sh"
        subprocess.call([small_norb_download_script_name])
        small_norb_sha1 = checksumdir.dirhash(
            small_norb_data_dir_path.as_posix(),
            "sha1")
    if jittered:
        jittered_norb_sha1 = checksumdir.dirhash(
            jittered_norb_data_dir_path.as_posix(),
            "sha1"
        )
        while jittered_norb_sha1 != "4ac10244bec6c981d00645239f0b0a22bf12ee25":
            print("jittered_norb_sha1 is incorrect... re-downloading")
            # Attempt to download files using shell script
            jittered_norb_download_script_name = "./scripts/download-jittered-norb.sh"
            subprocess.call([jittered_norb_download_script_name])
            jittered_norb_sha1 = checksumdir.dirhash(
                jittered_norb_data_dir_path.as_posix(),
                "sha1"
            )

    (train_images,
     train_labels,
     validation_images,
     validation_labels,
     test_images,
     test_labels) = dataset_norb.load_norb_data(
                            train_data_dir,
                            test_data_dir,
                            dims,
                            n_classes,
                            one_hot=False,
                            as_float=True,
                            validation_set=validation_set,
                            jittered=jittered,
                            data_slice=data_slice,
                            feat_slice=feat_slice)
    train_images = train_images.reshape(-1, *dims)
    validation_images = validation_images.reshape(-1, *dims)
    test_images = test_images.reshape(-1, *dims)
    if not validation_set:
        return (train_images, train_labels), (test_images, test_labels)
    else:
        return (train_images, train_labels), \
               (validation_images, validation_labels), \
               (test_images, test_labels)


def select_subset_classes(classes, X, y, binarize=True):
    """
    Args:
        classes (list or tuple): A list or tuple of length 2 containing the
            positive and negative classes. Normally the 2 elements are ints,
            in which case they represent single class labels. If any of the
            elements are list or tuple, they will be interpretted as the union
            of those classes. [(1, 2), 3] means classes 1 and 2 get returned
            as one class and class 3 gets returned as the other class.
        X (np.ndarray): The feature matrix
        y (np.ndarray): The response vector
        binarize (bool): If true, returns classes[0] as 0 and classes[1] as 1.
            Otherwise, return selected classes unchanged.
    """
    y = y.flatten()
    mask = np.zeros(len(y), dtype=np.bool)
    for c in classes:
        mask |= (y == c)
    X_sub = X[mask]
    y_sub = y
    if binarize:
        assert len(set(classes)) == 2
        y_sub[mask & (y == classes[0])] = 0
        y_sub[mask & (y == classes[1])] = 1
    y_sub = y_sub[mask]
    return X_sub, y_sub


def select_dataset_samples(X, y, n_samples_per_class):
    return X[:2*n_samples_per_class], y[:2*n_samples_per_class]
