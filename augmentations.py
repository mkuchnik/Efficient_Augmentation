import enum

import numpy as np

from imgaug import augmenters as iaa

class Image_Transformation(enum.Enum):
    translate = 1
    rotate = 2
    crop = 3


def get_transformation(transformation):
    if transformation == Image_Transformation.translate or \
            transformation == "translate":
        aug_f = transform_translate
    elif transformation == Image_Transformation.rotate or \
            transformation == "rotate":
        aug_f = transform_rotate
    elif transformation == Image_Transformation.crop or \
            transformation == "crop":
        aug_f = transform_crop
    else:
        raise ValueError("Unknown transformation: {}".format(transformation))
    return aug_f


def _transform_translate(images, x_mag_aug, y_mag_aug):
    augmentors = [
        iaa.Affine(
            translate_px={
                "x": (x_mag_aug, x_mag_aug),
                "y": (y_mag_aug, y_mag_aug),
            },
            mode="edge",
        ),
    ]
    seq = iaa.Sequential(augmentors)
    return seq.augment_images(images)


def transform_translate(X, y, mag_aug):
    mag_augs = [(mag_aug, mag_aug),
                (-mag_aug, mag_aug),
                (mag_aug, -mag_aug),
                (-mag_aug, -mag_aug),
                ]
    X_auged = []
    y_auged = []
    aug_idxs = []
    for x_mag_aug, y_mag_aug in mag_augs:
        X_aug = _transform_translate(X, x_mag_aug, y_mag_aug)
        y_aug = np.array(y)
        aug_idx = np.arange(len(X_aug))
        X_auged.append(X_aug)
        y_auged.append(y_aug)
        aug_idxs.append(aug_idx)
    X_auged = np.concatenate(
            X_auged,
            axis=0,
    )
    y_auged = np.concatenate(
            y_auged,
            axis=0,
    )
    aug_idxs = np.concatenate(
            aug_idxs,
            axis=0,
    )
    return aug_idxs, (X_auged, y_auged)


def _transform_rotate(images, mag_aug):
    augmentors = [
        iaa.Affine(
            rotate=(mag_aug, mag_aug),
            mode="edge",
        ),
    ]
    seq = iaa.Sequential(augmentors)
    return seq.augment_images(images)


def transform_rotate(X, y, mag_aug, n_rotations):
    """
    Drops identity rotates
    """
    assert mag_aug >= 0
    mag_augs = np.linspace(-mag_aug, mag_aug, n_rotations)
    mag_augs = mag_augs[np.nonzero(mag_augs)]

    X_auged = []
    y_auged = []
    aug_idxs = []
    for _mag_aug in mag_augs:
        X_aug = _transform_rotate(X, _mag_aug)
        y_aug = np.array(y)
        aug_idx = np.arange(len(X_aug))
        X_auged.append(X_aug)
        y_auged.append(y_aug)
        aug_idxs.append(aug_idx)
    X_auged = np.concatenate(
            X_auged,
            axis=0,
    )
    y_auged = np.concatenate(
            y_auged,
            axis=0,
    )
    aug_idxs = np.concatenate(
            aug_idxs,
            axis=0,
    )
    return aug_idxs, (X_auged, y_auged)


def _transform_crop(images, mag_aug):
    augmentors = [
        iaa.Crop(
            px=tuple((mag_aug, mag_aug) for i in range(4)),
            sample_independently=True,
            keep_size=True,
        ),
    ]
    seq = iaa.Sequential(augmentors)
    return seq.augment_images(images)


def transform_crop(X, y, mag_augs):
    """
    Drops identity rotates
    """
    X_auged = []
    y_auged = []
    aug_idxs = []
    for mag_aug in mag_augs:
        X_aug = _transform_crop(X, mag_aug)
        y_aug = np.array(y)
        aug_idx = np.arange(len(X_aug))
        X_auged.append(X_aug)
        y_auged.append(y_aug)
        aug_idxs.append(aug_idx)
    X_auged = np.concatenate(
            X_auged,
            axis=0,
    )
    y_auged = np.concatenate(
            y_auged,
            axis=0,
    )
    aug_idxs = np.concatenate(
            aug_idxs,
            axis=0,
    )
    return aug_idxs, (X_auged, y_auged)
