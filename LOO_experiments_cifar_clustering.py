"""
This performs clustering and then samples from those clusters in a stratified
manner
"""
import augmentations
import dataset_loaders
import experiments_util

from pretrained_experiments import run_test_clustered

import tensorflow as tf
from keras import backend as K


def main():
    rounds = 5
    n_aug_sample_points = [1, 10, 50, 100, 250, 500, 750, 1000]
    n_train = 1000
    n_jobs = 1
    cv = 1
    use_GPU = True
    batch_size = 128
    CNN_extractor_max_iter = 40
    use_loss = False
    n_clusters_arr = [1, 10, 50, 100, 250, 500, 750, 1000]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    # Can use multiple valus of C for cross-validation
    logistic_reg__Cs = [[10]]
    classes_datasets = [
        ((0, 1), dataset_loaders.Dataset.CIFAR10),
    ]
    selected_augmentations = [
        (augmentations.Image_Transformation.translate, {"mag_aug": 3}),
        (augmentations.Image_Transformation.rotate, {"mag_aug": 5,
                                                     "n_rotations": 4}),
        (augmentations.Image_Transformation.crop, {"mag_augs": [2]}),
    ]
    experiment_configs = [
        ("baseline", False, False),
        ("random_proportional", False, False),
        #("random_proportional", False, True),
        #("random_proportional", True, False),
        #("random_proportional", True, True),
        #("random_inverse_proportional", False, False),
        # ("random_inverse_proportional", True, False),
        # ("random_softmax_proportional", False, False),
        # ("random_softmax_proportional", False, True),
        # ("random_softmax_proportional", True, False),
        # ("random_softmax_proportional", True, True),
        # ("random_inverse_softmax_proportional", False, False),
        # ("random_inverse_softmax_proportional", True, False),
        #("deterministic_proportional", False, False),
        #("deterministic_proportional", False, True),
        #("deterministic_proportional", True, False),
        #("deterministic_proportional", True, True),
        #("deterministic_inverse_proportional", False, False),
        #("deterministic_inverse_proportional", True, False),
    ]
    for n_clusters in n_clusters_arr:
        for logistic_reg__C in logistic_reg__Cs:
            for classes, dataset in classes_datasets:
                for aug_transformation, aug_kw_args in selected_augmentations:
                    dataset_class_str = experiments_util.classes_to_class_str(
                        classes
                    )
                    print("Class types: {}".format(dataset_class_str))
                    reg_str = "-".join(list(map(str, logistic_reg__C)))
                    results_filename = "aug_results_{}_{}_{}_{}_{}{}".format(
                        dataset.name,
                        dataset_class_str,
                        aug_transformation.name,
                        reg_str,
                        n_clusters,
                        "_loss" if use_loss else "",
                    )
                    if dataset == dataset_loaders.Dataset.CIFAR10:
                        model_filename = "models/cifar10_ResNet56v2_model.h5"
                    elif dataset == dataset_loaders.Dataset.NORB:
                        if (aug_transformation ==
                                augmentations.Image_Transformation.translate):
                            model_filename = \
                                "models/norb_small_ResNet56v2_model_translate.h5"
                        else:
                            model_filename = \
                                "models/norb_small_ResNet56v2_model_rotate_crop.h5"
                    else:
                        raise RuntimeError("Unknown model for configuration")

                    run_test_clustered(
                        classes,
                        rounds,
                        n_aug_sample_points,
                        n_train,
                        n_jobs,
                        cv,
                        use_GPU,
                        batch_size,
                        dataset,
                        aug_transformation,
                        aug_kw_args,
                        logistic_reg__C,
                        CNN_extractor_max_iter,
                        use_loss,
                        experiment_configs,
                        results_filename,
                        model_filename,
                        n_clusters,
                    )
    use_loss = True
    for n_clusters in n_clusters_arr:
        for logistic_reg__C in logistic_reg__Cs:
            for classes, dataset in classes_datasets:
                for aug_transformation, aug_kw_args in selected_augmentations:
                    dataset_class_str = experiments_util.classes_to_class_str(
                        classes
                    )
                    print("Class types: {}".format(dataset_class_str))
                    reg_str = "-".join(list(map(str, logistic_reg__C)))
                    results_filename = "aug_results_{}_{}_{}_{}_{}{}".format(
                        dataset.name,
                        dataset_class_str,
                        aug_transformation.name,
                        reg_str,
                        n_clusters,
                        "_loss" if use_loss else "",
                    )
                    if dataset == dataset_loaders.Dataset.CIFAR10:
                        model_filename = "models/cifar10_ResNet56v2_model.h5"
                    elif dataset == dataset_loaders.Dataset.NORB:
                        if (aug_transformation ==
                                augmentations.Image_Transformation.translate):
                            model_filename = \
                                "models/norb_small_ResNet56v2_model_translate.h5"
                        else:
                            model_filename = \
                                "models/norb_small_ResNet56v2_model_rotate_crop.h5"
                    else:
                        raise RuntimeError("Unknown model for configuration")

                    run_test_clustered(
                        classes,
                        rounds,
                        n_aug_sample_points,
                        n_train,
                        n_jobs,
                        cv,
                        use_GPU,
                        batch_size,
                        dataset,
                        aug_transformation,
                        aug_kw_args,
                        logistic_reg__C,
                        CNN_extractor_max_iter,
                        use_loss,
                        experiment_configs,
                        results_filename,
                        model_filename,
                        n_clusters,
                    )


if __name__ == "__main__":
    main()
