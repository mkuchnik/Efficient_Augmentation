import pprint
import time

import keras
import numpy as np
import joblib

import dataset_loaders
import selection_policy
import augmentations
import experiments
import experiments_util
import featurized_classifiers
import visualization_util

import matplotlib.pyplot as plt

mem = joblib.Memory(cachedir="./cache", verbose=1)


def run_test(classes,
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
             ):

    run_params = {
        "classes": classes,
        "rounds": rounds,
        "n_aug_sample_points": n_aug_sample_points,
        "n_train": n_train,
        "n_jobs": n_jobs,
        "cv": cv,
        "use_GPU": use_GPU,
        "batch_size": batch_size,
        "dataset": dataset.name,
        "aug_transformation": aug_transformation.name,
        "aug_kw_args": aug_kw_args,
        "logistic_reg__C": logistic_reg__C,
        "CNN_extractor_max_iter": CNN_extractor_max_iter,
        "use_loss": use_loss,
        "experiment_configs": experiment_configs,
        "results_filename": results_filename,
    }

    pprint.pprint(run_params)

    assert n_aug_sample_points

    (x_train, y_train), (x_test, y_test) = experiments_util.prepare_dataset(
        dataset,
        classes,
        n_train,
    )
    print("Train class breakdown: {}".format(
        np.unique(y_train, return_counts=True))
    )
    print("Test class breakdown: {}".format(
        np.unique(y_test, return_counts=True))
    )

    aug_f = augmentations.get_transformation(aug_transformation)
    (orig_and_auged_x_train,
     orig_and_auged_y_train,
     orig_and_auged_idxs_train) = \
        experiments_util.poison_dataset(x_train,
                                        y_train,
                                        aug_f,
                                        aug_kw_args)
    (orig_and_auged_x_test,
     orig_and_auged_y_test,
     orig_and_auged_idxs_test) = \
        experiments_util.poison_dataset(x_test,
                                        y_test,
                                        aug_f,
                                        aug_kw_args)
    print("x_train", x_train.shape)
    print("orig_and_auged_x_train", orig_and_auged_x_train.shape)

    feature_clf = featurized_classifiers.build_featurized_ResNet_feature_clf(
        CNN_extractor_max_iter,
        use_GPU,
        batch_size)

    @mem.cache
    def transform_features(x, y):
        return feature_clf.fit_transform(x, y=y)

    featurized_x_train = transform_features(x=x_train, y=y_train)
    featurized_y_train = y_train
    featurized_x_test = transform_features(x=x_test, y=y_test)
    featurized_y_test = y_test
    orig_and_auged_featurized_x_train = transform_features(x=orig_and_auged_x_train,
                                                  y=orig_and_auged_y_train)
    orig_and_auged_featurized_y_train = orig_and_auged_y_train
    orig_and_auged_featurized_x_train_to_source_idxs = orig_and_auged_idxs_train
    orig_and_auged_featurized_x_test = transform_features(x=orig_and_auged_x_test,
                                                 y=orig_and_auged_y_test)
    orig_and_auged_featurized_y_test = orig_and_auged_y_test
    orig_and_auged_featurized_x_test_to_source_idxs = orig_and_auged_idxs_test

    clf = featurized_classifiers.build_logistic_reg_clf(
        logistic_reg__C,
        cv,
    )


    (no_aug_no_poison_acc,
     poisoned_acc,
     all_aug_train_poisoned_acc,
     aug_scores,
     after_aug_scores,
     best_params,
     training_total_time) = experiments_util.train_and_score_clf(
         clf,
         featurized_x_train,
         y_train,
         featurized_x_test,
         y_test,
         orig_and_auged_featurized_x_train,
         orig_and_auged_featurized_y_train,
         orig_and_auged_featurized_x_test,
         orig_and_auged_featurized_y_test,
         use_loss,
         cv,
     )
    training_end_time = time.time()

    img_ranks = np.argsort(np.abs(aug_scores))

    top_n = 100

    good_imgs = x_train[img_ranks][-top_n:]
    bad_imgs = x_train[img_ranks][:top_n]
    print("scores", aug_scores)
    print("scores", aug_scores.shape)
    print("ranks", img_ranks)
    print("ranks", img_ranks.shape)
    print("good", good_imgs.shape)
    print("bad", bad_imgs.shape)

    figures = {"{}".format(i): img for i, img in enumerate(good_imgs)}
    assert len(figures) == top_n
    visualization_util.plot_figures(figures, nrows=10, ncols=10)
    plt.savefig("good_images.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

    figures = {"{}".format(i): img for i, img in enumerate(bad_imgs)}
    assert len(figures) == top_n
    visualization_util.plot_figures(figures, nrows=10, ncols=10)
    plt.savefig("bad_images.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def main():
    rounds = 5
    #rounds = 3
    n_aug_sample_points = [1, 10, 50, 100, 250, 500, 750, 1000]
    n_train = 1000
    n_jobs = 1
    cv = 1
    use_GPU = True
    batch_size = 128
    CNN_extractor_max_iter = 40
 #   use_loss = False
    use_loss = True

    # Can use multiple valus of C for cross-validation
    logistic_reg__Cs = [[10], [100], [1000]]
    classes_datasets = [
#        ((0, 1), dataset_loaders.Dataset.NORB),
        ((0, 1), dataset_loaders.Dataset.CIFAR10),
    ]
    selected_augmentations = [
        #(augmentations.Image_Transformation.translate, {"mag_aug": 6}),
        (augmentations.Image_Transformation.translate, {"mag_aug": 3}),
        #(augmentations.Image_Transformation.rotate, {"mag_aug": 5,
        #                                             "n_rotations": 4}),
        #(augmentations.Image_Transformation.crop, {"mag_augs": [2]}),
    ]
    experiment_configs = [
        ("baseline", False, False),
        ("random_proportional", False, False),
        ("random_proportional", False, True),
        ("random_proportional", True, False),
        ("random_proportional", True, True),
        ("random_inverse_proportional", False, False),
        #("random_inverse_proportional", True, False),
        #("random_softmax_proportional", False, False),
        #("random_softmax_proportional", False, True),
        #("random_softmax_proportional", True, False),
        #("random_softmax_proportional", True, True),
        #("random_inverse_softmax_proportional", False, False),
        #("random_inverse_softmax_proportional", True, False),
        ("deterministic_proportional", False, False),
        ("deterministic_proportional", False, True),
        ("deterministic_proportional", True, False),
        ("deterministic_proportional", True, True),
        ("deterministic_inverse_proportional", False, False),
        ("deterministic_inverse_proportional", True, False),
    ]
    for logistic_reg__C in logistic_reg__Cs:
        for classes, dataset in classes_datasets:
            for aug_transformation, aug_kw_args in selected_augmentations:
                dataset_class_str = experiments_util.classes_to_class_str(classes)
                print("Class types: {}".format(dataset_class_str))
                reg_str = "-".join(list(map(str, logistic_reg__C)))
                results_filename = "aug_results_{}_{}_{}_{}{}".format(
                    dataset.name,
                    dataset_class_str,
                    aug_transformation.name,
                    reg_str,
                    "_loss" if use_loss else "",
                )

                run_test(classes,
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
                         )

    use_loss = False

    # Can use multiple valus of C for cross-validation
    logistic_reg__Cs = [[10], [100], [1000]]
    classes_datasets = [
#        ((0, 1), dataset_loaders.Dataset.NORB),
        ((0, 1), dataset_loaders.Dataset.CIFAR10),
    ]
    selected_augmentations = [
        #(augmentations.Image_Transformation.translate, {"mag_aug": 6}),
        #(augmentations.Image_Transformation.rotate, {"mag_aug": 5,
        #                                             "n_rotations": 4}),
        #(augmentations.Image_Transformation.crop, {"mag_augs": [2]}),
        (augmentations.Image_Transformation.compose, {"n_aug": 10}),
    ]
    experiment_configs = [
        ("baseline", False, False),
        ("random_proportional", False, False),
        ("random_proportional", False, True),
        ("random_proportional", True, False),
        ("random_proportional", True, True),
        ("random_inverse_proportional", False, False),
        #("random_inverse_proportional", True, False),
        #("random_softmax_proportional", False, False),
        #("random_softmax_proportional", False, True),
        #("random_softmax_proportional", True, False),
        #("random_softmax_proportional", True, True),
        #("random_inverse_softmax_proportional", False, False),
        #("random_inverse_softmax_proportional", True, False),
        ("deterministic_proportional", False, False),
        ("deterministic_proportional", False, True),
        ("deterministic_proportional", True, False),
        ("deterministic_proportional", True, True),
        ("deterministic_inverse_proportional", False, False),
        ("deterministic_inverse_proportional", True, False),
    ]
    for logistic_reg__C in logistic_reg__Cs:
        for classes, dataset in classes_datasets:
            for aug_transformation, aug_kw_args in selected_augmentations:
                dataset_class_str = experiments_util.classes_to_class_str(classes)
                print("Class types: {}".format(dataset_class_str))
                reg_str = "-".join(list(map(str, logistic_reg__C)))
                results_filename = "aug_results_{}_{}_{}_{}{}".format(
                    dataset.name,
                    dataset_class_str,
                    aug_transformation.name,
                    reg_str,
                    "_loss" if use_loss else "",
                )

                run_test(classes,
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
                         )

if __name__ == "__main__":
    main()
