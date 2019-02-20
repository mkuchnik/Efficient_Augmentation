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

    clf = featurized_classifiers.build_featurized_LeNet_logistic_clf(
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        logistic_reg__C,
        cv,
        n_jobs,
    )

    (no_aug_no_poison_acc,
     poisoned_acc,
     all_aug_train_poisoned_acc,
     aug_scores,
     after_aug_scores,
     best_params,
     training_total_time) = experiments_util.train_and_score_clf(
         clf,
         x_train,
         y_train,
         x_test,
         y_test,
         orig_and_auged_x_train,
         orig_and_auged_y_train,
         orig_and_auged_x_test,
         orig_and_auged_y_test,
         use_loss,
         cv,
     )
    training_end_time = time.time()

    img_ranks = np.argsort(np.abs(aug_scores))

    top_n = 10

    good_imgs = x_train[img_ranks][-top_n:]
    bad_imgs = x_train[img_ranks][:top_n]
    print("scores", aug_scores)
    print("scores", aug_scores.shape)
    print("ranks", img_ranks)
    print("ranks", img_ranks.shape)
    print("good", good_imgs.shape)
    print("bad", bad_imgs.shape)

    # show good
    #figures = {"{}_{}".format(img_ranks[i],
    #                          aug_scores[i]): x_train[i]
    #           for i in good_img_idxs}
    figures = {"{}".format(i): img for i, img in enumerate(good_imgs)}
    assert len(figures) == top_n
    visualization_util.plot_figures(figures, nrows=1, ncols=10)
    plt.savefig("good_images.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

    figures = {"{}".format(i): img for i, img in enumerate(bad_imgs)}
    assert len(figures) == top_n
    visualization_util.plot_figures(figures, nrows=1, ncols=10)
    plt.savefig("bad_images.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


    # Refit to original
    clf.fit(x_train, y_train)
    clf = clf.named_steps["logistic_reg"]
    inf_matrix = clf.inf_up_loss_influence()
    # We don't want LOO
    np.fill_diagonal(inf_matrix, 0)
    print(inf_matrix)

    neighbors_n = 20

    print("before")
    print("good")
    for idx in good_img_idxs:
        print("original", idx)
        x_ = [x_train[idx]]
        x_aug_ = orig_and_auged_x_train[orig_and_auged_idxs_train == idx]
        experiments.show_aug_images(x_, x_aug_)
        most_inf_idx = np.argsort(inf_matrix[idx])
        most_inf_idx = most_inf_idx[most_inf_idx < neighbors_n]
        print("Most inf", most_inf_idx)
        x_aug_ = x_train[most_inf_idx]
        experiments.show_aug_images(x_, x_aug_)
        least_inf_idx = np.argsort(inf_matrix[idx])
        least_inf_idx = least_inf_idx[most_inf_idx > len(most_inf_idx) - neighbors_n]
        print("Least inf", least_inf_idx)
        x_aug_ = x_train[least_inf_idx]
        experiments.show_aug_images(x_, x_aug_)

    print("bad")
    for idx in bad_img_idxs:
        print("original", idx)
        x_ = [x_train[idx]]
        x_aug_ = orig_and_auged_x_train[orig_and_auged_idxs_train == idx]
        experiments.show_aug_images(x_, x_aug_)
        most_inf_idx = np.argmax(inf_matrix[idx])
        print("Most inf", most_inf_idx)
        x_ = [x_train[most_inf_idx]]
        x_aug_ = orig_and_auged_x_train[orig_and_auged_idxs_train == idx]
        experiments.show_aug_images(x_, x_aug_)
        least_inf_idx = np.argmin(inf_matrix[idx])
        print("Least inf", least_inf_idx)
        x_ = [x_train[least_inf_idx]]
        x_aug_ = orig_and_auged_x_train[orig_and_auged_idxs_train == idx]
        experiments.show_aug_images(x_, x_aug_)

    print("after")
    print("good")
    img_ranks = np.argsort(after_aug_scores[:n_train])
    for idx in good_img_idxs:
        print("original", idx)
        x_ = [x_train[idx]]
        x_aug_ = orig_and_auged_x_train[orig_and_auged_idxs_train == idx]
        experiments.show_aug_images(x_, x_aug_)
        most_inf_idx = np.argmax(inf_matrix[idx])
        print("Most inf", most_inf_idx)
        x_ = [x_train[most_inf_idx]]
        x_aug_ = orig_and_auged_x_train[orig_and_auged_idxs_train == idx]
        experiments.show_aug_images(x_, x_aug_)
        least_inf_idx = np.argmin(inf_matrix[idx])
        print("Least inf", least_inf_idx)
        x_ = [x_train[least_inf_idx]]
        x_aug_ = orig_and_auged_x_train[orig_and_auged_idxs_train == idx]
        experiments.show_aug_images(x_, x_aug_)

    print("bad")
    for idx in bad_img_idxs:
        print("original", idx)
        x_ = [x_train[idx]]
        x_aug_ = orig_and_auged_x_train[orig_and_auged_idxs_train == idx]
        experiments.show_aug_images(x_, x_aug_)
        most_inf_idx = np.argmax(inf_matrix[idx])
        print("Most inf", most_inf_idx)
        x_ = [x_train[most_inf_idx]]
        x_aug_ = orig_and_auged_x_train[orig_and_auged_idxs_train == idx]
        experiments.show_aug_images(x_, x_aug_)
        least_inf_idx = np.argmin(inf_matrix[idx])
        print("Least inf", least_inf_idx)
        x_ = [x_train[least_inf_idx]]
        x_aug_ = orig_and_auged_x_train[orig_and_auged_idxs_train == idx]
        experiments.show_aug_images(x_, x_aug_)

    print("source")
    print("good")
    img_ranks = np.argsort(after_aug_scores[n_train:])
    img_ranks = orig_and_auged_idxs_train[img_ranks]
    for idx in good_img_idxs:
        print("original", idx)
        x_ = [x_train[idx]]
        x_aug_ = orig_and_auged_x_train[orig_and_auged_idxs_train == idx]
        experiments.show_aug_images(x_, x_aug_)
        most_inf_idx = np.argmax(inf_matrix[idx])
        print("Most inf", most_inf_idx)
        x_ = [x_train[most_inf_idx]]
        x_aug_ = orig_and_auged_x_train[orig_and_auged_idxs_train == idx]
        experiments.show_aug_images(x_, x_aug_)
        least_inf_idx = np.argmin(inf_matrix[idx])
        print("Least inf", least_inf_idx)
        x_ = [x_train[least_inf_idx]]
        x_aug_ = orig_and_auged_x_train[orig_and_auged_idxs_train == idx]
        experiments.show_aug_images(x_, x_aug_)

    print("bad")
    for idx in bad_img_idxs:
        print("original", idx)
        x_ = [x_train[idx]]
        x_aug_ = orig_and_auged_x_train[orig_and_auged_idxs_train == idx]
        experiments.show_aug_images(x_, x_aug_)
        most_inf_idx = np.argmax(inf_matrix[idx])
        print("Most inf", most_inf_idx)
        x_ = [x_train[most_inf_idx]]
        x_aug_ = orig_and_auged_x_train[orig_and_auged_idxs_train == idx]
        experiments.show_aug_images(x_, x_aug_)
        least_inf_idx = np.argmin(inf_matrix[idx])
        print("Least inf", least_inf_idx)
        x_ = [x_train[least_inf_idx]]
        x_aug_ = orig_and_auged_x_train[orig_and_auged_idxs_train == idx]
        experiments.show_aug_images(x_, x_aug_)

def main():
    rounds = 5
    #rounds = 3
    n_aug_sample_points = [1, 10, 50, 100, 250, 500, 750, 1000]
    n_train = 1000
    n_jobs = 1
    cv = 1
    use_GPU = True
    batch_size = 512
    CNN_extractor_max_iter = 40
    use_loss = False
    #use_loss = True

    # Can use multiple valus of C for cross-validation
    #logistic_reg__Cs = [[10], [100], [1000], [1e6]]
    logistic_reg__Cs = [[10]]
    classes_datasets = [
        ((3, 8), dataset_loaders.Dataset.MNIST),
        #((0, 1), dataset_loaders.Dataset.CIFAR10),
    ]
    selected_augmentations = [
        (augmentations.Image_Transformation.translate, {"mag_aug": 2}),
        (augmentations.Image_Transformation.rotate, {"mag_aug": 30,
                                                     "n_rotations": 15}),
        (augmentations.Image_Transformation.crop, {"mag_augs": [1, 2, 3, 4, 5, 6]}),
    ]
    experiment_configs = [
        #("baseline", False, False),
        #("random_proportional", False, False),
        #("random_proportional", False, True),
        #("random_proportional", True, False),
        #("random_proportional", True, True),
        #("random_inverse_proportional", False, False),
        #("random_inverse_proportional", True, False),
        #("random_softmax_proportional", False, False),
        #("random_softmax_proportional", False, True),
        #("random_softmax_proportional", True, False),
        #("random_softmax_proportional", True, True),
        #("random_inverse_softmax_proportional", False, False),
        #("random_inverse_softmax_proportional", True, False),
        #("deterministic_proportional", False, False),
        #("deterministic_proportional", False, True),
        #("deterministic_proportional", True, False),
        #("deterministic_proportional", True, True),
        ("deterministic_inverse_proportional", False, False),
        #("deterministic_inverse_proportional", True, False),
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
