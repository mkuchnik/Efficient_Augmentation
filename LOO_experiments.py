import dataset_loaders
import augmentations
import experiments
import experiments_util


def main():
    rounds = 5
    n_aug_sample_points = [1, 10, 50, 100, 250, 500, 750, 1000]
    n_train = 1000
    n_jobs = 1
    cv = 1
    use_GPU = True
    batch_size = 512
    CNN_extractor_max_iter = 40
    use_loss = False

    # Can use multiple valus of C for cross-validation
    # logistic_reg__Cs = [[10], [100], [1000], [1e6]]
    logistic_reg__Cs = [[10]]
    classes_datasets = [
        ((3, 8), dataset_loaders.Dataset.MNIST),
        # ((0, 1), dataset_loaders.Dataset.CIFAR10),
    ]
    selected_augmentations = [
        (augmentations.Image_Transformation.translate, {"mag_aug": 2}),
        (augmentations.Image_Transformation.rotate, {"mag_aug": 30,
                                                     "n_rotations": 15}),
        (augmentations.Image_Transformation.crop,
         {"mag_augs": [1, 2, 3, 4, 5, 6]}),
    ]
    experiment_configs = [
        ("baseline", False, False),
        ("random_proportional", False, False),
        ("random_proportional", False, True),
        ("random_proportional", True, False),
        ("random_proportional", True, True),
        ("random_inverse_proportional", False, False),
        # ("random_inverse_proportional", True, False),
        # ("random_softmax_proportional", False, False),
        # ("random_softmax_proportional", False, True),
        # ("random_softmax_proportional", True, False),
        # ("random_softmax_proportional", True, True),
        # ("random_inverse_softmax_proportional", False, False),
        # ("random_inverse_softmax_proportional", True, False),
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
                dataset_class_str = experiments_util.classes_to_class_str(
                    classes
                )
                print("Class types: {}".format(dataset_class_str))
                reg_str = "-".join(list(map(str, logistic_reg__C)))
                results_filename = "aug_results_{}_{}_{}_{}{}".format(
                    dataset.name,
                    dataset_class_str,
                    aug_transformation.name,
                    reg_str,
                    "_loss" if use_loss else "",
                )

                experiments.run_test(
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
                )


if __name__ == "__main__":
    main()
