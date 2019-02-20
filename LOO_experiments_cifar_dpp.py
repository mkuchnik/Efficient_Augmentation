import augmentations
import dataset_loaders
import experiments_util

from pretrained_experiments import run_test_dpp


def main():
    rounds = 5
    #n_aug_sample_points = [1, 10, 50, 100, 250, 500, 750, 1000]
    #n_aug_sample_points = [1, 10, 30, 50, 70, 100, 150, 250, 500]
    n_aug_sample_points = [1, 10, 30, 50, 70, 100, 150, 200, 250]
    #n_aug_sample_points = [x for x in range(1, 1001)]
    n_train = 1000
    n_jobs = 1
    cv = 1
    use_GPU = True
    batch_size = 128
    CNN_extractor_max_iter = 40
    use_loss = False
    dpp_normalize_features = True
    dpp_phi_scale = 1e3
    dpp_weight_by_scores_opts = [True, False]

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
#        ("baseline", False, False),
        ("random_proportional", False, False),
    ]
    for dpp_weight_by_scores in dpp_weight_by_scores_opts:
        for logistic_reg__C in logistic_reg__Cs:
            for classes, dataset in classes_datasets:
                for aug_transformation, aug_kw_args in selected_augmentations:
                    dataset_class_str = experiments_util.classes_to_class_str(
                        classes
                    )
                    print("Class types: {}".format(dataset_class_str))
                    reg_str = "-".join(list(map(str, logistic_reg__C)))
                    if not dpp_weight_by_scores:
                        results_filename = "aug_results_dpp_raw_norm_{}_{}_{}_{}{}".format(
                            dataset.name,
                            dataset_class_str,
                            aug_transformation.name,
                            reg_str,
                            "_loss" if use_loss else "",
                        )
                    else:
                        results_filename = "aug_results_dpp_norm_{}_{}_{}_{}{}".format(
                            dataset.name,
                            dataset_class_str,
                            aug_transformation.name,
                            reg_str,
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

                    run_test_dpp(
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
                        dpp_weight_by_scores,
                        dpp_normalize_features,
                        dpp_phi_scale,
                    )


if __name__ == "__main__":
    main()
