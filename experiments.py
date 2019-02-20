import numpy as np
import sklearn.model_selection
import logging

import collections
import pprint
import time

import selection_policy
import augmentations
import experiments
import experiments_util
import featurized_classifiers
import sklearn.cluster
import sklearn.preprocessing
import sample_dpp


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
             baseline_test=False,
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
        "baseline_test": baseline_test,
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
    print("x_train shape: {}".format(x_train.shape))
    print("orig_and_auged_x_train shape: {}".format(
        orig_and_auged_x_train.shape))

    clf = featurized_classifiers.build_featurized_LeNet_logistic_clf(
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        logistic_reg__C,
        cv,
        n_jobs,
    )

    svm__C = [0.01, 0.1, 1, 10, 100]
    svm_cv = 4
    is_SV = experiments_util.get_SV_featurized_LeNet(
        x_train,
        y_train,
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        svm__C,
        svm_cv,
        n_jobs
    )
    print("Number of support vectors is: {}".format(np.sum(is_SV)))
    SV_idxs = np.where(is_SV)[0]
    orig_and_SV_idxs = np.concatenate([SV_idxs, [-1]])
    print("orig_and_SV_idxs: {}".format(orig_and_SV_idxs))
    print("orig_and_SV_idxs shape: {}".format(orig_and_SV_idxs.shape))
    SV_orig_and_auged_mask = np.isin(orig_and_auged_idxs_train,
                                     orig_and_SV_idxs)
    SV_x_train = orig_and_auged_x_train[SV_orig_and_auged_mask]
    SV_y_train = orig_and_auged_y_train[SV_orig_and_auged_mask]
    clf.fit(SV_x_train, SV_y_train)
    VSV_acc = clf.score(orig_and_auged_x_test, orig_and_auged_y_test)
    print("VSV acc: {}".format(VSV_acc))

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

    if baseline_test:
        # If baseline test, we just use unmodified test set
        exp_x_test = x_test
        exp_y_test = y_test
    else:
        exp_x_test = orig_and_auged_x_test
        exp_y_test = orig_and_auged_y_test

    experiment_results = {}
    for policy_name, update_score, downweight_points in experiment_configs:
        policy_f = selection_policy.get_policy_by_name(policy_name)
        if "deterministic" in policy_name:
            _rounds = 1
        else:
            _rounds = rounds
        acc = experiments.aug_experiment_rounds(
            clf,
            x_train,
            y_train,
            exp_x_test,
            exp_y_test,
            policy_f,
            aug_scores,
            aug_f,
            aug_kw_args,
            n_aug_sample_points,
            _rounds,
            update_score,
            downweight_points,
            use_loss=use_loss,
        )
        config_name = [policy_name]
        if update_score:
            config_name.append("update")
        if downweight_points:
            config_name.append("downweight")
        config_name = "_".join(config_name)
        experiment_results[config_name] = acc

    all_results = {
        "no_aug_no_poison_acc": no_aug_no_poison_acc,
        "poisoned_acc": poisoned_acc,
        "all_aug_train_poisoned_acc": all_aug_train_poisoned_acc,
        "is_SV": is_SV,
        "VSV_acc": VSV_acc,
        "best_params": best_params,
        "initial_aug_scores": aug_scores,
        "after_aug_scores": after_aug_scores,
        "experiment_results": experiment_results,
        "n_aug_sample_points": n_aug_sample_points,
        "run_parameters": run_params,
        "n_train": n_train,
        "rounds": rounds,
    }

    tests_total_time = time.time() - training_end_time

    all_results["tests_total_runtime"] = tests_total_time

    pprint.pprint(all_results)
    np.savez(results_filename,
             **all_results,
             )

    print("*" * 80)
    print("Training took {} seconds".format(training_total_time))
    print("All tests took {} seconds".format(tests_total_time))
    print("*" * 80)


def show_aug_images(x_, x_aug_):
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(len(x_aug_), 2)
    if len(x_) > 1:
        logging.warning("x_ has shape '{}' which is greater than"
                        " length 1".format(x_.shape))
    x_show = x_[0]
    if x_show.shape[2] < 3:
        x_show = x_show.reshape(x_show.shape[:2])
        ax[0, 0].imshow(x_show,
                        cmap="gray")
    else:
        ax[0, 0].imshow(x_show)
    for i in range(len(x_aug_)):
        ax[i, 0].axis("off")
    for i, x_i in enumerate(x_aug_):
        if x_aug_[i].shape[2] < 3:
            ax[i, 1].imshow(x_aug_[i].reshape(x_aug_[i].shape[:2]),
                            cmap="gray")
        else:
            ax[i, 1].imshow(x_aug_[i])
        ax[i, 1].axis("off")
    plt.show()


def aug_experiment(clf, x_train, y_train, auged_x_test, auged_y_test,
                   aug_iter, train_idxs_scores, aug_f, aug_kw_args,
                   n_aug_sample_points, update_LOO=False,
                   weight_aug_samples=False, use_loss=False,
                   show_aug_images=False,
                   stratified_sampling_x_train_ks=None,
                   ):
    auged_x_train = np.copy(x_train)
    auged_y_train = np.copy(y_train)
    if weight_aug_samples:
        sample_weight = np.ones(len(x_train))
    else:
        sample_weight = None
    influence_acc = []
    n_aug_sample_points = set(n_aug_sample_points)
    assert len(y_train) == len(x_train)
    if stratified_sampling_x_train_ks is not None:
        aug_idxs = stratified_sampling_to_aug_idxs(
            train_idxs_scores,
            aug_iter,
            stratified_sampling_x_train_ks,
        )
    else:
        aug_idxs = np.array(
            list(aug_iter(train_idxs_scores))
                            ).flatten()
    already_auged = set()
    while len(already_auged) < len(x_train):
        assert len(train_idxs_scores) == len(x_train)
        next_idxs = [idx for idx in aug_idxs if idx not in already_auged]
        idx = next_idxs[0]
        already_auged.add(idx)
        idx = [idx]
        x_ = x_train[idx]
        y_ = y_train[idx]
        aug_idxs_, (x_aug_, y_aug_) = aug_f(x_, y_, **aug_kw_args)
        if show_aug_images:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(len(x_aug_), 2)
            if len(x_) > 1:
                logging.warning("x_ has shape '{}' which is greater than"
                                " length 1".format(x_.shape))
            x_show = x_[0]
            if x_show.shape[2] < 3:
                x_show = x_show.reshape(x_show.shape[:2])
                ax[0, 0].imshow(x_show,
                                cmap="gray")
            else:
                ax[0, 0].imshow(x_show)
            for i in range(len(x_aug_)):
                ax[i, 0].axis("off")
            for i, x_i in enumerate(x_aug_):
                if x_aug_[i].shape[2] < 3:
                    ax[i, 1].imshow(x_aug_[i].reshape(x_aug_[i].shape[:2]),
                                    cmap="gray")
                else:
                    ax[i, 1].imshow(x_aug_[i])
                ax[i, 1].axis("off")
            plt.show()
        auged_x_train = np.concatenate([
                auged_x_train,
                x_aug_,
            ],
            axis=0)
        auged_y_train = np.concatenate([
                auged_y_train,
                y_aug_,
            ],
            axis=0)
        if weight_aug_samples:
            # We downweight all points from the original train point
            rescale_weight = 1.0 / (len(x_aug_) + 1)
            weight_aug_ = np.full(len(x_aug_), rescale_weight)
            sample_weight = np.concatenate([
                    sample_weight,
                    weight_aug_,
                ],
                axis=0)
            sample_weight[idx] = rescale_weight
        if len(already_auged) in n_aug_sample_points:
            fit_params = {"logistic_reg__sample_weight": sample_weight}
            clf.fit(auged_x_train, auged_y_train, **fit_params)
            aug_train_poisoned_acc = clf.score(
                auged_x_test,
                auged_y_test)
            influence_acc.append(aug_train_poisoned_acc)
            if update_LOO:
                if isinstance(clf, sklearn.model_selection.GridSearchCV):
                    if use_loss:
                        train_idxs_scores = (clf
                                             .best_estimator_
                                             .named_steps["logistic_reg"]
                                             .log_losses(L2_alpha=0.0))
                    else:
                        train_idxs_scores = (clf
                                             .best_estimator_
                                             .named_steps["logistic_reg"]
                                             .LOO_influence())
                else:
                    if use_loss:
                        train_idxs_scores = (clf
                                             .named_steps["logistic_reg"]
                                             .log_losses(L2_alpha=0.0))
                    else:
                        train_idxs_scores = (clf
                                             .named_steps["logistic_reg"]
                                             .LOO_influence())
                train_idxs_scores = train_idxs_scores[:len(x_train)]
                if stratified_sampling_x_train_ks is not None:
                    aug_idxs = stratified_sampling_to_aug_idxs(
                        train_idxs_scores,
                        aug_iter,
                        stratified_sampling_x_train_ks,
                    )
                else:
                    aug_idxs = np.array(
                        list(aug_iter(train_idxs_scores))
                    ).flatten()
    return influence_acc


def aug_experiment_rounds(clf, x_train, y_train, auged_x_test, auged_y_test,
                          aug_iter, LOO_influences,
                          aug_f, aug_kw_args, n_aug_sample_points, rounds,
                          update_LOO=False,
                          weight_aug_samples=False,
                          use_loss=False,
                          stratified_sampling_x_train_ks=None):
    all_accs = []
    for r in range(rounds):
        acc = aug_experiment(clf,
                             x_train,
                             y_train,
                             auged_x_test,
                             auged_y_test,
                             aug_iter,
                             LOO_influences,
                             aug_f,
                             aug_kw_args,
                             n_aug_sample_points,
                             update_LOO,
                             weight_aug_samples,
                             use_loss,
                             False,
                             stratified_sampling_x_train_ks,
                             )
        all_accs.append(acc)
    return all_accs


def stratified_sampling_to_aug_idxs(
        train_idxs_scores,
        aug_iter,
        stratified_sampling_x_train_ks,
):
    """
    Creates an ordering of sampling by using aug_iter in a round-robin fashion
    over all populations.
    """
    stratified_aug_idxs_arr = []
    stratified_train_rev_idxs_arr = []
    for k in sorted(np.unique(stratified_sampling_x_train_ks)):
        is_stratified_train = stratified_sampling_x_train_ks == k
        stratified_train_rev_idxs = np.where(is_stratified_train)[0]
        stratified_train_rev_idxs_arr.append(stratified_train_rev_idxs)
        stratified_train_idxs_scores = train_idxs_scores[
            is_stratified_train
        ]
        stratified_aug_idxs = np.array(
            list(aug_iter(stratified_train_idxs_scores))
        ).flatten()
        stratified_aug_idxs_arr.append(stratified_aug_idxs)
    aug_idxs = []
    has_elements = True
    strat_aug_idxs_i = 0
    while has_elements:
        for i, strat_idxs in enumerate(stratified_aug_idxs_arr):
            if strat_aug_idxs_i < len(strat_idxs):
                strat_idx = strat_idxs[strat_aug_idxs_i]
                strat_idx = stratified_train_rev_idxs_arr[i][strat_idx]
                aug_idxs.append(strat_idx)
        has_elements = np.any(
            list(map(lambda x: strat_aug_idxs_i < len(x),
                     stratified_aug_idxs_arr,
                     )
                 )
        )
        strat_aug_idxs_i += 1
    assert len(train_idxs_scores) == len(aug_idxs)
    return aug_idxs


def precomputed_aug_experiment(
        clf,
        auged_featurized_x_train,
        auged_featurized_y_train,
        auged_featurized_x_train_to_source_idxs,
        auged_featurized_x_test,
        auged_featurized_y_test,
        auged_featurized_x_test_to_source_idxs,
        aug_iter,
        train_idxs_scores,
        n_aug_sample_points,
        update_scores=False,
        weight_aug_samples=False,
        use_loss=False,
        stratified_sampling_x_train_ks=None,
):
    """
    This is a precomputed version of the aug_experiment.
    Here, we expect training sets to be augmented and featurized up front.
    This function will index into the augmented set (with featurization)
    to get the input that would be fed into the classifier.

    @param clf The classifier to use (e.g., logistic regression)
    @param auged_featurized_x_train The augmented and featurized training set.
    @param auged_featurized_y_train The labels of the training set.
    @param auged_featurized_x_train_to_source_idxs A list of idxs corresponding
    to the source of augmented images from the original training set. -1 means
    that the point is an original point.
    @param auged_featurized_x_test The augmented and featurized test set.
    @param auged_featurized_y_test The labels of the test set.
    @param auged_featurized_x_test_to_source_idxs A list of idxs corresponding
    to the source of augmented images from the original test set. -1 means
    that the point is an original point.
    @param aug_iter The policy to use.
    @param train_idxs_scores The scores to use for the policies (e.g.,
    LOO influence or loss).
    @param stratified_sampling_x_train_ks The population type of each train
    sample for stratified sampling. Sampling is round robin in numeric order.

    @return An list of accuracies on the test set and a list of the points that
    were chosen for augmentation.
    """
    influence_acc = []
    aug_iter_idxs = []
    original_mask_train = auged_featurized_x_train_to_source_idxs < 0
    original_x_train = auged_featurized_x_train[original_mask_train]
    original_y_train = auged_featurized_y_train[original_mask_train]
    auged_x_train = np.copy(original_x_train)
    auged_y_train = np.copy(original_y_train)
    n_aug_sample_points = set(n_aug_sample_points)
    if weight_aug_samples:
        sample_weight = np.ones(len(original_x_train))
    else:
        sample_weight = None
    if stratified_sampling_x_train_ks is not None:
        aug_idxs = stratified_sampling_to_aug_idxs(
            train_idxs_scores,
            aug_iter,
            stratified_sampling_x_train_ks,
        )
    else:
        aug_idxs = np.array(list(aug_iter(train_idxs_scores))).flatten()
    assert len(np.unique(aug_idxs)) == len(aug_idxs)
    already_auged = set()
    while len(already_auged) < len(original_x_train):
        assert len(train_idxs_scores) == len(original_x_train)
        next_idxs = [idx for idx in aug_idxs if idx not in already_auged]
        idx = next_idxs[0]
        already_auged.add(idx)
        aug_mask = auged_featurized_x_train_to_source_idxs == idx
        x_aug_ = auged_featurized_x_train[aug_mask]
        auged_x_train = np.concatenate(
            [
                auged_x_train,
                x_aug_,
            ],
            axis=0)
        y_aug_ = auged_featurized_y_train[aug_mask]
        auged_y_train = np.concatenate(
            [
                auged_y_train,
                y_aug_,
            ],
            axis=0)
        if weight_aug_samples:
            # We downweight all points from the original train point
            rescale_weight = 1.0 / (len(x_aug_) + 1)
            weight_aug_ = np.full(len(x_aug_), rescale_weight)
            sample_weight = np.concatenate([
                    sample_weight,
                    weight_aug_,
                ],
                axis=0)
            sample_weight[idx] = rescale_weight
        if len(already_auged) in n_aug_sample_points:
            fit_params = {"logistic_reg__sample_weight": sample_weight}
            clf.fit(auged_x_train, auged_y_train, **fit_params)
            aug_train_poisoned_acc = clf.score(
                auged_featurized_x_test,
                auged_featurized_y_test)
            influence_acc.append(aug_train_poisoned_acc)
            aug_iter_idxs.append(idx)
            if update_scores:
                if isinstance(clf, sklearn.model_selection.GridSearchCV):
                    if use_loss:
                        train_idxs_scores = (clf
                                             .best_estimator_
                                             .named_steps["logistic_reg"]
                                             .log_losses(L2_alpha=0.0))
                    else:
                        train_idxs_scores = (clf
                                             .best_estimator_
                                             .named_steps["logistic_reg"]
                                             .LOO_influence())
                else:
                    if use_loss:
                        train_idxs_scores = (clf
                                             .named_steps["logistic_reg"]
                                             .log_losses(L2_alpha=0.0))
                    else:
                        train_idxs_scores = (clf
                                             .named_steps["logistic_reg"]
                                             .LOO_influence())
                train_idxs_scores = train_idxs_scores[:len(original_x_train)]
                if stratified_sampling_x_train_ks is not None:
                    aug_idxs = stratified_sampling_to_aug_idxs(
                        train_idxs_scores,
                        aug_iter,
                        stratified_sampling_x_train_ks,
                    )
                else:
                    aug_idxs = np.array(
                        list(aug_iter(train_idxs_scores))
                    ).flatten()
    return influence_acc, aug_iter_idxs


def precomputed_aug_experiment_rounds(
        clf,
        auged_featurized_x_train,
        auged_featurized_y_train,
        auged_featurized_x_train_to_source_idxs,
        auged_featurized_x_test,
        auged_featurized_y_test,
        auged_featurized_x_test_to_source_idxs,
        aug_iter,
        train_idxs_scores,
        n_aug_sample_points,
        rounds,
        update_scores=False,
        weight_aug_samples=False,
        use_loss=False,
        stratified_sampling_x_train_ks=None,
):
    all_accs = []
    all_idxs = []
    for r in range(rounds):
        acc, idxs = precomputed_aug_experiment(
            clf,
            auged_featurized_x_train,
            auged_featurized_y_train,
            auged_featurized_x_train_to_source_idxs,
            auged_featurized_x_test,
            auged_featurized_y_test,
            auged_featurized_x_test_to_source_idxs,
            aug_iter,
            train_idxs_scores,
            n_aug_sample_points,
            update_scores,
            weight_aug_samples,
            use_loss,
            stratified_sampling_x_train_ks,
        )
        all_accs.append(acc)
        all_idxs.append(idxs)
    return all_accs, all_idxs


def run_test_clustered(
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
        n_clusters,
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
        "n_clusters": n_clusters,
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

    print("x_train shape: {}".format(x_train.shape))
    print("orig_and_auged_x_train shape: {}".format(
        orig_and_auged_x_train.shape))

    clf = featurized_classifiers.build_featurized_LeNet_logistic_clf(
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        logistic_reg__C,
        cv,
        n_jobs,
    )

    svm__C = [0.01, 0.1, 1, 10, 100]
    svm_cv = 4
    is_SV = experiments_util.get_SV_featurized_LeNet(
        x_train,
        y_train,
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        svm__C,
        svm_cv,
        n_jobs
    )
    print("Number of support vectors is: {}".format(np.sum(is_SV)))
    SV_idxs = np.where(is_SV)[0]
    orig_and_SV_idxs = np.concatenate([SV_idxs, [-1]])
    print("orig_and_SV_idxs: {}".format(orig_and_SV_idxs))
    print("orig_and_SV_idxs shape: {}".format(orig_and_SV_idxs.shape))
    SV_orig_and_auged_mask = np.isin(orig_and_auged_idxs_train,
                                     orig_and_SV_idxs)
    SV_x_train = orig_and_auged_x_train[SV_orig_and_auged_mask]
    SV_y_train = orig_and_auged_y_train[SV_orig_and_auged_mask]
    clf.fit(SV_x_train, SV_y_train)
    VSV_acc = clf.score(orig_and_auged_x_test, orig_and_auged_y_test)
    print("VSV acc: {}".format(VSV_acc))

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

    featurizer = sklearn.pipeline.Pipeline([
        ("image_rescaler", (clf.named_steps["image_rescaler"])),
        ("feature_map", clf.named_steps["feature_map"]),
    ])
    clustering_clf = sklearn.cluster.KMeans(n_clusters=n_clusters)
    featurized_x_train = featurizer.transform(x_train)
    print("featurized_x_train", featurized_x_train.shape)
    train_cluster_IDs = clustering_clf.fit_predict(featurized_x_train)
    # test_cluster_IDs = clustering_clf.predict(featurized_x_test)

    training_end_time = time.time()

    experiment_results = {}
    for policy_name, update_score, downweight_points in experiment_configs:
        policy_f = selection_policy.get_policy_by_name(policy_name)
        if "deterministic" in policy_name:
            _rounds = 1
        else:
            _rounds = rounds
        acc = experiments.aug_experiment_rounds(
            clf,
            x_train,
            y_train,
            orig_and_auged_x_test,
            orig_and_auged_y_test,
            policy_f,
            aug_scores,
            aug_f,
            aug_kw_args,
            n_aug_sample_points,
            _rounds,
            update_score,
            downweight_points,
            use_loss=use_loss,
            stratified_sampling_x_train_ks=train_cluster_IDs,
        )
        config_name = [policy_name]
        if update_score:
            config_name.append("update")
        if downweight_points:
            config_name.append("downweight")
        config_name = "_".join(config_name)
        experiment_results[config_name] = acc

    all_results = {
        "no_aug_no_poison_acc": no_aug_no_poison_acc,
        "poisoned_acc": poisoned_acc,
        "all_aug_train_poisoned_acc": all_aug_train_poisoned_acc,
        "is_SV": is_SV,
        "VSV_acc": VSV_acc,
        "best_params": best_params,
        "initial_aug_scores": aug_scores,
        "after_aug_scores": after_aug_scores,
        "experiment_results": experiment_results,
        "n_aug_sample_points": n_aug_sample_points,
        "run_parameters": run_params,
        "n_train": n_train,
        "rounds": rounds,
    }

    tests_total_time = time.time() - training_end_time

    all_results["tests_total_runtime"] = tests_total_time

    pprint.pprint(all_results)
    np.savez(results_filename,
             **all_results,
             )

    print("*" * 80)
    print("Training took {} seconds".format(training_total_time))
    print("All tests took {} seconds".format(tests_total_time))
    print("*" * 80)


def run_test_margin(
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
        ):
    """
    Uses SVM margin for score
    """

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
    print("x_train shape: {}".format(x_train.shape))
    print("orig_and_auged_x_train shape: {}".format(
        orig_and_auged_x_train.shape
    ))

    clf = featurized_classifiers.build_featurized_LeNet_logistic_clf(
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        logistic_reg__C,
        cv,
        n_jobs,
    )

    svm__C = [0.01, 0.1, 1, 10, 100]
    svm_cv = 4
    is_SV = experiments_util.get_SV_featurized_LeNet(
        x_train,
        y_train,
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        svm__C,
        svm_cv,
        n_jobs
    )
    SVM_margins = experiments_util.get_SVM_margins_featurized_LeNet(
        x_train,
        y_train,
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        svm__C,
        svm_cv,
        n_jobs
    )
    print("SVM margins: {}".format(SVM_margins))
    print("Number of support vectors is: {}".format(np.sum(is_SV)))
    SV_idxs = np.where(is_SV)[0]
    orig_and_SV_idxs = np.concatenate([SV_idxs, [-1]])
    print("orig_and_SV_idxs: {}".format(orig_and_SV_idxs))
    print("orig_and_SV_idxs shape: {}".format(orig_and_SV_idxs.shape))
    SV_orig_and_auged_mask = np.isin(orig_and_auged_idxs_train,
                                     orig_and_SV_idxs)
    SV_x_train = orig_and_auged_x_train[SV_orig_and_auged_mask]
    SV_y_train = orig_and_auged_y_train[SV_orig_and_auged_mask]
    clf.fit(SV_x_train, SV_y_train)
    VSV_acc = clf.score(orig_and_auged_x_test, orig_and_auged_y_test)
    print("VSV acc: {}".format(VSV_acc))

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

    # Here we use margins
    aug_scores = np.abs(SVM_margins)
    print("Aug scores: {}".format(aug_scores))
    experiment_results = {}
    for policy_name, update_score, downweight_points in experiment_configs:
        policy_f = selection_policy.get_policy_by_name(policy_name)
        if "deterministic" in policy_name:
            _rounds = 1
        else:
            _rounds = rounds
        acc = experiments.aug_experiment_rounds(
            clf,
            x_train,
            y_train,
            orig_and_auged_x_test,
            orig_and_auged_y_test,
            policy_f,
            aug_scores,
            aug_f,
            aug_kw_args,
            n_aug_sample_points,
            _rounds,
            update_score,
            downweight_points,
            use_loss=use_loss,
        )
        config_name = [policy_name]
        if update_score:
            config_name.append("update")
        if downweight_points:
            config_name.append("downweight")
        config_name = "_".join(config_name)
        experiment_results[config_name] = acc

    all_results = {
        "no_aug_no_poison_acc": no_aug_no_poison_acc,
        "poisoned_acc": poisoned_acc,
        "all_aug_train_poisoned_acc": all_aug_train_poisoned_acc,
        "is_SV": is_SV,
        "VSV_acc": VSV_acc,
        "best_params": best_params,
        "initial_aug_scores": aug_scores,
        "after_aug_scores": after_aug_scores,
        "experiment_results": experiment_results,
        "n_aug_sample_points": n_aug_sample_points,
        "run_parameters": run_params,
        "n_train": n_train,
        "rounds": rounds,
    }

    tests_total_time = time.time() - training_end_time

    all_results["tests_total_runtime"] = tests_total_time

    pprint.pprint(all_results)
    np.savez(results_filename,
             **all_results,
             )

    print("*" * 80)
    print("Training took {} seconds".format(training_total_time))
    print("All tests took {} seconds".format(tests_total_time))
    print("*" * 80)


def run_test_clustered_sweep(
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
        ):
    """
    Gets intertia and silhouette score for clusters
    """

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
    print("x_train: {}".format(x_train.shape))
    print("orig_and_auged_x_train: {}".format(orig_and_auged_x_train.shape))

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

    featurizer = sklearn.pipeline.Pipeline([
        ("image_rescaler", (clf.named_steps["image_rescaler"])),
        ("feature_map", clf.named_steps["feature_map"]),
    ])
    featurized_x_train = featurizer.transform(x_train)
    print("featurized_x_train: {}".format(featurized_x_train.shape))
    featurized_x_test = featurizer.transform(x_test)

    all_results = collections.defaultdict(list)
    n_clusters_arr = np.unique(
        np.clip(
            np.around(
                np.geomspace(2,
                             len(featurized_x_train) - 1,
                             num=50)
            ).astype(int),
            2,
            len(featurized_x_train) - 1
        ),
    )
    print("n_clusters_arr: {}".format(n_clusters_arr))
    assert np.all(n_clusters_arr < len(featurized_x_train))

    for n_clusters in n_clusters_arr:
        print("n_clusters: {}".format(n_clusters))
        clustering_clf = sklearn.cluster.KMeans(n_clusters=n_clusters)
        train_cluster_IDs = clustering_clf.fit_predict(featurized_x_train)
        test_cluster_IDs = clustering_clf.predict(featurized_x_test)
        train_silhouette_score = sklearn.metrics.silhouette_score(
            featurized_x_train,
            train_cluster_IDs,
            metric="euclidean",
        )
        train_inertia = clustering_clf.inertia_
        test_silhouette_score = sklearn.metrics.silhouette_score(
            featurized_x_test,
            test_cluster_IDs,
            metric="euclidean",
        )
        all_results["n_clusters"].append(n_clusters)
        all_results["train_inertia"].append(train_inertia)
        all_results["train_silhouette_score"].append(train_silhouette_score)
        all_results["test_silhouette_score"].append(test_silhouette_score)
        pprint.pprint(all_results)

    return all_results


def precomputed_aug_experiment_rounds_dpp(
        clf,
        auged_featurized_x_train,
        auged_featurized_y_train,
        auged_featurized_x_train_to_source_idxs,
        auged_featurized_x_test,
        auged_featurized_y_test,
        auged_featurized_x_test_to_source_idxs,
        train_idxs_scores,
        n_aug_sample_points,
        rounds,
        weight_by_scores,
        normalize_features,
        phi_scale,
):
    all_accs = []
    all_idxs = []
    for r in range(rounds):
        acc, idxs = precomputed_aug_experiment_dpp(
            clf,
            auged_featurized_x_train,
            auged_featurized_y_train,
            auged_featurized_x_train_to_source_idxs,
            auged_featurized_x_test,
            auged_featurized_y_test,
            auged_featurized_x_test_to_source_idxs,
            train_idxs_scores,
            n_aug_sample_points,
            weight_by_scores,
            normalize_features,
            phi_scale,
        )
        all_accs.append(acc)
        all_idxs.append(idxs)
    return all_accs, all_idxs

def precomputed_aug_experiment_dpp(
        clf,
        auged_featurized_x_train,
        auged_featurized_y_train,
        auged_featurized_x_train_to_source_idxs,
        auged_featurized_x_test,
        auged_featurized_y_test,
        auged_featurized_x_test_to_source_idxs,
        train_idxs_scores,
        n_aug_sample_points,
        weight_by_scores,
        normalize_features,
        phi_scale,
        weight_f="dot_prod",
):
    """
    DPP version
    normalize_features If true, scales features to [0,1]
    phi_scale The scale to apply to the similarity matrix's phis
    weight_f The type of weighting function to use in the DPP similarity matrix
    """
    influence_acc = []
    aug_iter_idxs = []
    original_mask_train = auged_featurized_x_train_to_source_idxs < 0
    original_x_train = auged_featurized_x_train[original_mask_train]
    original_y_train = auged_featurized_y_train[original_mask_train]

    # DPP part
    if normalize_features:
        phi = sklearn.preprocessing.normalize(original_x_train,
                                              norm="l2",
                                              axis=1)
    else:
        phi = original_x_train
    phi *= phi_scale

    if weight_by_scores:
        # Diversity and quality
        if weight_f == "dot_prod":
            weighted_features = phi * train_idxs_scores[:, np.newaxis]
            L = train_idxs_scores[:, np.newaxis] * phi.dot(phi.T) * train_idxs_scores[:, np.newaxis].T
            # We add some diagonal component to ensure PSD matrix
            L += np.diag(np.full(len(L), 1e-3))
        elif weight_f == "gauss_dist":
            L = sklearn.metrics.pairwise.euclidean_distances(
                phi,
                phi,
            )
            L = np.exp(-L)
            L += np.diag(np.abs(train_idxs_scores))
        else:
            raise ValueError("Unknown weight: {}".format(weight_f))
    else:
        # Just diversity
        if weight_f == "dot_prod":
            L = phi.dot(phi.T)
            # We add some diagonal component to ensure PSD matrix
            L += np.diag(np.full(len(L), 1e-3))
        elif weight_f == "gauss_dist":
            L = sklearn.metrics.pairwise.euclidean_distances(
                phi,
                phi,
            )
            L = np.exp(-L)
    print("L: {}".format(L))
    print("L shape: {}".format(L.shape))
    assert len(L) == len(original_x_train)

    for k in n_aug_sample_points:
        dpp_idxs = sample_dpp.oct_sample_k_dpp(
            L,
            k=k,
            one_hot=False)

        print("DPP idxs: {}".format(dpp_idxs))
        orig_and_dpp_idxs = np.concatenate([dpp_idxs, [-1]])
        print("orig_and_dpp_idxs: {}".format(orig_and_dpp_idxs))
        print("orig_and_dpp_idxs shape: {}".format(orig_and_dpp_idxs.shape))
        dpp_orig_and_auged_mask = np.isin(
            auged_featurized_x_train_to_source_idxs,
            orig_and_dpp_idxs)
        print("dpp_orig_and_auged_mask count: {}/{}".format(
            np.sum(dpp_orig_and_auged_mask),
            len(dpp_orig_and_auged_mask),
        ))
        dpp_x_train = auged_featurized_x_train[dpp_orig_and_auged_mask]
        dpp_y_train = auged_featurized_y_train[dpp_orig_and_auged_mask]
        clf.fit(dpp_x_train, dpp_y_train)

        aug_train_poisoned_acc = clf.score(
            auged_featurized_x_test,
            auged_featurized_y_test)
        influence_acc.append(aug_train_poisoned_acc)
        aug_iter_idxs.append(dpp_idxs) # Batch append

    return influence_acc, aug_iter_idxs
