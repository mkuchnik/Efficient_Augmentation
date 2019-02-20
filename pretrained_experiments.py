import pprint
import time

import numpy as np
import joblib

import selection_policy
import augmentations
import experiments
import experiments_util
import featurized_classifiers
import sklearn.cluster
import sklearn.metrics
import collections

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
             model_filename,
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
        "model_filename": model_filename,
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
    print("x_train", x_train.shape)
    print("orig_and_auged_x_train", orig_and_auged_x_train.shape)

    feature_clf = featurized_classifiers.build_featurized_ResNet_feature_clf(
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        model_filename,
    )

    @mem.cache
    def transform_features(x, y, model_filename):
        # We need model filename to invalidate cache on model change
        return feature_clf.fit_transform(x, y=y)

    featurized_x_train = transform_features(x=x_train,
                                            y=y_train,
                                            model_filename=model_filename)
    featurized_y_train = y_train
    featurized_x_test = transform_features(x=x_test,
                                           y=y_test,
                                           model_filename=model_filename)
    # featurized_y_test = y_test
    orig_and_auged_featurized_x_train = transform_features(
        x=orig_and_auged_x_train,
        y=orig_and_auged_y_train,
        model_filename=model_filename,
    )
    orig_and_auged_featurized_y_train = orig_and_auged_y_train
    orig_and_auged_featurized_x_train_to_source_idxs = \
        orig_and_auged_idxs_train
    orig_and_auged_featurized_x_test = transform_features(
        x=orig_and_auged_x_test,
        y=orig_and_auged_y_test,
        model_filename=model_filename,
    )
    orig_and_auged_featurized_y_test = orig_and_auged_y_test
    orig_and_auged_featurized_x_test_to_source_idxs = orig_and_auged_idxs_test

    clf = featurized_classifiers.build_logistic_reg_clf(
        logistic_reg__C,
        cv,
    )

    svm__C = [0.01, 0.1, 1, 10, 100]
    svm_cv = 4
    is_SV = experiments_util.get_SV_raw(
        featurized_x_train,
        featurized_y_train,
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        svm__C,
        svm_cv,
        n_jobs
    )
    SVM_losses = experiments_util.get_SVM_losses_raw(
        featurized_x_train,
        featurized_y_train,
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
    print("orig_and_SV_idxs", orig_and_SV_idxs)
    print("orig_and_SV_idxs", orig_and_SV_idxs.shape)
    SV_orig_and_auged_mask = np.isin(orig_and_auged_idxs_train,
                                     orig_and_SV_idxs)
    print("SV_orig_and_auged_mask count {}/{}".format(
        np.sum(SV_orig_and_auged_mask),
        len(SV_orig_and_auged_mask),
    ))
    SV_x_train = orig_and_auged_featurized_x_train[SV_orig_and_auged_mask]
    SV_y_train = orig_and_auged_featurized_y_train[SV_orig_and_auged_mask]
    clf.fit(SV_x_train, SV_y_train)
    VSV_acc = clf.score(orig_and_auged_featurized_x_test,
                        orig_and_auged_featurized_y_test)
    print("VSV acc: {}".format(VSV_acc))

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

    np_data_dict = {
        "x_train": orig_and_auged_x_train,
        "y_train": orig_and_auged_y_train,
        "train_to_source_idxs":
            orig_and_auged_idxs_train,
        "featurized_x_train": orig_and_auged_featurized_x_train,
        "featurized_y_train": orig_and_auged_featurized_y_train,
        "x_test": orig_and_auged_x_test,
        "y_test": orig_and_auged_y_test,
        "test_to_source_idxs":
            orig_and_auged_idxs_test,
        "featurized_x_test": orig_and_auged_featurized_x_test,
        "featurized_y_test": orig_and_auged_featurized_y_test,
        "SV_x_train": orig_and_auged_x_train[SV_orig_and_auged_mask],
        "SV_y_train": orig_and_auged_y_train[SV_orig_and_auged_mask],
        "featurized_SV_x_train": SV_x_train,
        "featurized_SV_y_train": SV_y_train,
        "SVM_losses": SVM_losses,
        "aug_scores": aug_scores,
        "after_aug_scores": after_aug_scores,
    }

    np_data_filename = results_filename + "_data.npz"
    np.savez(np_data_filename,
             **np_data_dict)

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

    if baseline_test:
        test_orig_and_auged_mask = np.isin(orig_and_auged_idxs_test, [-1])
        auged_featurized_x_test = orig_and_auged_featurized_x_test[
            test_orig_and_auged_mask
        ]
        auged_featurized_y_test = orig_and_auged_featurized_y_test[
            test_orig_and_auged_mask
        ]
    else:
        auged_featurized_x_test = orig_and_auged_featurized_x_test
        auged_featurized_y_test = orig_and_auged_featurized_y_test

    experiment_results = {}
    for policy_name, update_score, downweight_points in experiment_configs:
        policy_f = selection_policy.get_policy_by_name(policy_name)
        if "deterministic" in policy_name:
            _rounds = 1
        else:
            _rounds = rounds
        acc, idxs = experiments.precomputed_aug_experiment_rounds(
            clf=clf,
            auged_featurized_x_train=orig_and_auged_featurized_x_train,
            auged_featurized_y_train=orig_and_auged_featurized_y_train,
            auged_featurized_x_train_to_source_idxs=orig_and_auged_featurized_x_train_to_source_idxs,
            auged_featurized_x_test=auged_featurized_x_test,
            auged_featurized_y_test=auged_featurized_y_test,
            auged_featurized_x_test_to_source_idxs=orig_and_auged_featurized_x_test_to_source_idxs,
            aug_iter=policy_f,
            train_idxs_scores=aug_scores,
            n_aug_sample_points=n_aug_sample_points,
            rounds=_rounds,
            update_scores=update_score,
            weight_aug_samples=downweight_points,
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
        model_filename,
        n_clusters,
        cluster_type="kmeans",
        #cluster_type="kmedoids",
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
        "model_filename": model_filename,
        "n_clusters": n_clusters,
        "cluster_type": cluster_type,
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
        batch_size,
        model_filename,
    )

    @mem.cache
    def transform_features(x, y, model_filename):
        # We need model filename to invalidate cache on model change
        return feature_clf.fit_transform(x, y=y)

    featurized_x_train = transform_features(
        x=x_train,
        y=y_train,
        model_filename=model_filename,
    )
    featurized_y_train = y_train
    featurized_x_test = transform_features(
        x=x_test,
        y=y_test,
        model_filename=model_filename,
    )
    featurized_y_test = y_test
    orig_and_auged_featurized_x_train = transform_features(
        x=orig_and_auged_x_train,
        y=orig_and_auged_y_train,
        model_filename=model_filename,
    )
    orig_and_auged_featurized_y_train = orig_and_auged_y_train
    orig_and_auged_featurized_x_train_to_source_idxs = \
        orig_and_auged_idxs_train
    orig_and_auged_featurized_x_test = transform_features(
        x=orig_and_auged_x_test,
        y=orig_and_auged_y_test,
        model_filename=model_filename,
    )
    orig_and_auged_featurized_y_test = orig_and_auged_y_test
    orig_and_auged_featurized_x_test_to_source_idxs = orig_and_auged_idxs_test

    if cluster_type == "kmeans":
        clustering_clf = sklearn.cluster.KMeans(n_clusters=n_clusters)
        train_cluster_IDs = clustering_clf.fit_predict(featurized_x_train)
        test_cluster_IDs = clustering_clf.predict(featurized_x_test)
    elif cluster_type == "kmedoids":
        from pyclustering.cluster.kmedoids import kmedoids
        from pyclustering.utils import timedcall
        import scipy.spatial
        # Using some code from kmedoids_examples.py from pyclustering
        clustering_clf = sklearn.cluster.KMeans(n_clusters=n_clusters)
        init_medoids = clustering_clf.fit_predict(featurized_x_train)
        #init_medoids = np.random.choice(len(featurized_x_train),
        #                                n_clusters,
        #                                replace=False)
        tolerance = 0.25
        kmedoids_instance = kmedoids(featurized_x_train,
                                     init_medoids,
                                     tolerance)
        (ticks, result) = timedcall(kmedoids_instance.process)  # Run
        cluster_IDs = kmedoids_instance.get_medoids()  # index into training set
        clusters = featurized_x_train[cluster_IDs]
        tree = scipy.spatial.cKDTree(clusters)
        _, train_cluster_IDs = tree.query(featurized_x_train, 1)
        _, test_cluster_IDs = tree.query(featurized_x_test, 1)
    print("Train cluster IDs: {}".format(train_cluster_IDs))
    print("Test cluster IDs: {}".format(test_cluster_IDs))

    clf = featurized_classifiers.build_logistic_reg_clf(
        logistic_reg__C,
        cv,
    )

    svm__C = [0.01, 0.1, 1, 10, 100]
    svm_cv = 4
    is_SV = experiments_util.get_SV_raw(
        featurized_x_train,
        featurized_y_train,
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        svm__C,
        svm_cv,
        n_jobs
    )
    SVM_losses = experiments_util.get_SVM_losses_raw(
        featurized_x_train,
        featurized_y_train,
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
    print("orig_and_SV_idxs", orig_and_SV_idxs)
    print("orig_and_SV_idxs", orig_and_SV_idxs.shape)
    SV_orig_and_auged_mask = np.isin(orig_and_auged_idxs_train,
                                     orig_and_SV_idxs)
    print("SV_orig_and_auged_mask count {}/{}".format(
        np.sum(SV_orig_and_auged_mask),
        len(SV_orig_and_auged_mask),
    ))
    SV_x_train = orig_and_auged_featurized_x_train[SV_orig_and_auged_mask]
    SV_y_train = orig_and_auged_featurized_y_train[SV_orig_and_auged_mask]
    clf.fit(SV_x_train, SV_y_train)
    VSV_acc = clf.score(orig_and_auged_featurized_x_test,
                        orig_and_auged_featurized_y_test)
    print("VSV acc: {}".format(VSV_acc))

    np_data_dict = {
        "x_train": orig_and_auged_x_train,
        "y_train": orig_and_auged_y_train,
        "train_to_source_idxs":
            orig_and_auged_idxs_train,
        "featurized_x_train": orig_and_auged_featurized_x_train,
        "featurized_y_train": orig_and_auged_featurized_y_train,
        "x_test": orig_and_auged_x_test,
        "y_test": orig_and_auged_y_test,
        "test_to_source_idxs":
            orig_and_auged_idxs_test,
        "featurized_x_test": orig_and_auged_featurized_x_test,
        "featurized_y_test": orig_and_auged_featurized_y_test,
        "SV_x_train": orig_and_auged_x_train[SV_orig_and_auged_mask],
        "SV_y_train": orig_and_auged_y_train[SV_orig_and_auged_mask],
        "featurized_SV_x_train": SV_x_train,
        "featurized_SV_y_train": SV_y_train,
    }

    np_data_filename = results_filename + "_data.npz"
    np.savez(np_data_filename,
             **np_data_dict)

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

    experiment_results = {}
    for policy_name, update_score, downweight_points in experiment_configs:
        policy_f = selection_policy.get_policy_by_name(policy_name)
        if "deterministic" in policy_name:
            _rounds = 1
        else:
            _rounds = rounds
        acc, idxs = experiments.precomputed_aug_experiment_rounds(
            clf=clf,
            auged_featurized_x_train=orig_and_auged_featurized_x_train,
            auged_featurized_y_train=orig_and_auged_featurized_y_train,
            auged_featurized_x_train_to_source_idxs=orig_and_auged_featurized_x_train_to_source_idxs,
            auged_featurized_x_test=orig_and_auged_featurized_x_test,
            auged_featurized_y_test=orig_and_auged_featurized_y_test,
            auged_featurized_x_test_to_source_idxs=orig_and_auged_featurized_x_test_to_source_idxs,
            aug_iter=policy_f,
            train_idxs_scores=aug_scores,
            n_aug_sample_points=n_aug_sample_points,
            rounds=_rounds,
            update_scores=update_score,
            weight_aug_samples=downweight_points,
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
        model_filename,
):
    """
    Runs a sweep over the full range of the number of clusters (from 2 to max)
    and calculates various statistics
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
        "model_filename": model_filename,
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
        batch_size,
        model_filename,
    )

    @mem.cache
    def transform_features(x, y, model_filename):
        # We need model filename to invalidate cache on model change
        return feature_clf.fit_transform(x, y=y)

    featurized_x_train = transform_features(
        x=x_train,
        y=y_train,
        model_filename=model_filename,
    )
    featurized_x_test = transform_features(
        x=x_test,
        y=y_test,
        model_filename=model_filename,
    )

    all_results = collections.defaultdict(list)
    n_clusters_arr = np.unique(
        np.clip(
            np.around(
                np.geomspace(2, len(featurized_x_train) - 1, num=50)
            ).astype(int),
            2,
            len(featurized_x_train) - 1
        ),
    )
    print("n_clusters_arr", n_clusters_arr)

    for n_clusters in n_clusters_arr:
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
        model_filename,
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
        "model_filename": model_filename,
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
        batch_size,
        model_filename,
    )

    @mem.cache
    def transform_features(x, y, model_filename):
        # We need model filename to invalidate cache on model change
        return feature_clf.fit_transform(x, y=y)

    featurized_x_train = transform_features(
        x=x_train,
        y=y_train,
        model_filename=model_filename,
    )
    featurized_y_train = y_train
    featurized_x_test = transform_features(
        x=x_test,
        y=y_test,
        model_filename=model_filename,
    )
    # featurized_y_test = y_test
    orig_and_auged_featurized_x_train = transform_features(
        x=orig_and_auged_x_train,
        y=orig_and_auged_y_train,
        model_filename=model_filename,
    )
    orig_and_auged_featurized_y_train = orig_and_auged_y_train
    orig_and_auged_featurized_x_train_to_source_idxs = \
        orig_and_auged_idxs_train
    orig_and_auged_featurized_x_test = transform_features(
        x=orig_and_auged_x_test,
        y=orig_and_auged_y_test,
        model_filename=model_filename,
    )
    orig_and_auged_featurized_y_test = orig_and_auged_y_test
    orig_and_auged_featurized_x_test_to_source_idxs = orig_and_auged_idxs_test

    clf = featurized_classifiers.build_logistic_reg_clf(
        logistic_reg__C,
        cv,
    )

    svm__C = [0.01, 0.1, 1, 10, 100]
    svm_cv = 4
    is_SV = experiments_util.get_SV_raw(
        featurized_x_train,
        featurized_y_train,
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        svm__C,
        svm_cv,
        n_jobs
    )
    SVM_losses = experiments_util.get_SVM_losses_raw(
        featurized_x_train,
        featurized_y_train,
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        svm__C,
        svm_cv,
        n_jobs
    )
    SVM_margins = experiments_util.get_SVM_margins_raw(
        featurized_x_train,
        featurized_y_train,
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
    print("orig_and_SV_idxs", orig_and_SV_idxs)
    print("orig_and_SV_idxs", orig_and_SV_idxs.shape)
    SV_orig_and_auged_mask = np.isin(orig_and_auged_idxs_train,
                                     orig_and_SV_idxs)
    print("SV_orig_and_auged_mask count {}/{}".format(
        np.sum(SV_orig_and_auged_mask),
        len(SV_orig_and_auged_mask),
    ))
    SV_x_train = orig_and_auged_featurized_x_train[SV_orig_and_auged_mask]
    SV_y_train = orig_and_auged_featurized_y_train[SV_orig_and_auged_mask]
    clf.fit(SV_x_train, SV_y_train)
    VSV_acc = clf.score(orig_and_auged_featurized_x_test,
                        orig_and_auged_featurized_y_test)
    print("VSV acc: {}".format(VSV_acc))

    np_data_dict = {
        "x_train": orig_and_auged_x_train,
        "y_train": orig_and_auged_y_train,
        "train_to_source_idxs":
            orig_and_auged_idxs_train,
        "featurized_x_train": orig_and_auged_featurized_x_train,
        "featurized_y_train": orig_and_auged_featurized_y_train,
        "x_test": orig_and_auged_x_test,
        "y_test": orig_and_auged_y_test,
        "test_to_source_idxs":
            orig_and_auged_idxs_test,
        "featurized_x_test": orig_and_auged_featurized_x_test,
        "featurized_y_test": orig_and_auged_featurized_y_test,
        "SV_x_train": orig_and_auged_x_train[SV_orig_and_auged_mask],
        "SV_y_train": orig_and_auged_y_train[SV_orig_and_auged_mask],
        "featurized_SV_x_train": SV_x_train,
        "featurized_SV_y_train": SV_y_train,
        "SVM_losses": SVM_losses,
    }

    np_data_filename = results_filename + "_data.npz"
    np.savez(np_data_filename,
             **np_data_dict)

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

    # Here we use margins
    aug_scores = np.abs(SVM_margins)
    experiment_results = {}
    for policy_name, update_score, downweight_points in experiment_configs:
        policy_f = selection_policy.get_policy_by_name(policy_name)
        if "deterministic" in policy_name:
            _rounds = 1
        else:
            _rounds = rounds
        acc, idxs = experiments.precomputed_aug_experiment_rounds(
            clf=clf,
            auged_featurized_x_train=orig_and_auged_featurized_x_train,
            auged_featurized_y_train=orig_and_auged_featurized_y_train,
            auged_featurized_x_train_to_source_idxs=orig_and_auged_featurized_x_train_to_source_idxs,
            auged_featurized_x_test=orig_and_auged_featurized_x_test,
            auged_featurized_y_test=orig_and_auged_featurized_y_test,
            auged_featurized_x_test_to_source_idxs=orig_and_auged_featurized_x_test_to_source_idxs,
            aug_iter=policy_f,
            train_idxs_scores=aug_scores,
            n_aug_sample_points=n_aug_sample_points,
            rounds=_rounds,
            update_scores=update_score,
            weight_aug_samples=downweight_points,
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


def run_test_dpp(classes,
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
        "model_filename": model_filename,
        "dpp_weight_by_scores": dpp_weight_by_scores,
        "dpp_normalize_features": dpp_normalize_features,
        "dpp_phi_scale": dpp_phi_scale,
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
        batch_size,
        model_filename,
    )

    @mem.cache
    def transform_features(x, y, model_filename):
        # We need model filename to invalidate cache on model change
        return feature_clf.fit_transform(x, y=y)

    featurized_x_train = transform_features(x=x_train,
                                            y=y_train,
                                            model_filename=model_filename)
    featurized_y_train = y_train
    featurized_x_test = transform_features(x=x_test,
                                           y=y_test,
                                           model_filename=model_filename)
    # featurized_y_test = y_test
    orig_and_auged_featurized_x_train = transform_features(
        x=orig_and_auged_x_train,
        y=orig_and_auged_y_train,
        model_filename=model_filename,
    )
    orig_and_auged_featurized_y_train = orig_and_auged_y_train
    orig_and_auged_featurized_x_train_to_source_idxs = \
        orig_and_auged_idxs_train
    orig_and_auged_featurized_x_test = transform_features(
        x=orig_and_auged_x_test,
        y=orig_and_auged_y_test,
        model_filename=model_filename,
    )
    orig_and_auged_featurized_y_test = orig_and_auged_y_test
    orig_and_auged_featurized_x_test_to_source_idxs = orig_and_auged_idxs_test

    clf = featurized_classifiers.build_logistic_reg_clf(
        logistic_reg__C,
        cv,
    )

    svm__C = [0.01, 0.1, 1, 10, 100]
    svm_cv = 4
    is_SV = experiments_util.get_SV_raw(
        featurized_x_train,
        featurized_y_train,
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        svm__C,
        svm_cv,
        n_jobs
    )
    SVM_losses = experiments_util.get_SVM_losses_raw(
        featurized_x_train,
        featurized_y_train,
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
    print("orig_and_SV_idxs", orig_and_SV_idxs)
    print("orig_and_SV_idxs", orig_and_SV_idxs.shape)
    SV_orig_and_auged_mask = np.isin(orig_and_auged_idxs_train,
                                     orig_and_SV_idxs)
    print("SV_orig_and_auged_mask count {}/{}".format(
        np.sum(SV_orig_and_auged_mask),
        len(SV_orig_and_auged_mask),
    ))
    SV_x_train = orig_and_auged_featurized_x_train[SV_orig_and_auged_mask]
    SV_y_train = orig_and_auged_featurized_y_train[SV_orig_and_auged_mask]
    clf.fit(SV_x_train, SV_y_train)
    VSV_acc = clf.score(orig_and_auged_featurized_x_test,
                        orig_and_auged_featurized_y_test)
    print("VSV acc: {}".format(VSV_acc))

    np_data_dict = {
        "x_train": orig_and_auged_x_train,
        "y_train": orig_and_auged_y_train,
        "train_to_source_idxs":
            orig_and_auged_idxs_train,
        "featurized_x_train": orig_and_auged_featurized_x_train,
        "featurized_y_train": orig_and_auged_featurized_y_train,
        "x_test": orig_and_auged_x_test,
        "y_test": orig_and_auged_y_test,
        "test_to_source_idxs":
            orig_and_auged_idxs_test,
        "featurized_x_test": orig_and_auged_featurized_x_test,
        "featurized_y_test": orig_and_auged_featurized_y_test,
        "SV_x_train": orig_and_auged_x_train[SV_orig_and_auged_mask],
        "SV_y_train": orig_and_auged_y_train[SV_orig_and_auged_mask],
        "featurized_SV_x_train": SV_x_train,
        "featurized_SV_y_train": SV_y_train,
        "SVM_losses": SVM_losses,
    }

    np_data_filename = results_filename + "_data.npz"
    np.savez(np_data_filename,
             **np_data_dict)

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

    experiment_results = {}
    experiment_results_idxs = {}
    for policy_name, update_score, downweight_points in experiment_configs:
        policy_f = selection_policy.get_policy_by_name(policy_name)
        if "deterministic" in policy_name:
            _rounds = 1
        else:
            _rounds = rounds
        acc, idxs = experiments.precomputed_aug_experiment_rounds_dpp(
            clf=clf,
            auged_featurized_x_train=orig_and_auged_featurized_x_train,
            auged_featurized_y_train=orig_and_auged_featurized_y_train,
            auged_featurized_x_train_to_source_idxs=orig_and_auged_featurized_x_train_to_source_idxs,
            auged_featurized_x_test=orig_and_auged_featurized_x_test,
            auged_featurized_y_test=orig_and_auged_featurized_y_test,
            auged_featurized_x_test_to_source_idxs=orig_and_auged_featurized_x_test_to_source_idxs,
            train_idxs_scores=aug_scores,
            n_aug_sample_points=n_aug_sample_points,
            rounds=_rounds,
            weight_by_scores=dpp_weight_by_scores,
            normalize_features=dpp_normalize_features,
            phi_scale=dpp_phi_scale,
        )

        config_name = [policy_name]
        if update_score:
            config_name.append("update")
        if downweight_points:
            config_name.append("downweight")
        config_name = "_".join(config_name)
        experiment_results[config_name] = acc
        experiment_results_idxs[config_name] = idxs

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
        "experiment_results_idxs": experiment_results_idxs,
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
