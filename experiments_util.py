import numpy as np
import dataset_loaders
import time
import featurized_classifiers

import sklearn.model_selection


def classes_to_class_str(classes):
    """ Converts a 2-tuple to classes[0]_vs_classes[1] """
    if len(classes) == 2:
        if (hasattr(classes[0], "__len__")
                or hasattr(classes[1], "__len__")):
            raise ValueError("Unsupported datatype")
        else:
            return "{}_vs_{}".format(classes[0], classes[1])
    else:
        raise ValueError("Unsupported datatype")


def prepare_dataset(dataset, classes, n_train):
    """ Loads dataset, filters classes, and subsamples """
    (x_train, y_train), (x_test, y_test) = dataset_loaders.get_dataset(dataset)
    x_train, y_train = dataset_loaders.select_subset_classes(
        classes,
        x_train,
        y_train,
    )
    x_test, y_test = dataset_loaders.select_subset_classes(
        classes,
        x_test,
        y_test
    )
    # x_train, y_train = x_train[:n_train], y_train[:n_train]
    assert n_train % 2 == 0, "Not even split"
    n_samples_per_class = n_train // 2
    x_train, y_train = dataset_loaders.select_dataset_samples(
        x_train,
        y_train,
        n_samples_per_class)
    assert len(np.unique(y_train)) > 1
    return (x_train, y_train), (x_test, y_test)


def poison_dataset(x, y, aug_f, aug_kw_args):
    """ Applies augmentations to dataset and concatenates """
    auged_idxs, (auged_x, auged_y) = aug_f(x, y, **aug_kw_args)
    orig_and_auged_x = np.concatenate(
        [x,
         auged_x,
         ],
        axis=0,
    )
    orig_and_auged_y = np.concatenate(
        [y,
         auged_y,
         ],
        axis=0,
    )
    orig_and_auged_idxs = np.concatenate(
        [np.full(len(x), -1),
         auged_idxs,
         ],
        axis=0,
    )
    return orig_and_auged_x, orig_and_auged_y, orig_and_auged_idxs


def get_aug_scores(clf, cv, use_loss):
    if cv >= 2:
        if use_loss:
            aug_scores = (clf
                          .best_estimator_
                          .named_steps["logistic_reg"]
                          .log_losses(L2_alpha=0.0))
        else:
            aug_scores = (clf
                          .best_estimator_
                          .named_steps["logistic_reg"]
                          .LOO_influence())
    else:
        if use_loss:
            aug_scores = (clf
                          .named_steps["logistic_reg"]
                          .log_losses(L2_alpha=0.0))
        else:
            aug_scores = (clf
                          .named_steps["logistic_reg"]
                          .LOO_influence())
    return aug_scores


def train_and_score_clf(clf,
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
                        ):
    # Train
    training_start_time = time.time()
    clf.fit(x_train, y_train)

    # Unpack estimator
    if cv >= 2:
        best_params = (clf
                       .best_estimator_
                       .named_steps["logistic_reg"]
                       .get_params())
    else:
        best_params = clf.named_steps["logistic_reg"].get_params()
    print("best_params: {}".format(best_params))

    # Test
    no_aug_no_poison_acc = clf.score(x_test, y_test)
    print("baseline_acc: {}".format(no_aug_no_poison_acc))

    # Get augmentation scores
    aug_scores = get_aug_scores(clf, cv, use_loss)
    print("aug_scores: {}".format(aug_scores))
    print("aug_scores mean: {}".format(np.mean(aug_scores)))
    print("aug_scores std: {}".format(np.std(aug_scores)))

    training_end_time = time.time()
    training_total_time = training_end_time - training_start_time
    print("*" * 80)
    print("Training took {} seconds".format(training_total_time))
    print("*" * 80)

    # Poison test
    poisoned_acc = clf.score(orig_and_auged_x_test, orig_and_auged_y_test)
    print("poisoned_acc: {}".format(poisoned_acc))

    clf.fit(orig_and_auged_x_train, orig_and_auged_y_train)

    # Get augmentation scores
    after_aug_scores = get_aug_scores(clf, cv, use_loss)

    all_aug_train_poisoned_acc = clf.score(orig_and_auged_x_test,
                                           orig_and_auged_y_test)
    print("all_aug_train_poisoned_acc: {}".format(all_aug_train_poisoned_acc))

    if all_aug_train_poisoned_acc < poisoned_acc:
        print("***WARNING: Augmentation lowered accuracy***")

    return (no_aug_no_poison_acc,
            poisoned_acc,
            all_aug_train_poisoned_acc,
            aug_scores,
            after_aug_scores,
            best_params,
            training_total_time)


def get_SV_featurized_LeNet(
        x_train,
        y_train,
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        svm__C,
        cv,
        n_jobs):
    svm_clf = featurized_classifiers.build_featurized_LeNet_SVM_clf(
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        svm__C,
        cv,
        n_jobs)
    svm_clf.fit(x_train, y_train)
    if isinstance(svm_clf, sklearn.model_selection.GridSearchCV):
        is_SV = svm_clf.best_estimator_.named_steps["svm"].is_support_vector()
    else:
        is_SV = svm_clf.named_steps["svm"].is_support_vector()
    return is_SV


def get_SVM_losses_featurized_LeNet(
        x_train,
        y_train,
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        svm__C,
        cv,
        n_jobs):
    svm_clf = featurized_classifiers.build_featurized_LeNet_SVM_clf(
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        svm__C,
        cv,
        n_jobs)
    svm_clf.fit(x_train, y_train)
    if isinstance(svm_clf, sklearn.model_selection.GridSearchCV):
        losses = svm_clf.best_estimator_.named_steps["svm"].pred_losses()
    else:
        losses = svm_clf.named_steps["svm"].pred_losses()
    return losses


def get_SV_raw(
        x_train,
        y_train,
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        svm__C,
        cv,
        n_jobs):
    svm_clf = featurized_classifiers.build_SVM_clf(
        svm__C,
        cv,
        n_jobs)
    svm_clf.fit(x_train, y_train)
    if isinstance(svm_clf, sklearn.model_selection.GridSearchCV):
        is_SV = svm_clf.best_estimator_.named_steps["svm"].is_support_vector()
    else:
        is_SV = svm_clf.named_steps["svm"].is_support_vector()
    return is_SV


def get_SVM_losses_raw(
        x_train,
        y_train,
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        svm__C,
        cv,
        n_jobs):
    svm_clf = featurized_classifiers.build_SVM_clf(
        svm__C,
        cv,
        n_jobs)
    svm_clf.fit(x_train, y_train)
    if isinstance(svm_clf, sklearn.model_selection.GridSearchCV):
        losses = svm_clf.best_estimator_.named_steps["svm"].pred_losses()
    else:
        losses = svm_clf.named_steps["svm"].pred_losses()
    return losses


def get_SVM_margins_raw(
        x_train,
        y_train,
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        svm__C,
        cv,
        n_jobs):
    svm_clf = featurized_classifiers.build_SVM_clf(
        svm__C,
        cv,
        n_jobs)
    svm_clf.fit(x_train, y_train)
    if isinstance(svm_clf, sklearn.model_selection.GridSearchCV):
        margins = (svm_clf
                   .best_estimator_
                   .named_steps["svm"]
                   .decision_function(x_train))
    else:
        margins = svm_clf.named_steps["svm"].decision_function(x_train)
    return margins


def get_SVM_margins_featurized_LeNet(
        x_train,
        y_train,
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        svm__C,
        cv,
        n_jobs):
    svm_clf = featurized_classifiers.build_featurized_LeNet_SVM_clf(
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        svm__C,
        cv,
        n_jobs)
    svm_clf.fit(x_train, y_train)
    if isinstance(svm_clf, sklearn.model_selection.GridSearchCV):
        featurizer = sklearn.pipeline.Pipeline([
            ("image_rescaler", (svm_clf
                                .best_estimator_
                                .named_steps["image_rescaler"])),
            ("feature_map", (svm_clf
                             .best_estimator_
                             .named_steps["feature_map"])),
        ])
        featurized_x_train = featurizer.transform(x_train)
        margins = (svm_clf
                   .best_estimator_
                   .named_steps["svm"]
                   .decision_function(featurized_x_train))
    else:
        featurizer = sklearn.pipeline.Pipeline([
            ("image_rescaler", (svm_clf.named_steps["image_rescaler"])),
            ("feature_map", svm_clf.named_steps["feature_map"]),
        ])
        featurized_x_train = featurizer.transform(x_train)
        margins = svm_clf.named_steps["svm"].decision_function(
            featurized_x_train
        )
    return margins
