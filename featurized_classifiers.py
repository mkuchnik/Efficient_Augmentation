import sklearn.pipeline

import preprocessing_transformers
import libinfluence.linear_model.LogisticRegression
import libinfluence.svm.LinearSVM

def build_featurized_LeNet_logistic_clf(
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        logistic_reg__C,
        cv,
        n_jobs):
    import deep_transformers.keras_dnn_feature_extractor
    # This extracts features
    deep_feature_map = (deep_transformers
                        .keras_dnn_feature_extractor
                        .MNIST_CNN_Extractor(max_iter=CNN_extractor_max_iter,
                                             use_GPU=use_GPU,
                                             batch_size=batch_size,
                                             verbose=1))
    if cv < 2:
        assert len(logistic_reg__C) == 1, "Must specific logistic params"
        clf = libinfluence.linear_model.LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=10000,
                C=logistic_reg__C[0],
                warm_start=False,
        )
    else:
        clf = libinfluence.linear_model.LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=10000,
                warm_start=False,
        )
    clf = sklearn.pipeline.Pipeline([
        ("image_rescaler", (preprocessing_transformers
                            .image_rescaler_transformer())),
        ("feature_map", deep_feature_map),
        ("logistic_reg", clf)])

    # Grid Search
    hyperparameters = [
            {
                 "logistic_reg__C": logistic_reg__C,
            },
    ]
    if cv >= 2:
        clf = sklearn.model_selection.GridSearchCV(
                clf,
                hyperparameters,
                cv=cv,
                n_jobs=n_jobs,
                iid=True,
                verbose=1,
                refit=True,
                )
    else:
        print("Not using grid search")
    return clf

def build_featurized_LeNet_SVM_clf(
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        svm__C,
        cv,
        n_jobs):
    import deep_transformers.keras_dnn_feature_extractor
    # This extracts features
    deep_feature_map = (deep_transformers
                        .keras_dnn_feature_extractor
                        .MNIST_CNN_Extractor(max_iter=CNN_extractor_max_iter,
                                             use_GPU=use_GPU,
                                             batch_size=batch_size,
                                             verbose=1))
    if cv < 2:
        assert len(svm__C) == 1, "Must specific SVM params"
        clf = libinfluence.svm.LinearSVM(
            C=svm__C[0],
        )
    else:
        clf = libinfluence.svm.LinearSVM()
    clf = sklearn.pipeline.Pipeline([
        ("image_rescaler", (preprocessing_transformers
                            .image_rescaler_transformer())),
        ("feature_map", deep_feature_map),
        ("svm", clf)])

    # Grid Search
    hyperparameters = [
            {
                 "svm__C": svm__C,
            },
    ]
    if cv >= 2:
        clf = sklearn.model_selection.GridSearchCV(
                clf,
                hyperparameters,
                cv=cv,
                n_jobs=n_jobs,
                iid=True,
                verbose=1,
                refit=True,
                )
    else:
        print("Not using grid search")
    return clf


def build_SVM_clf(
        svm__C,
        cv,
        n_jobs):
    if cv < 2:
        assert len(svm__C) == 1, "Must specific SVM params"
        clf = libinfluence.svm.LinearSVM(
            C=svm__C[0],
        )
    else:
        clf = libinfluence.svm.LinearSVM()
    clf = sklearn.pipeline.Pipeline([("svm", clf)])

    # Grid Search
    hyperparameters = [
            {
                 "svm__C": svm__C,
            },
    ]
    if cv >= 2:
        clf = sklearn.model_selection.GridSearchCV(
                clf,
                hyperparameters,
                cv=cv,
                n_jobs=n_jobs,
                iid=True,
                verbose=1,
                refit=True,
                )
    else:
        print("Not using grid search")
    return clf

def build_featurized_ResNet_logistic_clf(
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        logistic_reg__C,
        cv,
        n_jobs):
    import deep_transformers.keras_dnn_pretrained_feature_extractor
    # This extracts features
    deep_feature_map = (deep_transformers
                        .keras_dnn_pretrained_feature_extractor
                        .MNIST_CNN_Extractor(max_iter=CNN_extractor_max_iter,
                                             use_GPU=use_GPU,
                                             batch_size=batch_size,
                                             verbose=1))
    if cv < 2:
        assert len(logistic_reg__C) == 1, "Must specific logistic params"
        clf = libinfluence.linear_model.LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=10000,
                C=logistic_reg__C[0],
                warm_start=False,
        )
    else:
        clf = libinfluence.linear_model.LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=10000,
                warm_start=False,
        )
    clf = sklearn.pipeline.Pipeline([
        ("image_rescaler", (preprocessing_transformers
                            .image_rescaler_transformer())),
        ("feature_map", deep_feature_map),
        ("logistic_reg", clf)])

    # Grid Search
    hyperparameters = [
            {
                 "logistic_reg__C": logistic_reg__C,
            },
    ]
    if cv >= 2:
        clf = sklearn.model_selection.GridSearchCV(
                clf,
                hyperparameters,
                cv=cv,
                n_jobs=n_jobs,
                iid=True,
                verbose=1,
                refit=True,
                )
    else:
        print("Not using grid search")
    return clf

def build_featurized_ResNet_feature_clf(
        CNN_extractor_max_iter,
        use_GPU,
        batch_size,
        model_filename
):
    import deep_transformers.keras_dnn_pretrained_feature_extractor_norb
    deep_feature_map = (deep_transformers
                        .keras_dnn_pretrained_feature_extractor_norb
                        .MNIST_CNN_Extractor(max_iter=CNN_extractor_max_iter,
                                             use_GPU=use_GPU,
                                             batch_size=batch_size,
                                             verbose=1,
                                             model_filename=model_filename,
                                             ))
    feature_clf = sklearn.pipeline.Pipeline([
        ("image_rescaler", (preprocessing_transformers
                            .image_rescaler_transformer())),
        ("feature_map", deep_feature_map)])
    return feature_clf

def build_logistic_reg_clf(logistic_reg__C, cv):
    if cv < 2:
        assert len(logistic_reg__C) == 1, "Must specific logistic params"
        clf = libinfluence.linear_model.LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=10000,
                C=logistic_reg__C[0],
                warm_start=False,
        )
    else:
        clf = libinfluence.linear_model.LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=10000,
                warm_start=False,
        )
    clf = sklearn.pipeline.Pipeline([("logistic_reg", clf)])
    return clf
