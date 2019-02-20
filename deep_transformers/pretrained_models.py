import keras_dnn_feature_extractor.MNIST_CNN_Extractor
import sklearn.base
import sklearn.utils.multiclass
import sklearn.utils.validation
import numpy as np

cached_sess = None
cached_model = None


class Pretrained_Feature_Extractor(
        keras_dnn_feature_extractor.MNIST_CNN_Extractor):
    def __init__(self, filename, optimizer="adam", batch_size=32,
                 max_iter=20, shuffle=True, verbose=0,
                 class_weight=None, use_GPU=False):
        self.filename = filename
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.verbose = verbose
        self.class_weight = class_weight
        self.use_GPU = use_GPU

    def get_params(self, deep=True):
        return {
                "filename": self.filename,
                "optimizer": self.optimizer,
                "batch_size": self.batch_size,
                "max_iter": self.max_iter,
                "shuffle": self.shuffle,
                "verbose": self.verbose,
                "class_weight": self.class_weight,
                "use_GPU": self.use_GPU,
        }

    def fit(self, X, y, sample_weight=None):
        # Local import for multiprocessing
        import keras
        import tensorflow as tf
        # TODO can use LabelEncoder
        sklearn.utils.multiclass.check_classification_targets(y)
        X = np.array(X)
        y = np.array(y)
        self.X_ = X
        self.y_ = y
        self.classes_ = sklearn.utils.multiclass.unique_labels(y)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError("Cannot train classifier with less"
                             " than 2 classes.")
        self._create_and_set_TF_sess_and_model(self.filename)
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
        if self.verbose:
            print(self.model_.summary())
        if self.max_iter > 0:
            self._fit(sample_weight=sample_weight)
        self.fitted_ = True
        return self

    def _create_and_set_TF_sess_and_model(self, filename=None, **kwargs):
        global cached_model
        global cached_sess
        if not cached_model:
            self._create_and_set_TF_sess()
            self._create_and_set_TF_model(filename=filename, **kwargs)
            cached_model = self.model_
            cached_sess = self.sess_
        else:
            self.model_ = cached_model
            self.sess_ = cached_sess
        return self

    def transform(self, X):
        import keras
        import keras.backend
        import tensorflow as tf
        K = keras.backend
        with self.sess_.as_default():
            outs = []
            features = K.function([self.model_.input],
                                  [self.model_.layers[-2].output])
            if self.batch_size <= len(X):
                for i in range(0, len(X) - self.batch_size + 1, self.batch_size):
                    x = X[i:i+self.batch_size]
                    f = features([x])
                    assert len(f) == 1
                    f = f[0]
                    outs.append(f.reshape(len(f), -1))
                i = i + self.batch_size
                if i != len(X):
                    x = X[i:]
                    f = features([x])
                    assert len(f) == 1
                    f = f[0]
                    outs.append(f.reshape(len(f), -1))
            else:
                x = X[:]
                f = features([x])
                assert len(f) == 1
                f = f[0]
                outs.append(f.reshape(len(f), -1))
            outs = np.concatenate(outs, axis=0)
            return outs

    def __del__(self):
        pass
