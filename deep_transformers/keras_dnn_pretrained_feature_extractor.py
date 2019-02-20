import sklearn.base
import sklearn.utils.multiclass
import sklearn.utils.validation
import tempfile
import numpy as np
import keras

session = keras.backend.get_session()
pre_model = keras.applications.NASNetMobile(
    include_top=False,
    weights="imagenet",
    input_shape=(197, 197, 3),
    pooling="avg"
)

def create_MNIST_CNN_model(input_shape=(28, 28),
                           optimizer="adam",
                           n_classes=2):
    # import keras
    # model = keras.models.Sequential()
    # normal_init = keras.initializers.RandomNormal(
    #         mean=0.0,
    #         stddev=0.05,
    #         seed=42)
    # model.add(keras.layers.Conv2D(32,
    #                               kernel_size=(3, 3),
    #                               activation="relu",
    #                               input_shape=input_shape,
    #                               kernel_initializer=normal_init,
    #                               bias_initializer="zeros",
    #                               ))
    # model.add(keras.layers.Conv2D(64,
    #                               (3, 3),
    #                               activation="relu",
    #                               kernel_initializer=normal_init,
    #                               bias_initializer="zeros",
    #                               ))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(keras.layers.Dropout(0.25))
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(128,
    #                              activation="relu",
    #                              kernel_initializer=normal_init,
    #                              bias_initializer="zeros",
    #                              ))
    # model.add(keras.layers.Dropout(0.5))
    # model.add(keras.layers.Dense(n_classes,
    #                              activation="softmax",
    #                              kernel_initializer=normal_init,
    #                              bias_initializer="zeros",
    #                              ))
    # model.compile(loss="sparse_categorical_crossentropy",
    #               optimizer=optimizer,
    #               metrics=["accuracy"],
    #               )
    # return model

    return pre_model



class MNIST_CNN_Extractor(sklearn.base.BaseEstimator,
                          sklearn.base.ClassifierMixin,
                          sklearn.base.TransformerMixin):
    def __init__(self, optimizer="adam", batch_size=32,
                 max_iter=20, shuffle=True, verbose=0,
                 class_weight=None, use_GPU=False):
        """
        A Keras-accelerated Neural-Network.
        use_GPU (bool): If GPU should be used. Warning: using GPU with
        multi-process training will likely result in out-of-memory (OOM)
        exceptions.

        Uses a CNN for MNIST from keras
        """
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.verbose = verbose
        self.class_weight = class_weight
        self.use_GPU = use_GPU

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
        self._create_and_set_TF_sess_and_model()
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
        if self.verbose:
            print(self.model_.summary())
        if self.max_iter > 0:
            pass
            # self._fit(sample_weight=sample_weight)
        self.fitted_ = True
        return self

    def predict(self, X):
        X = np.array(X)
        y_pred_proba = self.predict_proba(X)
        y_pred_classes = np.argmax(y_pred_proba, axis=-1)
        return y_pred_classes

    def predict_proba(self, X):
        X = np.array(X)
        sklearn.utils.validation.check_is_fitted(self, ["X_", "y_"])
        with self.sess_.as_default():
            pred = self.model_.predict(
                    x=X,
                    batch_size=self.batch_size,
                    verbose=self.verbose,
            )
        return pred

    def score(self, X, y, sample_weight=None):
        X = np.array(X)
        y = np.array(y)
        sklearn.utils.validation.check_is_fitted(self, ["X_", "y_"])
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
        with self.sess_.as_default():
            _, acc = self.model_.evaluate(
                    x=X,
                    y=y,
                    batch_size=self.batch_size,
                    verbose=self.verbose,
                    sample_weight=sample_weight,
            )
        return acc

    def transform(self, X):
        import keras
        import keras.backend
        import tensorflow as tf
        K = keras.backend
        sklearn.utils.validation.check_is_fitted(self, ["X_", "y_"])
        # Assumes sequential
        with self.sess_.as_default():
            outs = []
            print("Using outputs layer: {}".format(self.model_.layers[-2]))
            features = K.function([self.model_.input],
                                  # [self.model_.layers[-2].output])
                                  [self.model_.layers[-1].output])
            if self.batch_size <= len(X):
                for i in range(0, len(X) - self.batch_size + 1, self.batch_size):
                    x = X[i:i+self.batch_size]
                    f = features([x])
                    assert len(f) == 1
                    f = f[0]
                    outs.append(f.reshape(len(f), -1))
                i = i + self.batch_size
                if i != len(X):
                    assert i < len(X)
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

    def _fit(self, sample_weight):
        import tensorflow as tf
        import keras
        callbacks = None  # TODO
        with self.sess_.as_default():
            keras.backend.get_session().run(
                tf.global_variables_initializer()
            )
            self.model_.fit(
                    x=self.X_,
                    y=self.y_,
                    batch_size=self.batch_size,
                    epochs=self.max_iter,
                    verbose=self.verbose,
                    callbacks=callbacks,
                    shuffle=self.shuffle,
                    class_weight=self.class_weight,
                    sample_weight=sample_weight,
            )
        return self

    def _create_TF_sess(self):
        import tensorflow as tf
        if self.use_GPU:
            config = tf.ConfigProto(
                    allow_soft_placement=True,
                    )
        else:
            config = tf.ConfigProto(
                    allow_soft_placement=True,
                    device_count={"GPU": 0},
                    )
        # session = tf.Session(config=config)
        return session

    def _create_and_set_TF_sess(self):
        sess = self._create_TF_sess()
        self.sess_ = sess
        return self

    def _create_TF_model(self):
        model = create_MNIST_CNN_model(input_shape=self.X_.shape[1:],
                                       optimizer=self.optimizer,
                                       n_classes=self.n_classes_)
        return model

    def _create_and_set_TF_model(self, filename=None, **kwargs):
        import tensorflow as tf
        import keras
        with self.sess_.as_default():
            if filename is None:
                if hasattr(self, "model_"):
                    pass
                    #for l in self.model_.layers:
                    #    if hasattr(l, "kernel_initializer"):
                    #        l.kernel.initializer.run(session=self.sess_)
                else:
                    model = self._create_TF_model(**kwargs)
                    self.model_ = model
            else:
                model = keras.models.load_model(filename)
                self.model_ = model
        return self

    def _create_and_set_TF_sess_and_model(self, filename=None, **kwargs):
        self._create_and_set_TF_sess()
        self._create_and_set_TF_model(filename=filename, **kwargs)
        return self

    def get_params(self, deep=True):
        return {
                "optimizer": self.optimizer,
                "batch_size": self.batch_size,
                "max_iter": self.max_iter,
                "shuffle": self.shuffle,
                "verbose": self.verbose,
                "class_weight": self.class_weight,
                "use_GPU": self.use_GPU,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        if "model_" in state:
            with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
                with self.sess_.as_default():
                    self.model_.save(f.name, overwrite=True)
                model_f_str = f.read()
            state["model_f_"] = model_f_str
            del state["model_"]
        if "sess_" in state:
            del state["sess_"]
        return state

    def __setstate__(self, state):
        import keras
        import tensorflow as tf
        self.__dict__.update(state)
        if "model_f_" in state:
            with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
                f.write(state["model_f_"])
                f.flush()
                self._create_and_set_TF_sess_and_model(filename=f.name)
            del state["model_f_"]
        return state

    def __del__(self):
        pass
        import keras.backend
        K = keras.backend
        if hasattr(self, "sess_"):
            with self.sess_.as_default():
                K.clear_session()
