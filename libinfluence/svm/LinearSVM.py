import sklearn
import sklearn.svm
import sklearn.utils.multiclass
import sklearn.utils.validation

import autograd
import autograd.misc
import autograd.numpy as np

import scipy.linalg as slin

class LinearSVM(sklearn.base.BaseEstimator,
                sklearn.base.ClassifierMixin):
    """
    A linear SVM.
    This is a composition of tensorflow and sklearn.
    """
    def __init__(self, C=1.0, probability=False, shrinking=True,
                 tol=1e-3, class_weight=None, verbose=False,
                 max_iter=-1, random_state=None,
                 fit_intercept=True,
                 ):
        self.C = C
        self.probability = probability
        self.shrinking = shrinking
        self.tol = tol
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.random_state = random_state
        self.fit_intercept = fit_intercept  # TODO add full support

    def decision_function(self, X):
        if not hasattr(self, "sklearn_model"):
            raise ValueError("Model was not fit")
        return self.sklearn_model.decision_function(X)

    def densify(self):
        if not hasattr(self, "sklearn_model"):
            raise ValueError("Model was not fit")
        return self.sklearn_model.densify()

    def fit(self, X, y, sample_weight=None):
        sklearn.utils.validation.check_X_y(X, y)
        sklearn.utils.multiclass.check_classification_targets(y)
        self.X_ = X
        self.y_ = y
        self.classes_ = sklearn.utils.multiclass.unique_labels(y)
        self.sklearn_model_ = sklearn.svm.SVC(
                C=self.C,
                kernel="linear",
                probability=self.probability,
                shrinking=self.shrinking,
                tol=self.tol,
                class_weight=self.class_weight,
                verbose=self.verbose,
                max_iter=self.max_iter,
                random_state=self.random_state,
                )
        self.sklearn_model_.fit(X, y, sample_weight)
        return self

    @staticmethod
    def _pred(X, W):
        probs = (np.dot(X, W.T) >= 0).astype(np.float)
        probs = probs * 2 - 1
        return probs

    @staticmethod
    def _pred_score(X, W):
        return np.dot(X, W.T)

    @staticmethod
    def _l2_norm(W):
        flattened, _ = autograd.misc.flatten(W)
        return np.dot(flattened, flattened)

    @staticmethod
    def _loss(W, X, y, C, intercept=False, temp=1e-2):
        pred_score = LinearSVM._pred_score(X, W)
        if temp == 0:
            loss_pred = C * np.sum(hinge(y * pred_score))
        else:
            loss_pred = C * np.sum(smooth_hinge(y * pred_score, temp))
        if intercept:
            loss_reg = LinearSVM._l2_norm(W[:-1])
        else:
            loss_reg = LinearSVM._l2_norm(W)
        loss = loss_pred + loss_reg
        return loss

    def pred(self, X):
        W_b = self.W_b
        X_b = LinearSVM._X_b(X)
        pred = LinearSVM._pred(X_b, W_b)
        return pred

    def pred_score(self, X):
        W_b = self.W_b
        X_b = LinearSVM._X_b(X)
        return np.dot(X_b, W_b.T)

    def pred_losses(self, X=None, y=None):
        if X is None or y is None:
            assert X is None and y is None
            X = self.X_
            y = self.y_
        loss_pred = 1.0 - y * self.pred_score(X)
        loss_pred[loss_pred < 0.0] = 0.0
        loss_pred = self.C * loss_pred
        return loss_pred

    @staticmethod
    def _X_b(X):
        """bias trick pad X"""
        X_b = np.concatenate((X,
                              np.ones((len(X), 1))),
                             axis=1)
        return X_b

    def LOO_influence(self,
                      smooth_hinge_temperature=1e-2,
                      X=None,
                      include_reg=True,
                      include_hessian=True,
                      flip_predicted_labels=False):
        """
        Non-sklearn function.
        Get the Leave One Out (LOO) influence of all the points in X.
        """
        raise NotImplementedError("LOO_influence is not implemented")

    def retrain_LOO_loss(self, include_reg=False):
        """
        Non-sklearn function.
        Get the Leave One Out (LOO) influence of all the points in the training
        set by retraining. This function is slow!
        """
        X = self.X_
        y = self.y_
        LOO_infs = np.zeros(len(X))
        assert len(X) == len(y)
        assert not include_reg
        curr_losses = self.sum_pred_losses(X=X, y=y)
        # TODO can use multiprocessing
        for i in range(len(X)):
            mask = np.ones(len(X), dtype=np.bool)
            mask[i] = False
            X_without_i = X[mask]
            y_without_i = y[mask]
            self.fit(X_without_i, y_without_i)
            curr_losses_i = self.sum_pred_losses(X_without_i, y_without_i)
            LOO_inf = curr_losses - curr_losses_i
            LOO_infs[i] = LOO_inf

        self.fit(X, y)

        return LOO_infs

    def get_params(self, deep=True):
        return {
                "C": self.C,
                "probability": self.probability,
                "shrinking": self.shrinking,
                "tol": self.tol,
                "class_weight": self.class_weight,
                "verbose": self.verbose,
                "max_iter": self.max_iter,
                "random_state": self.random_state,
                "fit_intercept": self.fit_intercept
                }

    def predict(self, X):
        sklearn.utils.validation.check_is_fitted(self, ["X_", "y_"])
        if not hasattr(self, "sklearn_model"):
            raise ValueError("Model was not fit")
        X = sklearn.utils.validation.check_array(X)
        return self.sklearn_model.predict(X)

    @property
    def predict_log_proba(self):
        if not self.probability:
            raise AttributeError("Must set probability in __init__ for"
                                 " probabilities.")
        return self._predict_log_proba

    def _predict_log_proba(self, X):
        if not hasattr(self, "sklearn_model"):
            raise ValueError("Model was not fit")
        return self.sklearn_model.predict_log_proba()

    @property
    def predict_proba(self):
        if not self.probability:
            raise AttributeError("Must set probability in __init__ for"
                                 " probabilities.")
        return self._predict_proba

    def _predict_proba(self, X):
        if not hasattr(self, "sklearn_model"):
            raise ValueError("Model was not fit")
        return self.sklearn_model.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        if not hasattr(self, "sklearn_model"):
            raise ValueError("Model was not fit")
        return self.sklearn_model.score(X, y, sample_weight)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def sparsity(self):
        if not hasattr(self, "sklearn_model"):
            raise ValueError("Model was not fit")
        return self.sklearn_model.sparsify()

    @property
    def coef_(self):
        if not hasattr(self, "sklearn_model"):
            raise ValueError("Model was not fit")
        return self.sklearn_model.coef_

    @coef_.setter
    def coef_(self, value):
        if not hasattr(self, "sklearn_model"):
            raise ValueError("Model was not fit")
        # TODO this is readonly, must change dual
        self.sklearn_model.coef_ = value

    @property
    def intercept_(self):
        if not hasattr(self, "sklearn_model"):
            raise ValueError("Model was not fit")
        if self.fit_intercept:
            return self.sklearn_model.intercept_
        else:
            return None

    @intercept_.setter
    def intercept_(self, value):
        if not hasattr(self, "sklearn_model"):
            raise ValueError("Model was not fit")
        self.sklearn_model.intercept_ = value

    @property
    def W_b(self):
        """
        Gets the weights concatenated with the bias (bias trick)
        """
        weights = np.array(self.coef_)
        assert weights.shape == (1, self.X_.shape[1])
        weights = weights[0, :]  # Unpack the weights
        if not self.fit_intercept:
            return weights
        else:
            intercept = np.array(self.intercept_)
            assert intercept.shape == (1,)
            W_b = np.concatenate((weights, intercept), axis=0)
            return W_b

    @property
    def n_iter_(self):
        if not hasattr(self, "sklearn_model"):
            raise ValueError("Model was not fit")
        return 1
        # TODO hack this will never return
        return self.sklearn_model.n_iter_

    @W_b.setter
    def W_b(self, value):
        if self.fit_intercept:
            self.coef_ = np.array([value[:-1]])
            self.intercept_ = value[-1:]
        else:
            self.coef_ = np.array([value])

    @property
    def support_(self):
        if not hasattr(self, "sklearn_model"):
            raise ValueError("Model was not fit")
        return self.sklearn_model.support_

    @property
    def support_vectors_(self):
        if not hasattr(self, "sklearn_model"):
            raise ValueError("Model was not fit")
        return self.sklearn_model.support_vectors_

    @property
    def n_support_(self):
        if not hasattr(self, "sklearn_model"):
            raise ValueError("Model was not fit")
        return self.sklearn_model.n_support_

    @property
    def dual_coef_(self):
        if not hasattr(self, "sklearn_model"):
            raise ValueError("Model was not fit")
        return self.sklearn_model.dual_coef_

    @property
    def sklearn_model(self):
        return self.sklearn_model_

    def is_support_vector(self):
        sv = self.support_
        is_sv = np.zeros(len(self.X_), dtype=np.bool)
        is_sv[sv] = True
        return is_sv

def hinge(s):
    return np.max(0, 1 - s)

def smooth_hinge(s, temp):
    return temp * np.log(1 + np.exp((1 - s) / temp))