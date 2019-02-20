import sklearn
import sklearn.linear_model
import sklearn.utils.multiclass
import sklearn.utils.validation

import autograd
import autograd.misc
import autograd.numpy as np

import scipy.linalg as slin


class LogisticRegression(sklearn.base.BaseEstimator,
                         sklearn.base.ClassifierMixin):
    """
    A logitic regression model.
    """
    def __init__(self, penalty="l2", dual=False, tol=1e-7,
                 C=1.0, fit_intercept=True, intercept_scaling=1.0,
                 class_weight=None, random_state=None, solver="lbfgs",
                 max_iter=1000000, multi_class="ovr", verbose=0,
                 warm_start=False, n_jobs=1):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs

    def decision_function(self, X):
        assert len(X.shape) == 2, "X must be 2d"
        return self.sklearn_model.decision_function(X)

    def densify(self):
        return self.sklearn_model.densify()

    def fit(self, X, y, sample_weight=None):
        assert len(X.shape) == 2, "X must be 2d"
        sklearn.utils.validation.check_X_y(X, y)
        sklearn.utils.multiclass.check_classification_targets(y)
        self.X_ = np.array(X)
        self.y_ = np.array(y)
        self.classes_ = sklearn.utils.multiclass.unique_labels(y)
        self.n_classes_ = len(self.classes_)
        assert self.n_classes_ >= 1, "Must have more than 1 class"
        if self.multi_class != "multinomial" and self.n_classes_ > 2:
            raise NotImplementedError("Only multinomial multiclass is"
                                      " supported.")
        self.sklearn_model_ = sklearn.linear_model.LogisticRegression(
                penalty=self.penalty,
                dual=self.dual,
                tol=self.tol,
                fit_intercept=self.fit_intercept,
                intercept_scaling=self.intercept_scaling,
                class_weight=self.class_weight,
                random_state=self.random_state,
                solver=self.solver,
                max_iter=self.max_iter,
                multi_class=self.multi_class,
                verbose=self.verbose,
                warm_start=self.warm_start,
                n_jobs=self.n_jobs)
        self.sklearn_model.fit(X, y, sample_weight)

        return self

    @staticmethod
    def _logit(X, W):
        probs = sigmoid(np.dot(X, W.T))
        return probs

    def logit(self, X):
        assert len(X.shape) == 2, "X must be 2d"
        W_b = self.W_b
        X_b = LogisticRegression._X_b(X)
        probs = LogisticRegression._logit(X_b, W_b)
        return probs

    def logit_predict(self, X):
        assert len(X.shape) == 2, "X must be 2d"
        probs = self.logit(X)
        assert len(probs.shape) == 1
        pred = probs > 0.5
        return pred

    @staticmethod
    def _l2_norm(W):
        # Flattening makes this work regardless of shape/outputs
        flattened, _ = autograd.misc.flatten(W)
        return np.dot(flattened, flattened)

    @staticmethod
    def _log_posterior(W, X, y, L2_alpha, intercept=False, n_classes=2,
                       W_shape=None):
        assert len(X.shape) == 2, "X must be 2d"
        if n_classes <= 2:
            pred = LogisticRegression._logit(X, W)
            assert len(pred.shape) <= 2
            log_lik = np.sum(np.log(pred * y + (1 - pred) * (1 - y)))
        else:
            raise RuntimeError("Multiclass is not supported")
        if intercept:
            log_prior = -L2_alpha * LogisticRegression._l2_norm(W[:-1])
        else:
            log_prior = -L2_alpha * LogisticRegression._l2_norm(W)
        log_posterior = log_prior + log_lik
        #print("log_posterior", log_posterior)
        return log_posterior


    @staticmethod
    def _posterior(W, X, y, L2_alpha, intercept=False):
        assert len(X.shape) == 2, "X must be 2d"
        pred = LogisticRegression._logit(X, W)
        indiv_lik = sigmoid(pred * y)
        lik = np.prod(indiv_lik)
        if intercept:
            prior = -L2_alpha * LogisticRegression._l2_norm(W[:-1])
        else:
            prior = -L2_alpha * LogisticRegression._l2_norm(W)
        posterior = prior + lik
        return posterior

    @staticmethod
    def _log_loss(W, X, y, L2_alpha, intercept=False, n_classes=2,
                  W_shape=None):
        return -LogisticRegression._log_posterior(W,
                                                  X,
                                                  y,
                                                  L2_alpha,
                                                  intercept,
                                                  n_classes=n_classes,
                                                  W_shape=W_shape)

    @staticmethod
    def _loss(W, X, y, L2_alpha, intercept=False):
        assert len(X.shape) == 2, "X must be 2d"
        return -LogisticRegression._posterior(W, X, y, L2_alpha, intercept)

    def mean_log_loss(self, X=None, y=None, L2_alpha=None):
        return np.mean(self.log_losses(X=X,
                                       y=y,
                                       L2_alpha=L2_alpha))

    def sum_log_loss(self, X=None, y=None, L2_alpha=None):
        return np.sum(self.log_losses(X=X,
                                      y=y,
                                      L2_alpha=L2_alpha))

    def pred_losses(self, X=None, y=None):
        if X is None or y is None:
            assert X is None and y is None
            X = self.X_
            y = self.y_
        assert len(X.shape) == 2, "X must be 2d"
        y_prob = self.logit(X)
        log_y_prob = np.log(y_prob)
        losses = y * log_y_prob
        return losses

    def mean_pred_losses(self, X=None, y=None):
        return np.mean(self.pred_losses(X=X,
                                        y=y))

    def sum_pred_losses(self, X=None, y=None):
        return np.sum(self.pred_losses(X=X,
                                       y=y))

    def log_losses(self, X=None, y=None, L2_alpha=None):
        # TODO check
        if X is None or y is None:
            assert X is None and y is None
            X = self.X_
            y = self.y_
        assert len(X.shape) == 2, "X must be 2d"
        X_b = LogisticRegression._X_b(X)
        if L2_alpha is None:
            L2_alpha = self.L2_alpha

        W_b = self.W_b
        losses = []
        for _xx, _yy in zip(X_b, y):
            loss = LogisticRegression._log_loss(W=W_b.flatten(),
                                                X=_xx.reshape(1, -1),
                                                y=_yy,
                                                L2_alpha=L2_alpha,
                                                intercept=self.fit_intercept,
                                                n_classes=self.n_classes_,
                                                W_shape=W_b.shape)
            losses.append(loss)
        losses = np.array(losses)
        return losses

    def losses(self, X=None, y=None, L2_alpha=None):
        # TODO check
        if X is None or y is None:
            assert X is None and y is None
            X = self.X_
            y = self.y_
        X_b = LogisticRegression._X_b(X)
        if L2_alpha is None:
            L2_alpha = self.L2_alpha

        W_b = self.W_b
        losses = []
        for _xx, _yy in zip(X_b, y):
            loss = LogisticRegression._loss(W=W_b,
                                            X=_xx,
                                            y=_yy,
                                            L2_alpha=L2_alpha,
                                            intercept=self.fit_intercept)
            losses.append(loss)
        losses = np.array(losses)
        return losses

    def grad_losses(self, X=None, y=None, L2_alpha=None):
        if X is None or y is None:
            assert X is None and y is None
            X = self.X_
            y = self.y_
        W_b = self.W_b
        X_b = LogisticRegression._X_b(X)
        if L2_alpha is None:
            L2_alpha = self.L2_alpha
        grad_loss = autograd.grad(LogisticRegression._log_loss)
        grads = []
        for _xx, _yy in zip(X_b, y):
            grad = grad_loss(W_b,
                             _xx.reshape(1, -1),
                             _yy,
                             L2_alpha,
                             intercept=self.fit_intercept,
                             n_classes=self.n_classes_,
                             W_shape=W_b.shape)
            grads.append(grad)
        grads = np.array(grads)
        if len(W_b.shape) == 1:
            assert grads.shape[1] == W_b.shape[0]
        return grads

    def grad_loss(self, X=None, y=None, L2_alpha=None):
        if X is None or y is None:
            assert X is None and y is None
            X = self.X_
            y = self.y_
        W_b = self.W_b
        X_b = LogisticRegression._X_b(X)
        if L2_alpha is None:
            L2_alpha = self.L2_alpha
        grad_loss = autograd.grad(LogisticRegression._log_loss)
        curr_grad = grad_loss(W_b,
                              X_b,
                              y,
                              L2_alpha,
                              intercept=self.fit_intercept,
                              n_classes=self.n_classes_,
                              W_shape=W_b.shape)
        assert curr_grad.shape == W_b.shape
        return curr_grad

    def hess_losses(self, X=None, y=None, L2_alpha=None):
        if X is None or y is None:
            assert X is None and y is None
            X = self.X_
            y = self.y_
        if L2_alpha is None:
            L2_alpha = self.L2_alpha
        W_b = self.W_b
        X_b = LogisticRegression._X_b(X)
        hess_loss = autograd.hessian(LogisticRegression._log_loss)
        hesses = []
        for _xx, _yy in zip(X_b, y):
            #print("W_B flat", W_b.flatten().shape)
            hess = hess_loss(W_b.flatten(),
                             _xx.reshape(1, -1),  # TODO remove reshape?
                             _yy,
                             L2_alpha,
                             intercept=self.fit_intercept,
                             n_classes=self.n_classes_,
                             W_shape=W_b.shape)
            if hess.shape[0] == 1:
                # TODO make this not happen
                hess = np.squeeze(hess)
            #print("Hess", hess.shape)
            hesses.append(hess)
        hesses = np.array(hesses)
        #print("hesses", hesses.shape)
        if len(W_b.shape) == 1:
            assert hesses.shape[1] == W_b.shape[0]
            assert hesses.shape[2] == W_b.shape[0]
        return hesses

    def hess_loss(self, X=None, y=None, L2_alpha=None):
        if X is None or y is None:
            assert X is None and y is None
            X = self.X_
            y = self.y_
        if L2_alpha is None:
            L2_alpha = self.L2_alpha
        W_b = self.W_b
        X_b = LogisticRegression._X_b(X)
        hess_loss = autograd.hessian(LogisticRegression._log_loss)
        curr_hess = hess_loss(W_b.flatten(),
                              X_b,
                              y,
                              L2_alpha,
                              intercept=self.fit_intercept,
                              n_classes=self.n_classes_,
                              W_shape=W_b.shape)
        if len(W_b.shape) == 1:
            assert curr_hess.shape[0] == W_b.shape[0]
            assert curr_hess.shape[1] == W_b.shape[0]

        if curr_hess.shape[0] == 1:
            # TODO make this not happen
            curr_hess = np.squeeze(curr_hess)
        return curr_hess

    @staticmethod
    def _X_b(X):
        """bias trick pad X"""
        X_b = np.concatenate((X,
                              np.ones((len(X), 1))),
                             axis=1)
        return X_b

    def LOO_influence(self, X=None, include_reg=False, include_hessian=True,
                      flip_predicted_labels=False):
        """
        Non-sklearn function.
        Get the Leave One Out (LOO) influence of all the points in X.
        """
        if X is not None:
            y_inspect = self.predict(X)
            if self.multi_class == "multinomial" and self.n_classes_ > 2:
                pass
                assert not flip_predicted_labels
            else:
                assert set(np.unique(y_inspect)).issubset(set([0, 1])), \
                    "y values must be 0 or 1"
                if flip_predicted_labels:
                    y_inspect = y_inspect.astype(np.bool)
                    y_inspect = np.invert(y_inspect)
                    y_inspect = y_inspect.astype(np.int)
        else:
            X = self.X_
            y_inspect = self.y_
            if self.multi_class == "multinomial" and self.n_classes_ > 2:
                pass
            else:
                assert set(np.unique(y_inspect)).issubset(set([0, 1])), \
                    "y values must be 0 or 1"

        y_fit = self.y_

        if self.multi_class == "multinomial" and self.n_classes_ > 2:
            pass
        else:
            assert set(np.unique(y_fit)).issubset(set([0, 1])), \
                "y values must be 0 or 1"

        X_b_inspect = X
        X_b_fit = self.X_
        L2_alpha = self.L2_alpha
        if not include_reg:
            L2_alpha = 1e-10

        # Precompute global hess
        curr_hess = self.hess_loss(X_b_fit,
                                   y_fit,
                                   L2_alpha=L2_alpha,
                                   )
        print("curr_hess", curr_hess.shape, X_b_fit.shape, y_fit.shape)

        # TODO below fixes multinomial, but is slow
        #curr_hesses = self.hess_losses(X_b_fit,
        #                           y_fit,
        #                           L2_alpha=L2_alpha,
        #                           )
        #curr_hess = np.mean(curr_hesses, axis=0)
        #print("curr_hesses",curr_hesses.shape, X_b_fit.shape, y_fit.shape)

        inv_emp_hess = slin.inv(curr_hess)  # invert

        LOO_infs = np.zeros(len(X))
        assert len(X_b_inspect) == len(y_inspect)

        for i, (X_b_i, y_i) in enumerate(zip(X_b_inspect, y_inspect)):
            curr_loss_i = self.grad_loss(X_b_i.reshape(1, -1),
                                         y_i,
                                         L2_alpha=L2_alpha,
                                         ).flatten()
            if include_hessian:
                LOO_inf = -curr_loss_i.dot(inv_emp_hess).dot(curr_loss_i.T)
            else:
                LOO_inf = -curr_loss_i.dot(curr_loss_i.T)
            LOO_infs[i] = LOO_inf

        return LOO_infs

    def inf_up_loss_influence(self,
                              X_test,
                              y_test,
                              include_reg=False,
                              include_hessian=True,
                              ):
        """
        Non-sklearn function.
        This is the influence of a training point on a testing point.
        """
        y_fit = self.y_
        assert set(np.unique(y_fit)).issubset(set([0, 1])), \
            "y values must be 0 or 1"

        # TODO put inside other functions
        X_fit = self.X_
        assert len(X_fit) == len(y_fit)

        L2_alpha = self.L2_alpha
        if not include_reg:
            L2_alpha = 1e-10

        # Precompute global hess
        curr_hess = self.hess_loss(X_fit,
                                   y_fit,
                                   L2_alpha=L2_alpha,
                                   )
        inv_emp_hess = slin.inv(curr_hess)  # invert

        curr_losses_train = np.zeros((len(X_fit), len(self.W_b)))
        for i, (X_i, y_i) in enumerate(zip(X_fit, y_fit)):
            curr_loss_i = self.grad_loss(X_i.reshape(1, -1),
                                         y_i,
                                         L2_alpha=L2_alpha,
                                         )
            curr_losses_train[i] = curr_loss_i

        curr_losses_test = np.zeros((len(X_test), len(self.W_b)))
        for i, (X_i, y_i) in enumerate(zip(X_test, y_test)):
            curr_loss_i = self.grad_loss(X_i.reshape(1, -1),
                                         y_i,
                                         L2_alpha=L2_alpha,
                                         )
            curr_losses_test[i] = curr_loss_i

        # Rows are test points
        LOO_infs = np.zeros((len(X_test), len(X_fit)))
        for i, curr_loss_i in enumerate(curr_losses_test):
            for j, curr_loss_j in enumerate(curr_losses_train):
                if include_hessian:
                    LOO_inf = -curr_loss_i.dot(inv_emp_hess).dot(curr_loss_j.T)
                else:
                    LOO_inf = -curr_loss_i.dot(curr_loss_j.T)
                LOO_infs[i, j] = LOO_inf
        return LOO_infs

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
                "penalty": self.penalty,
                "dual": self.dual,
                "tol": self.tol,
                "C": self.C,
                "fit_intercept": self.fit_intercept,
                "intercept_scaling": self.intercept_scaling,
                "class_weight": self.class_weight,
                "random_state": self.random_state,
                "solver": self.solver,
                "max_iter": self.max_iter,
                "multi_class": self.multi_class,
                "verbose": self.verbose,
                "warm_start": self.warm_start,
                "n_jobs": self.n_jobs,
        }

    def predict(self, X):
        sklearn.utils.validation.check_is_fitted(self, ["X_", "y_"])
        X = sklearn.utils.validation.check_array(X)
        return self.sklearn_model.predict(X)

    def predict_log_proba(self, X):
        return self.sklearn_model.predict_log_proba(X)

    def predict_proba(self, X):
        return self.sklearn_model.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        return self.sklearn_model.score(X, y, sample_weight)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def sparsity(self):
        return self.sklearn_model.sparsify()

    @property
    def coef_(self):
        return self.sklearn_model.coef_

    @coef_.setter
    def coef_(self, value):
        self.sklearn_model.coef_ = value

    @property
    def intercept_(self):
        if self.fit_intercept:
            return self.sklearn_model.intercept_
        else:
            return None

    @intercept_.setter
    def intercept_(self, value):
        self.sklearn_model.intercept_ = value

    @property
    def n_iter_(self):
        return self.sklearn_model.n_iter_

    @property
    def W_b(self):
        """
        Gets the weights concatenated with the bias (bias trick)
        """
        weights = np.array(self.coef_)
        if self.multi_class == "multinomial" and self.n_classes_ > 2:
            assert weights.shape == (self.n_classes_, self.X_.shape[1])
            weights = weights[:, :]  # Unpack the weights
            if not self.fit_intercept:
                return weights
            intercept = np.array(self.intercept_).reshape(-1, 1)
            assert intercept.shape == (self.n_classes_, 1)
            W_b = np.concatenate((weights, intercept), axis=1)
            return W_b
        else:
            assert weights.shape == (1, self.X_.shape[1])
            weights = weights[0, :]  # Unpack the weights
            if not self.fit_intercept:
                return weights
            intercept = np.array(self.intercept_)
            assert intercept.shape == (1,)
            W_b = np.concatenate((weights, intercept), axis=0)
            return W_b

    @W_b.setter
    def W_b(self, value):
        # TODO multinomial
        if self.fit_intercept:
            self.coef_ = np.array([value[:-1]])
            self.intercept_ = value[-1:]
        else:
            self.coef_ = np.array([value])

    @property
    def L2_alpha(self):
        alpha = 1.0 / self.C
        return alpha

    @property
    def sklearn_model(self):
        if not hasattr(self, "sklearn_model_"):
            raise ValueError("Model was not fit")
        return self.sklearn_model_


def sigmoid(x):
    """
    Autograd sigmoid function
    """
    return 0.5 * (np.tanh(x / 2.0) + 1)

def indices_to_one_hot(data, nb_classes):
    """
    Convert an iterable of indices to one-hot encoded labels.
    From: https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
    """
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]
