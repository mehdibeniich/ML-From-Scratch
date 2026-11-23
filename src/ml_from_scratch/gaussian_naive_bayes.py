"""Gaussian Naive Bayes classifier implemented with NumPy."""

import numpy as np


def _contains_label(labels, candidate):
    for label in labels:
        try:
            equal = candidate == label
            if np.ndim(equal) == 0 and bool(equal):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _unique_labels(y):
    labels = []
    for value in y:
        if value is None:
            raise ValueError("y must not contain missing labels")
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValueError("y must not contain missing or non-finite labels")
        if not _contains_label(labels, value):
            labels.append(value)
    if len(labels) < 2:
        raise ValueError("y must contain at least two distinct classes")
    try:
        labels = sorted(labels)
    except TypeError:
        pass
    return np.asarray(labels, dtype=object)


class GaussianNB:
    """Gaussian Naive Bayes classifier for numeric features.

    Each feature is modeled as a Gaussian distribution within each class.
    var_smoothing adds a global variance-scaled floor to class variances
    to keep constant and nearly constant features numerically usable.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        if isinstance(var_smoothing, (bool, np.bool_)) or not np.isscalar(
            var_smoothing
        ):
            raise ValueError("var_smoothing must be a finite non-negative number")
        try:
            smoothing = float(var_smoothing)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "var_smoothing must be a finite non-negative number"
            ) from exc
        if not np.isfinite(smoothing) or smoothing < 0:
            raise ValueError("var_smoothing must be a finite non-negative number")
        self.var_smoothing = smoothing
        self.classes_ = None
        self.class_count_ = None
        self.class_prior_ = None
        self.theta_ = None
        self.var_ = None
        self.epsilon_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        """Estimate class priors, feature means, and variances.

        Returns this estimator. A failed refit leaves the previous fitted
        parameters unchanged.
        """
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must contain numeric features") from exc
        y = np.asarray(y, dtype=object)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("training data must contain samples and features")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        classes = _unique_labels(y)
        counts = np.array(
            [sum(value == label for value in y) for label in classes],
            dtype=float,
        )
        priors = counts / X.shape[0]
        means = np.vstack([np.mean(X[y == label], axis=0) for label in classes])
        raw_variances = np.vstack(
            [np.var(X[y == label], axis=0) for label in classes]
        )

        global_variance = float(np.max(np.var(X, axis=0)))
        epsilon = max(
            self.var_smoothing * global_variance,
            np.finfo(float).tiny,
        )
        variances = raw_variances + epsilon

        if (
            not np.all(np.isfinite(means))
            or not np.all(np.isfinite(variances))
            or np.any(variances <= 0)
            or not np.all(np.isfinite(priors))
        ):
            raise FloatingPointError("GaussianNB fitting produced invalid statistics")

        self.classes_ = classes
        self.class_count_ = counts
        self.class_prior_ = priors
        self.theta_ = means
        self.var_ = variances
        self.epsilon_ = float(epsilon)
        self.n_features_in_ = X.shape[1]
        return self

    def _validated_features(self, X):
        if self.classes_ is None:
            raise RuntimeError("Call fit before prediction")
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must contain numeric features") from exc
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X must be 2D with the same number of features used in fit"
            )
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def _joint_log_likelihood(self, X):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            log_priors = np.log(self.class_prior_)
            log_determinants = np.sum(np.log(2.0 * np.pi * self.var_), axis=1)
            squared_distances = (X[:, None, :] - self.theta_[None, :, :]) ** 2
            quadratic_terms = np.sum(
                squared_distances / self.var_[None, :, :], axis=2
            )
            scores = -0.5 * (
                log_determinants[None, :] + quadratic_terms
            ) + log_priors[None, :]
        if np.any(np.isnan(scores)) or np.all(np.isneginf(scores), axis=1).any():
            raise FloatingPointError(
                "prediction produced non-finite Gaussian log probabilities"
            )
        return scores

    def predict_proba(self, X):
        """Return normalized class probabilities in classes_ order."""
        X = self._validated_features(X)
        scores = self._joint_log_likelihood(X)
        row_max = np.max(scores, axis=1, keepdims=True)
        with np.errstate(under="ignore", invalid="ignore"):
            exp_scores = np.exp(scores - row_max)
            probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        if not np.all(np.isfinite(probabilities)):
            raise FloatingPointError("prediction produced invalid probabilities")
        return probabilities

    def predict(self, X):
        """Predict the class with the highest posterior probability."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
