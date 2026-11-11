"""Logistic regression and loss utilities implemented with NumPy."""

import numpy as np


def sigmoid(z):
    """Return the logistic sigmoid of a scalar or array, stably.

    The log-add-exp formulation avoids overflow for very large positive or
    negative inputs and preserves the shape of array inputs.
    """
    values = np.asarray(z, dtype=float)
    result = np.exp(-np.logaddexp(0.0, -values))
    return float(result) if result.ndim == 0 else result


def binary_cross_entropy_with_logits(y_true, logits, reduction="mean"):
    """Compute binary cross-entropy directly from logits.

    Args:
        y_true: Targets in [0, 1], with the same shape as logits. Both hard
            labels (0 or 1) and soft labels (probabilities) are supported.
        logits: Raw, unscaled model scores.
        reduction: "none" for element-wise losses, "sum" for their sum,
            or "mean" for their average.

    Computing logaddexp(0, logits) - y_true * logits avoids first converting
    logits to probabilities, which is more stable for extreme values.

    Raises:
        ValueError: If inputs are empty, have different shapes, contain
            non-finite values, targets are outside [0, 1], or reduction is
            unsupported.
    """
    targets = np.asarray(y_true, dtype=float)
    scores = np.asarray(logits, dtype=float)

    if targets.size == 0 or scores.size == 0:
        raise ValueError("y_true and logits must be non-empty")
    if targets.shape != scores.shape:
        raise ValueError("y_true and logits must have the same shape")
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(scores)):
        raise ValueError("y_true and logits must contain only finite values")
    if np.any((targets < 0.0) | (targets > 1.0)):
        raise ValueError("y_true values must be between 0 and 1")
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")

    losses = np.logaddexp(0.0, scores) - targets * scores
    if reduction == "none":
        return float(losses) if losses.ndim == 0 else losses
    if reduction == "sum":
        return float(np.sum(losses))
    return float(np.mean(losses))


class LogisticRegression:
    """Binary logistic regression trained with batch gradient descent.

    predict_proba returns two columns in class order [0, 1].
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1_000):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive number")
        if (
            isinstance(n_iterations, (bool, np.bool_))
            or not isinstance(n_iterations, (int, np.integer))
            or n_iterations < 1
        ):
            raise ValueError("n_iterations must be a positive integer")
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0.0
        self.cost_history = []
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X, y):
        """Fit the classifier and return this estimator."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("training data must contain samples and features")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain only finite values")
        if not np.all((y == 0.0) | (y == 1.0)):
            raise ValueError("y must contain only binary labels 0 and 1")
        if np.unique(y).size != 2:
            raise ValueError("y must contain both classes 0 and 1")

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.cost_history = []

        for _ in range(self.n_iterations):
            logits = X @ self.weights + self.bias
            errors = sigmoid(logits) - y
            self.weights -= self.learning_rate * (X.T @ errors) / n_samples
            self.bias -= self.learning_rate * np.mean(errors)

            updated_logits = X @ self.weights + self.bias
            self.cost_history.append(
                binary_cross_entropy_with_logits(y, updated_logits)
            )
        return self

    def _validated_features(self, X):
        if self.weights is None:
            raise RuntimeError("Call fit before prediction")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.weights.shape[0]:
            raise ValueError(
                "X must be 2D with the same number of features used in fit"
            )
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def predict_proba(self, X):
        """Return probabilities for classes 0 and 1, in that column order."""
        X = self._validated_features(X)
        positive_probability = sigmoid(X @ self.weights + self.bias)
        return np.column_stack((1.0 - positive_probability, positive_probability))

    def predict(self, X):
        """Predict binary labels using a 0.5 probability threshold."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
