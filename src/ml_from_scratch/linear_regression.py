"""Linear regression implemented with NumPy and batch gradient descent."""

import numpy as np


class LinearRegression:
    """Fit y = Xw + b using batch gradient descent."""

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

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if X.shape[0] == 0:
            raise ValueError("training data cannot be empty")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain only finite values")

        n_samples, n_features = X.shape
        weights = np.zeros(n_features, dtype=float)
        bias = 0.0
        cost_history = []

        # Keep a previously fitted model intact unless every training step succeeds.
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(self.n_iterations):
                predictions = X @ weights + bias
                if not np.all(np.isfinite(predictions)):
                    raise FloatingPointError(
                        "training diverged: non-finite predictions encountered"
                    )

                errors = predictions - y
                weights -= self.learning_rate * (X.T @ errors) / n_samples
                bias -= self.learning_rate * np.mean(errors)

                if not np.all(np.isfinite(weights)) or not np.isfinite(bias):
                    raise FloatingPointError(
                        "training diverged: non-finite parameters encountered"
                    )

                updated_predictions = X @ weights + bias
                if not np.all(np.isfinite(updated_predictions)):
                    raise FloatingPointError(
                        "training diverged: non-finite predictions encountered"
                    )

                updated_errors = updated_predictions - y
                cost = np.mean(updated_errors ** 2) / 2
                if not np.isfinite(cost):
                    raise FloatingPointError(
                        "training diverged: non-finite loss encountered"
                    )
                cost_history.append(float(cost))

        self.weights = weights
        self.bias = float(bias)
        self.cost_history = cost_history
        return self

    def predict(self, X):
        if self.weights is None:
            raise RuntimeError("Call fit before predict")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.weights.shape[0]:
            raise ValueError("X must be 2D with the same number of features used in fit")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X @ self.weights + self.bias
