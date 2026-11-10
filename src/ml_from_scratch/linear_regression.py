"""Linear regression implemented with NumPy and batch gradient descent."""

import numpy as np


class LinearRegression:
    """Fit y = Xw + b using batch gradient descent."""

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1_000):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if n_iterations < 1:
            raise ValueError("n_iterations must be at least 1")
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
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.cost_history = []

        for _ in range(self.n_iterations):
            predictions = X @ self.weights + self.bias
            errors = predictions - y
            self.weights -= self.learning_rate * (X.T @ errors) / n_samples
            self.bias -= self.learning_rate * np.mean(errors)
            updated_errors = X @ self.weights + self.bias - y
            self.cost_history.append(float(np.mean(updated_errors ** 2) / 2))
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
