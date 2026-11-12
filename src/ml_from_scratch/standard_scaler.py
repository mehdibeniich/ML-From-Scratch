"""Feature scaling utilities implemented with NumPy."""

import numpy as np


class StandardScaler:
    """Standardize each feature using its training-set mean and population std."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None

    @staticmethod
    def _validate_features(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def fit(self, X):
        """Learn per-feature means and scales from a non-empty 2D array."""
        X = self._validate_features(X)
        with np.errstate(over="ignore", invalid="ignore"):
            mean = np.mean(X, axis=0)
            scale = np.std(X, axis=0)

        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("X must have finite feature means and standard deviations")

        # A constant feature should become all zeros rather than divide by zero.
        scale[scale == 0.0] = 1.0

        # Assign only after all validation and calculations succeed.
        self.mean_ = mean
        self.scale_ = scale
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Standardize data with the statistics learned by fit."""
        if self.mean_ is None:
            raise RuntimeError("Call fit before transform")

        X = self._validate_features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X must have the same number of features used in fit"
            )

        with np.errstate(over="ignore", invalid="ignore"):
            transformed = (X - self.mean_) / self.scale_
        if not np.all(np.isfinite(transformed)):
            raise FloatingPointError("transformed features contain non-finite values")
        return transformed

    def fit_transform(self, X):
        """Fit the scaler and return the standardized training data."""
        return self.fit(X).transform(X)
