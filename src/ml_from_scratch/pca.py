"""Principal component analysis implemented with NumPy's SVD."""

import numpy as np


class PCA:
    """Project centered data onto orthogonal directions of greatest variance.

    Parameters
    ----------
    n_components : int or float or None
        Integer number of components, a float in (0, 1] specifying the
        cumulative explained-variance target, or None to retain all available
        components. For a zero-variance dataset, a fractional target retains
        one component.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.n_components_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = None

    @staticmethod
    def _validate_X(X):
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must be a numeric 2D array") from exc
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def _resolve_n_components(self, variance_ratio, n_available):
        value = self.n_components
        if value is None:
            return n_available
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("n_components must be an integer, a float in (0, 1], or None")
        if isinstance(value, (int, np.integer)):
            if not 1 <= value <= n_available:
                raise ValueError("integer n_components must be between 1 and min(X.shape)")
            return int(value)
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError("fractional n_components must be in (0, 1]")
            if variance_ratio.sum() <= 0.0:
                return 1
            cumulative = np.cumsum(variance_ratio)
            return int(np.searchsorted(cumulative, value, side="left") + 1)
        raise ValueError("n_components must be an integer, a float in (0, 1], or None")

    def fit(self, X):
        """Learn the mean and principal directions from a finite 2D array."""
        X = self._validate_X(X)
        with np.errstate(over="ignore", invalid="ignore"):
            mean = np.mean(X, axis=0)
            centered = X - mean
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(centered)):
            raise ValueError("X must have finite means and centered values")

        try:
            _, singular_values, right_vectors = np.linalg.svd(
                centered, full_matrices=False
            )
        except np.linalg.LinAlgError as exc:
            raise FloatingPointError("SVD failed to converge") from exc

        denominator = max(X.shape[0] - 1, 1)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            variance = singular_values ** 2 / denominator
            total_variance = variance.sum()
            if total_variance > 0.0:
                variance_ratio = variance / total_variance
            else:
                variance_ratio = np.zeros_like(variance)

        if (not np.all(np.isfinite(singular_values))
                or not np.all(np.isfinite(variance))
                or not np.isfinite(total_variance)):
            raise FloatingPointError("PCA produced non-finite variance values")

        n_components = self._resolve_n_components(
            variance_ratio, min(X.shape)
        )
        components = right_vectors[:n_components].copy()
        selected_variance = variance[:n_components].copy()
        selected_ratio = variance_ratio[:n_components].copy()
        selected_singular_values = singular_values[:n_components].copy()

        # Update fitted state only after every validation and decomposition succeeds.
        self.mean_ = mean.copy()
        self.components_ = components
        self.explained_variance_ = selected_variance
        self.explained_variance_ratio_ = selected_ratio
        self.singular_values_ = selected_singular_values
        self.n_components_ = n_components
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = X.shape[0]
        return self

    def transform(self, X):
        """Project samples onto the learned principal component axes."""
        if self.components_ is None:
            raise RuntimeError("Call fit before transform")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number of features used in fit")
        with np.errstate(over="ignore", invalid="ignore"):
            transformed = (X - self.mean_) @ self.components_.T
        if not np.all(np.isfinite(transformed)):
            raise FloatingPointError("PCA transform produced non-finite values")
        return transformed

    def fit_transform(self, X):
        """Fit PCA and return the projected training data."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed):
        """Reconstruct samples from their principal-component coordinates."""
        if self.components_ is None:
            raise RuntimeError("Call fit before inverse_transform")
        try:
            X_transformed = np.asarray(X_transformed, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("X_transformed must be a numeric 2D array") from exc
        if (X_transformed.ndim != 2 or X_transformed.shape[0] == 0
                or X_transformed.shape[1] != self.n_components_):
            raise ValueError("X_transformed must have shape (n_samples, n_components_)")
        if not np.all(np.isfinite(X_transformed)):
            raise ValueError("X_transformed must contain only finite values")
        with np.errstate(over="ignore", invalid="ignore"):
            reconstructed = X_transformed @ self.components_ + self.mean_
        if not np.all(np.isfinite(reconstructed)):
            raise FloatingPointError("PCA inverse_transform produced non-finite values")
        return reconstructed
