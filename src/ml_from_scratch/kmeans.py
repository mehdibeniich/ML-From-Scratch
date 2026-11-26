"""K-Means clustering implemented with NumPy."""

import numpy as np


class KMeans:
    """Cluster samples using Lloyd's algorithm and K-Means++ initialization.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    max_iter : int
        Maximum iterations for each initialization.
    tol : float
        Convergence tolerance on the sum of squared center movements.
    n_init : int
        Number of independent initializations; keep the lowest-inertia fit.
    random_state : int or None
        Seed for reproducible initialization.
    """

    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, n_init=10,
                 random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.n_features_in_ = None

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def _validate_params(self, n_samples):
        if isinstance(self.n_clusters, bool) or not isinstance(
                self.n_clusters, (int, np.integer)):
            raise ValueError("n_clusters must be a positive integer")
        if not 1 <= self.n_clusters <= n_samples:
            raise ValueError("n_clusters must be between 1 and n_samples")
        if isinstance(self.max_iter, bool) or not isinstance(
                self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if isinstance(self.n_init, bool) or not isinstance(
                self.n_init, (int, np.integer)) or self.n_init < 1:
            raise ValueError("n_init must be a positive integer")
        if not np.isscalar(self.tol) or not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol must be a finite non-negative number")

    @staticmethod
    def _squared_distances(X, centers):
        # Clamp tiny negative round-off from the algebraic distance formula.
        distances = (np.sum(X * X, axis=1)[:, None]
                     + np.sum(centers * centers, axis=1)[None, :]
                     - 2.0 * X @ centers.T)
        return np.maximum(distances, 0.0)

    def _initialize(self, X, rng):
        n_samples = X.shape[0]
        centers = np.empty((self.n_clusters, X.shape[1]), dtype=float)
        first = int(rng.integers(n_samples))
        centers[0] = X[first]
        closest = self._squared_distances(X, centers[:1])[:, 0]

        for index in range(1, self.n_clusters):
            total = closest.sum()
            if not np.isfinite(total) or total <= 0.0:
                # All remaining observations coincide with chosen centers.
                candidates = np.flatnonzero(
                    ~np.any(np.all(X[:, None, :] == centers[:index][None, :, :],
                                 axis=2), axis=1)
                choice = int(candidates[0]) if candidates.size else int(index % n_samples)
            else:
                choice = int(rng.choice(n_samples, p=closest / total))
            centers[index] = X[choice]
            new_distances = self._squared_distances(X, centers[index:index + 1])[:, 0]
            closest = np.minimum(closest, new_distances)
        return centers

    def _assign_and_recompute(self, X, centers):
        distances = self._squared_distances(X, centers)
        labels = np.argmin(distances, axis=1)
        minimum = distances[np.arange(X.shape[0]), labels]
        updated = centers.copy()

        counts = np.bincount(labels, minlength=self.n_clusters)
        for cluster in range(self.n_clusters):
            members = X[labels == cluster]
            if members.size:
                updated[cluster] = members.mean(axis=0)

        # Empty clusters are placed on currently worst-represented samples.
        # Distinct samples are chosen where possible to avoid repeated repairs.
        empty = np.flatnonzero(counts == 0)
        if empty.size:
            order = np.argsort(minimum, kind="stable")[::-1]
            used = set()
            for cluster in empty:
                sample = next((int(i) for i in order if int(i) not in used), int(order[0]))
                used.add(sample)
                updated[cluster] = X[sample]
        return labels, updated

    def fit(self, X):
        """Fit cluster centers and return self."""
        X = self._validate_X(X)
        self._validate_params(X.shape[0])
        rng = np.random.default_rng(self.random_state)
        best = None

        for _ in range(self.n_init):
            centers = self._initialize(X, rng)
            iterations = 0
            for iterations in range(1, self.max_iter + 1):
                labels, updated = self._assign_and_recompute(X, centers)
                shift = float(np.sum((updated - centers) ** 2))
                centers = updated
                if shift <= self.tol:
                    break

            distances = self._squared_distances(X, centers)
            labels = np.argmin(distances, axis=1)
            inertia = float(np.sum(distances[np.arange(X.shape[0]), labels]))
            if not np.isfinite(inertia):
                raise FloatingPointError("K-Means produced non-finite inertia")
            if best is None or inertia < best[0]:
                best = (inertia, centers.copy(), labels.copy(), iterations)

        self.inertia_, self.cluster_centers_, self.labels_, self.n_iter_ = best
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Assign each sample to its nearest learned center."""
        if self.cluster_centers_ is None:
            raise RuntimeError("Call fit before predict")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number of features used in fit")
        return np.argmin(self._squared_distances(X, self.cluster_centers_), axis=1)

    def fit_predict(self, X):
        """Fit the model and return training cluster labels."""
        return self.fit(X).labels_.copy()

    def transform(self, X):
        """Return Euclidean distances from samples to every learned center."""
        if self.cluster_centers_ is None:
            raise RuntimeError("Call fit before transform")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number of features used in fit")
        return np.sqrt(self._squared_distances(X, self.cluster_centers_))
