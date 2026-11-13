"""K-nearest-neighbors classification implemented with NumPy."""

import numpy as np


class KNeighborsClassifier:
    """Classify samples by majority vote among their nearest training samples.

    Euclidean distance is used. If classes tie on vote count, the class of the
    closest tied neighbor wins. predict_proba columns follow classes_.
    """

    def __init__(self, n_neighbors: int = 5):
        if (
            isinstance(n_neighbors, (bool, np.bool_))
            or not isinstance(n_neighbors, (int, np.integer))
            or n_neighbors < 1
        ):
            raise ValueError("n_neighbors must be a positive integer")
        self.n_neighbors = int(n_neighbors)
        self._X_train = None
        self._y_train = None
        self.classes_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        """Store a validated training set and return this classifier."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("training data must contain samples and features")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        if np.any(y != y) or any(label is None for label in y):
            raise ValueError("y must not contain missing class labels")
        if np.issubdtype(y.dtype, np.number) and not np.all(np.isfinite(y)):
            raise ValueError("numeric class labels must be finite")
        if self.n_neighbors > X.shape[0]:
            raise ValueError("n_neighbors cannot exceed the number of training samples")

        try:
            classes = np.unique(y)
        except TypeError as exc:
            raise ValueError("class labels must be mutually comparable") from exc

        # Assign only after validation so a failed refit preserves the model.
        self._X_train = X.copy()
        self._y_train = y.copy()
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        return self

    def _validated_features(self, X):
        if self._X_train is None:
            raise RuntimeError("Call fit before prediction")

        X = np.asarray(X, dtype=float)
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

    def _get_neighbor_labels(self, X):
        X = self._validated_features(X)
        neighbors = []
        for sample in X:
            with np.errstate(over="ignore", invalid="ignore"):
                differences = self._X_train - sample
                squared_distances = np.sum(differences * differences, axis=1)
            if not np.all(np.isfinite(squared_distances)):
                raise FloatingPointError("Euclidean distances overflowed")
            nearest_indices = np.argsort(squared_distances, kind="stable")[
                : self.n_neighbors
            ]
            neighbors.append(self._y_train[nearest_indices])
        return neighbors

    def predict_proba(self, X):
        """Return neighbor-vote proportions in classes_ column order."""
        neighbor_labels = self._get_neighbor_labels(X)
        probabilities = np.zeros(
            (len(neighbor_labels), self.classes_.size), dtype=float
        )
        for row, labels in enumerate(neighbor_labels):
            votes = np.array(
                [np.count_nonzero(labels == label) for label in self.classes_],
                dtype=float,
            )
            probabilities[row] = votes / self.n_neighbors
        return probabilities

    def predict(self, X):
        """Predict labels; ties go to the closest class among tied classes."""
        neighbor_labels = self._get_neighbor_labels(X)
        predictions = np.empty(len(neighbor_labels), dtype=self.classes_.dtype)
        for row, labels in enumerate(neighbor_labels):
            votes = np.array(
                [np.count_nonzero(labels == label) for label in self.classes_],
                dtype=int,
            )
            tied_classes = self.classes_[votes == votes.max()]
            for label in labels:
                if np.any(tied_classes == label):
                    predictions[row] = label
                    break
        return predictions
