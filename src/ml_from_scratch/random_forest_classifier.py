"""Random-forest classification built from decision trees and NumPy."""

import numpy as np

from .decision_tree import gini_impurity
from .decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier:
    """A bootstrap ensemble of randomized decision-tree classifiers.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of decision trees in the forest.
    max_depth : int or None, default=None
        Maximum depth passed to every decision tree.
    min_samples_split : int, default=2
        Minimum node size required for a split.
    max_features : None, str, int, or float, default="sqrt"
        Candidate features sampled at each tree node. Supports None, "sqrt",
        "log2", a positive integer, or a fraction in (0, 1].
    bootstrap : bool, default=True
        Whether each tree is trained on a sample of the training rows drawn
        with replacement. When False, every tree sees all training rows.
    random_state : int or None, default=None
        Seed controlling bootstrap samples and each tree's feature sampling.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=None,
    ):
        if (
            isinstance(n_estimators, (bool, np.bool_))
            or not isinstance(n_estimators, (int, np.integer))
            or n_estimators < 1
        ):
            raise ValueError("n_estimators must be a positive integer")
        if not isinstance(bootstrap, (bool, np.bool_)):
            raise ValueError("bootstrap must be a boolean")

        # Reuse the tree estimator's validation for tree-specific parameters.
        DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=random_state,
        )

        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.random_state = None if random_state is None else int(random_state)

        self.estimators_ = None
        self.estimators_samples_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        """Fit the forest and return self.

        All fitted attributes are replaced only after every tree has trained
        successfully, so a failed refit preserves the previous forest.
        """
        try:
            raw_features = np.asarray(X)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must be a rectangular numeric array") from exc
        if np.iscomplexobj(raw_features):
            raise ValueError("X must contain real-valued features")
        try:
            features = np.asarray(X, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("X must contain numeric features") from exc

        if features.ndim != 2:
            raise ValueError("X must be a 2D array")
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("training data must contain samples and features")
        if not np.all(np.isfinite(features)):
            raise ValueError("X must contain only finite values")

        try:
            labels = np.asarray(y)
        except (TypeError, ValueError) as exc:
            raise ValueError("y must be a one-dimensional label array") from exc
        if labels.ndim != 1:
            raise ValueError("y must be a 1D array")
        if labels.size != features.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        # Validate missing labels and comparability before starting the ensemble.
        gini_impurity(labels)
        try:
            classes = np.unique(labels)
        except (TypeError, ValueError) as exc:
            raise ValueError("class labels must be mutually comparable") from exc

        if (
            isinstance(self.max_features, (int, np.integer))
            and not isinstance(self.max_features, (bool, np.bool_))
            and self.max_features > features.shape[1]
        ):
            raise ValueError("max_features cannot exceed the number of input features")

        rng = np.random.default_rng(self.random_state)
        sample_indices_by_tree = []
        estimators = []

        for _ in range(self.n_estimators):
            if self.bootstrap:
                sample_indices = rng.integers(
                    0, features.shape[0], size=features.shape[0]
                )
            else:
                sample_indices = np.arange(features.shape[0], dtype=int)

            tree_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            estimator = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=tree_seed,
            )
            estimator.fit(features[sample_indices], labels[sample_indices])
            estimators.append(estimator)
            sample_indices_by_tree.append(sample_indices)

        feature_importances = np.mean(
            np.vstack([estimator.feature_importances_ for estimator in estimators]),
            axis=0,
        )
        importance_total = float(np.sum(feature_importances))
        if importance_total > 0.0:
            feature_importances /= importance_total

        # Commit fitted state only after all estimators and aggregate statistics
        # have been constructed successfully.
        self.estimators_ = estimators
        self.estimators_samples_ = sample_indices_by_tree
        self.classes_ = classes
        self.n_features_in_ = features.shape[1]
        self.feature_importances_ = feature_importances
        return self

    def _validated_features(self, X):
        if self.estimators_ is None:
            raise RuntimeError("Call fit before prediction")

        try:
            raw_features = np.asarray(X)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must be a rectangular numeric array") from exc
        if np.iscomplexobj(raw_features):
            raise ValueError("X must contain real-valued features")
        try:
            features = np.asarray(X, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("X must contain numeric features") from exc

        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError(
                "X must be 2D with the same number of features used in fit"
            )
        if features.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if not np.all(np.isfinite(features)):
            raise ValueError("X must contain only finite values")
        return features

    def predict_proba(self, X):
        """Average tree probabilities, aligned to the forest's classes_ order.

        A bootstrap sample may omit a rare class. Such a tree contributes no
        probability to that class, while the remaining trees retain the same
        global output-column order.
        """
        features = self._validated_features(X)
        probabilities = np.zeros(
            (features.shape[0], self.classes_.size), dtype=float
        )

        for estimator in self.estimators_:
            tree_probabilities = estimator.predict_proba(features)
            class_positions = np.searchsorted(self.classes_, estimator.classes_)
            probabilities[:, class_positions] += tree_probabilities

        probabilities /= len(self.estimators_)
        return probabilities

    def predict(self, X):
        """Predict the class with the highest mean tree probability."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
