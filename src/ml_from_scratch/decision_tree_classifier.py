"""Recursive decision-tree classification implemented with NumPy."""

import numpy as np

from .decision_tree import best_split, gini_impurity


class DecisionTreeClassifier:
    """A recursive binary classifier that selects splits by Gini impurity.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum number of splits from the root. ``None`` grows the tree until
        nodes are pure or no improving split remains. A value of 0 creates a
        single majority-class leaf.
    min_samples_split : int, default=2
        Minimum number of samples a node must contain before it is considered
        for splitting.
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        if max_depth is not None and (
            isinstance(max_depth, (bool, np.bool_))
            or not isinstance(max_depth, (int, np.integer))
            or max_depth < 0
        ):
            raise ValueError("max_depth must be a non-negative integer or None")
        if (
            isinstance(min_samples_split, (bool, np.bool_))
            or not isinstance(min_samples_split, (int, np.integer))
            or min_samples_split < 2
        ):
            raise ValueError("min_samples_split must be an integer of at least 2")

        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.tree_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        """Build the tree from a finite 2D feature array and 1D labels."""
        raw_features = np.asarray(X)
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

        labels = np.asarray(y)
        if labels.ndim != 1:
            raise ValueError("y must be a 1D array")
        if labels.size != features.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        gini_impurity(labels)
        classes = np.unique(labels)
        feature_importances = np.zeros(features.shape[1], dtype=float)
        n_training_samples = features.shape[0]

        def make_leaf(sample_indices):
            node_labels = labels[sample_indices]
            node_classes, counts = np.unique(node_labels, return_counts=True)
            prediction = node_classes[int(np.argmax(counts))]
            probabilities = np.zeros(classes.size, dtype=float)
            probabilities[np.searchsorted(classes, node_classes)] = (
                counts / sample_indices.size
            )
            return {
                "prediction": prediction,
                "probabilities": probabilities,
            }

        def build_node(sample_indices, depth):
            leaf = make_leaf(sample_indices)
            node_classes = np.unique(labels[sample_indices])

            if (
                node_classes.size == 1
                or sample_indices.size < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)
            ):
                return leaf

            split = best_split(features[sample_indices], labels[sample_indices])
            if split is None:
                return leaf

            feature_index, threshold, weighted_child_impurity = split
            left_mask = features[sample_indices, feature_index] <= threshold
            if not np.any(left_mask) or np.all(left_mask):
                return leaf

            node_impurity = gini_impurity(labels[sample_indices])
            impurity_decrease = max(0.0, node_impurity - weighted_child_impurity)
            feature_importances[feature_index] += (
                sample_indices.size / n_training_samples
            ) * impurity_decrease

            return {
                "feature_index": feature_index,
                "threshold": threshold,
                "left": build_node(sample_indices[left_mask], depth + 1),
                "right": build_node(sample_indices[~left_mask], depth + 1),
            }

        # Assign only after the entire tree has been built successfully.
        tree = build_node(np.arange(features.shape[0]), 0)
        total_importance = feature_importances.sum()
        if total_importance > 0.0:
            feature_importances /= total_importance

        self.tree_ = tree
        self.classes_ = classes
        self.n_features_in_ = features.shape[1]
        self.feature_importances_ = feature_importances
        return self

    def _validated_features(self, X):
        if self.tree_ is None:
            raise RuntimeError("Call fit before prediction")

        raw_features = np.asarray(X)
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
        """Return leaf class frequencies with columns ordered by ``classes_``."""
        features = self._validated_features(X)
        probabilities = []

        for sample in features:
            node = self.tree_
            while "prediction" not in node:
                if sample[node["feature_index"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]
            probabilities.append(node["probabilities"])

        return np.asarray(probabilities, dtype=float)

    def predict(self, X):
        """Predict one class label for each input sample."""
        features = self._validated_features(X)
        predictions = []

        for sample in features:
            node = self.tree_
            while "prediction" not in node:
                if sample[node["feature_index"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]
            predictions.append(node["prediction"])

        return np.asarray(predictions, dtype=self.classes_.dtype)
