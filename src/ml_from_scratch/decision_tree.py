"""Decision-tree building blocks implemented with NumPy."""

import numpy as np


def gini_impurity(y):
    """Return the Gini impurity of a one-dimensional class-label array.

    The result is 0.0 when every sample belongs to the same class and is
    larger when the labels are more mixed.
    """
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("y must be a 1D array")
    if labels.size == 0:
        raise ValueError("y cannot be empty")

    if np.issubdtype(labels.dtype, np.number):
        if not np.all(np.isfinite(labels)):
            raise ValueError("y must not contain non-finite class labels")
    else:
        for label in labels:
            if label is None or (
                isinstance(label, (float, np.floating, complex, np.complexfloating))
                and not np.isfinite(label)
            ):
                raise ValueError("y must not contain missing or non-finite class labels")

    try:
        _, counts = np.unique(labels, return_counts=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("class labels must be mutually comparable") from exc

    probabilities = counts.astype(float) / labels.size
    return float(max(0.0, 1.0 - np.sum(probabilities ** 2)))


def best_split(X, y):
    """Find the best binary split using weighted child-node Gini impurity.

    Samples with a feature value less than or equal to the returned threshold
    go to the left child; larger values go to the right child. The return
    value is (feature_index, threshold, weighted_impurity). If no split
    strictly improves the parent impurity, return None.

    Ties are deterministic: the lowest feature index wins, followed by the
    lowest threshold for that feature.
    """
    raw_features = np.asarray(X)
    if np.iscomplexobj(raw_features):
        raise ValueError("X must contain real-valued features")
    try:
        features = np.asarray(X, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("X must contain numeric features") from exc

    if features.ndim != 2:
        raise ValueError("X must be a 2D array")
    if features.shape[0] < 2 or features.shape[1] == 0:
        raise ValueError("X must contain at least two samples and one feature")
    if not np.all(np.isfinite(features)):
        raise ValueError("X must contain only finite values")

    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("y must be a 1D array")
    if labels.size != features.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    parent_impurity = gini_impurity(labels)
    if parent_impurity == 0.0:
        return None

    best = None
    best_impurity = parent_impurity
    n_samples = labels.size

    for feature_index in range(features.shape[1]):
        values = features[:, feature_index]
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        sorted_labels = labels[order]

        # Only boundaries between distinct adjacent values can form a split.
        split_positions = np.flatnonzero(sorted_values[:-1] < sorted_values[1:]) + 1
        for split_position in split_positions:
            left_value = sorted_values[split_position - 1]
            right_value = sorted_values[split_position]

            # Halving before addition avoids overflow for opposite large values.
            threshold = left_value / 2.0 + right_value / 2.0
            if not (left_value <= threshold < right_value):
                threshold = left_value

            left_labels = sorted_labels[:split_position]
            right_labels = sorted_labels[split_position:]
            weighted_impurity = (
                left_labels.size * gini_impurity(left_labels)
                + right_labels.size * gini_impurity(right_labels)
            ) / n_samples

            if weighted_impurity < best_impurity:
                best_impurity = weighted_impurity
                best = (feature_index, float(threshold), float(weighted_impurity))

    return best
