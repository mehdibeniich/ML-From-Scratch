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


def best_split(X, y, feature_indices=None):
    """Find the best binary split using weighted child-node Gini impurity.

    Samples with a feature value less than or equal to the returned threshold
    go to the left child; larger values go to the right child. The return
    value is (feature_index, threshold, weighted_impurity). If no split
    strictly improves the parent impurity, return None.

    Ties are deterministic: the lowest feature index wins, followed by the
    lowest threshold for that feature. Each feature is sorted once, then
    class counts are updated incrementally while candidate boundaries are
    evaluated.
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

    if feature_indices is None:
        selected_features = np.arange(features.shape[1])
    else:
        selected_features = np.asarray(feature_indices)
        if (
            selected_features.ndim != 1
            or selected_features.size == 0
            or not np.issubdtype(selected_features.dtype, np.integer)
            or np.any(selected_features < 0)
            or np.any(selected_features >= features.shape[1])
            or np.unique(selected_features).size != selected_features.size
        ):
            raise ValueError("feature_indices must be unique valid feature indices")

    # Preserve the documented lowest-feature-index tie-break even when the
    # caller supplies sampled feature indices in random order.
    selected_features = np.sort(selected_features)

    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("y must be a 1D array")
    if labels.size != features.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    parent_impurity = gini_impurity(labels)
    if parent_impurity == 0.0:
        return None

    _, label_codes = np.unique(labels, return_inverse=True)
    n_classes = int(label_codes.max()) + 1
    best = None
    best_impurity = parent_impurity
    n_samples = labels.size

    for feature_index in selected_features:
        values = features[:, feature_index]
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        sorted_label_codes = label_codes[order]

        left_counts = np.zeros(n_classes, dtype=np.int64)
        right_counts = np.bincount(sorted_label_codes, minlength=n_classes)
        left_square_sum = 0
        right_square_sum = sum(int(count) ** 2 for count in right_counts)

        # Move one sample at a time; evaluate only boundaries between distinct
        # feature values so neither child can be empty.
        for split_position in range(1, n_samples):
            class_index = sorted_label_codes[split_position - 1]
            left_count = int(left_counts[class_index])
            right_count = int(right_counts[class_index])
            left_square_sum += 2 * left_count + 1
            right_square_sum += 1 - 2 * right_count
            left_counts[class_index] = left_count + 1
            right_counts[class_index] = right_count - 1

            left_value = sorted_values[split_position - 1]
            right_value = sorted_values[split_position]
            if not left_value < right_value:
                continue

            left_size = split_position
            right_size = n_samples - split_position
            left_impurity = max(
                0.0, 1.0 - left_square_sum / (left_size * left_size)
            )
            right_impurity = max(
                0.0, 1.0 - right_square_sum / (right_size * right_size)
            )
            weighted_impurity = (
                left_size * left_impurity + right_size * right_impurity
            ) / n_samples

            # Halving before addition avoids overflow for opposite large values.
            threshold = left_value / 2.0 + right_value / 2.0
            if not (left_value <= threshold < right_value):
                threshold = left_value

            if weighted_impurity < best_impurity:
                best_impurity = weighted_impurity
                best = (feature_index, float(threshold), float(weighted_impurity))

    return best
