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
