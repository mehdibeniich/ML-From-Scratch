"""Regression metrics implemented with NumPy."""

import numpy as np


def _validate_regression_inputs(y_true, y_pred):
    """Convert targets and predictions to finite, equally sized 1D arrays."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true.size == 0:
        raise ValueError("y_true and y_pred cannot be empty")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must contain the same number of values")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("y_true and y_pred must contain only finite values")

    return y_true, y_pred


def mean_squared_error(y_true, y_pred):
    """Return the average squared difference between targets and predictions."""
    y_true, y_pred = _validate_regression_inputs(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true, y_pred):
    """Return the average absolute difference between targets and predictions."""
    y_true, y_pred = _validate_regression_inputs(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred):
    """Return the coefficient of determination.

    For a constant target, return 1.0 for perfect predictions and 0.0 otherwise.
    """
    y_true, y_pred = _validate_regression_inputs(y_true, y_pred)

    residual_sum_squares = float(np.sum((y_true - y_pred) ** 2))
    total_sum_squares = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if total_sum_squares == 0.0:
        return 1.0 if residual_sum_squares == 0.0 else 0.0

    return float(1.0 - residual_sum_squares / total_sum_squares)
