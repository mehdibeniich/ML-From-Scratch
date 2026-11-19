"""Regression and classification metrics implemented with NumPy."""

import numpy as np


def _validate_classification_inputs(y_true, y_pred):
    """Validate equally sized 1D arrays of non-missing scalar class labels."""
    try:
        y_true = np.asarray(y_true, dtype=object)
        y_pred = np.asarray(y_pred, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true and y_pred must be 1D label arrays") from exc

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D label arrays")
    if y_true.size == 0:
        raise ValueError("y_true and y_pred cannot be empty")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must contain the same number of labels")

    for labels in (y_true, y_pred):
        for label in labels:
            if label is None:
                raise ValueError("class labels cannot contain missing values")
            try:
                finite = np.isfinite(label)
            except TypeError:
                continue
            if np.ndim(finite) == 0 and not bool(finite):
                raise ValueError("class labels must be finite")

    return y_true, y_pred


def accuracy_score(y_true, y_pred):
    """Return the fraction of class labels predicted correctly."""
    y_true, y_pred = _validate_classification_inputs(y_true, y_pred)
    try:
        matches = np.equal(y_true, y_pred)
    except (TypeError, ValueError) as exc:
        raise ValueError("class labels must support equality comparison") from exc
    return float(np.count_nonzero(matches) / y_true.size)


def _labels_equal(left, right):
    """Compare two scalar class labels without coercing their types."""
    try:
        result = left == right
    except (TypeError, ValueError) as exc:
        raise ValueError("class labels must support equality comparison") from exc
    if np.ndim(result) != 0:
        raise ValueError("class labels must be scalar values")
    return bool(result)


def _find_label_index(value, labels):
    """Return the index of a matching label, or -1 when it is absent."""
    for index, label in enumerate(labels):
        if _labels_equal(value, label):
            return index
    return -1


def confusion_matrix(y_true, y_pred, labels=None):
    """Count true/predicted class pairs, with true classes on rows.

    If labels is omitted, labels are sorted when mutually comparable;
    otherwise their first-occurrence order across y_true then y_pred is used.
    An explicit labels sequence fixes both matrix dimensions and their order,
    and must include every observed class.
    """
    y_true, y_pred = _validate_classification_inputs(y_true, y_pred)

    if labels is None:
        class_labels = []
        for value in np.concatenate((y_true, y_pred)):
            if _find_label_index(value, class_labels) == -1:
                class_labels.append(value)
        try:
            class_labels = sorted(class_labels)
        except TypeError:
            pass
    else:
        class_labels, _ = _validate_classification_inputs(labels, labels)
        for index, label in enumerate(class_labels):
            if _find_label_index(label, class_labels[:index]) != -1:
                raise ValueError("labels must not contain duplicates")

    for value in np.concatenate((y_true, y_pred)):
        if _find_label_index(value, class_labels) == -1:
            raise ValueError("labels must include every class in y_true and y_pred")

    matrix = np.zeros((len(class_labels), len(class_labels)), dtype=np.int64)
    for true_label, predicted_label in zip(y_true, y_pred):
        row = _find_label_index(true_label, class_labels)
        column = _find_label_index(predicted_label, class_labels)
        matrix[row, column] += 1

    return matrix


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


def _finite_metric_result(value, metric_name):
    """Return a finite metric value or explain that its calculation overflowed."""
    if not np.isfinite(value):
        raise FloatingPointError(
            f"{metric_name} overflowed: non-finite result encountered"
        )
    return float(value)


def mean_squared_error(y_true, y_pred):
    """Return the average squared difference between targets and predictions."""
    y_true, y_pred = _validate_regression_inputs(y_true, y_pred)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.mean((y_true - y_pred) ** 2)
    return _finite_metric_result(result, "mean squared error")


def mean_absolute_error(y_true, y_pred):
    """Return the average absolute difference between targets and predictions."""
    y_true, y_pred = _validate_regression_inputs(y_true, y_pred)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.mean(np.abs(y_true - y_pred))
    return _finite_metric_result(result, "mean absolute error")


def r2_score(y_true, y_pred):
    """Return the coefficient of determination.

    Requires at least two samples. For a constant target, return 1.0 for
    perfect predictions and 0.0 otherwise.
    """
    y_true, y_pred = _validate_regression_inputs(y_true, y_pred)
    if y_true.size < 2:
        raise ValueError("r2_score requires at least two samples")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        residual_sum_squares = np.sum((y_true - y_pred) ** 2)
        total_sum_squares = np.sum((y_true - np.mean(y_true)) ** 2)

        if not np.isfinite(residual_sum_squares) or not np.isfinite(total_sum_squares):
            raise FloatingPointError(
                "R2 score overflowed: non-finite intermediate result encountered"
            )

        if total_sum_squares == 0.0:
            return 1.0 if residual_sum_squares == 0.0 else 0.0

        result = 1.0 - residual_sum_squares / total_sum_squares

    return _finite_metric_result(result, "R2 score")
