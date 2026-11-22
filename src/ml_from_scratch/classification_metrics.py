"""Precision, recall, and F1 metrics for classification."""

import numpy as np

from .metrics import confusion_matrix, _find_label_index, _validate_classification_inputs


def _class_labels(y_true, y_pred, labels):
    if labels is not None:
        return list(np.asarray(labels, dtype=object))
    result = []
    for value in np.concatenate((y_true, y_pred)):
        if _find_label_index(value, result) == -1:
            result.append(value)
    try:
        result = sorted(result)
    except TypeError:
        pass
    return result


def _safe_divide(numerator, denominator, zero_division):
    if denominator == 0:
        return float(zero_division)
    return float(numerator / denominator)


def _per_class_scores(y_true, y_pred, labels, zero_division):
    if zero_division not in (0, 1, 0.0, 1.0):
        raise ValueError("zero_division must be 0 or 1")

    y_true, y_pred = _validate_classification_inputs(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    class_labels = _class_labels(y_true, y_pred, labels)

    true_positive = np.diag(matrix).astype(float)
    predicted_count = matrix.sum(axis=0).astype(float)
    support = matrix.sum(axis=1).astype(float)
    precision = np.array([
        _safe_divide(tp, predicted, zero_division)
        for tp, predicted in zip(true_positive, predicted_count)
    ])
    recall = np.array([
        _safe_divide(tp, actual, zero_division)
        for tp, actual in zip(true_positive, support)
    ])
    f1 = np.array([
        _safe_divide(2.0 * p * r, p + r, zero_division)
        for p, r in zip(precision, recall)
    ])
    return class_labels, matrix, precision, recall, f1, support


def _classification_score(
    metric, y_true, y_pred, average="binary", pos_label=1,
    labels=None, zero_division=0,
):
    class_labels, matrix, precision, recall, f1, support = _per_class_scores(
        y_true, y_pred, labels, zero_division
    )
    scores = {"precision": precision, "recall": recall, "f1": f1}[metric]

    if average is None:
        return scores
    if average == "binary":
        active = np.flatnonzero((matrix.sum(axis=0) + matrix.sum(axis=1)) > 0)
        if len(active) != 2:
            raise ValueError("average='binary' requires exactly two observed classes")
        positive_index = _find_label_index(pos_label, class_labels)
        if positive_index == -1:
            raise ValueError("pos_label must be present in labels")
        return float(scores[positive_index])
    if average == "micro":
        tp = float(np.trace(matrix))
        fp = float(matrix.sum(axis=0).sum() - tp)
        fn = float(matrix.sum(axis=1).sum() - tp)
        if metric == "precision":
            return _safe_divide(tp, tp + fp, zero_division)
        if metric == "recall":
            return _safe_divide(tp, tp + fn, zero_division)
        p = _safe_divide(tp, tp + fp, zero_division)
        r = _safe_divide(tp, tp + fn, zero_division)
        return _safe_divide(2.0 * p * r, p + r, zero_division)
    if average == "macro":
        return float(np.mean(scores))
    if average == "weighted":
        total = float(np.sum(support))
        if total == 0:
            return float(zero_division)
        return float(np.dot(scores, support) / total)
    raise ValueError("average must be 'binary', 'micro', 'macro', 'weighted', or None")


def precision_score(
    y_true, y_pred, average="binary", pos_label=1, labels=None, zero_division=0
):
    """Compute precision; supports binary, micro, macro, weighted, or per-class output."""
    return _classification_score(
        "precision", y_true, y_pred, average, pos_label, labels, zero_division
    )


def recall_score(
    y_true, y_pred, average="binary", pos_label=1, labels=None, zero_division=0
):
    """Compute recall; supports binary, micro, macro, weighted, or per-class output."""
    return _classification_score(
        "recall", y_true, y_pred, average, pos_label, labels, zero_division
    )


def f1_score(
    y_true, y_pred, average="binary", pos_label=1, labels=None, zero_division=0
):
    """Compute F1; supports binary, micro, macro, weighted, or per-class output."""
    return _classification_score(
        "f1", y_true, y_pred, average, pos_label, labels, zero_division
    )
