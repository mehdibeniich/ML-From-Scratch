"""Logistic-regression utilities implemented with NumPy."""

import numpy as np


def sigmoid(z):
    """Return the logistic sigmoid of a scalar or array, stably.

    The log-add-exp formulation avoids overflow for very large positive or
    negative inputs and preserves the shape of array inputs.
    """
    values = np.asarray(z, dtype=float)
    result = np.exp(-np.logaddexp(0.0, -values))
    return float(result) if result.ndim == 0 else result


def binary_cross_entropy_with_logits(y_true, logits, reduction="mean"):
    """Compute binary cross-entropy directly from logits.

    Args:
        y_true: Targets in [0, 1], with the same shape as logits. Both hard
            labels (0 or 1) and soft labels (probabilities) are supported.
        logits: Raw, unscaled model scores.
        reduction: "none" for element-wise losses, "sum" for their sum,
            or "mean" for their average.

    Computing logaddexp(0, logits) - y_true * logits avoids first converting
    logits to probabilities, which is more stable for extreme values.

    Raises:
        ValueError: If inputs are empty, have different shapes, contain
            non-finite values, targets are outside [0, 1], or reduction is
            unsupported.
    """
    targets = np.asarray(y_true, dtype=float)
    scores = np.asarray(logits, dtype=float)

    if targets.size == 0 or scores.size == 0:
        raise ValueError("y_true and logits must be non-empty")
    if targets.shape != scores.shape:
        raise ValueError("y_true and logits must have the same shape")
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(scores)):
        raise ValueError("y_true and logits must contain only finite values")
    if np.any((targets < 0.0) | (targets > 1.0)):
        raise ValueError("y_true values must be between 0 and 1")
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")

    losses = np.logaddexp(0.0, scores) - targets * scores
    if reduction == "none":
        return float(losses) if losses.ndim == 0 else losses
    if reduction == "sum":
        return float(np.sum(losses))
    return float(np.mean(losses))
