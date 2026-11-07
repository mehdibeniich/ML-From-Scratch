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
