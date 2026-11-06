"""Machine-learning algorithms implemented from scratch."""

from .linear_regression import LinearRegression
from .metrics import mean_absolute_error, mean_squared_error, r2_score

__all__ = [
    "LinearRegression",
    "mean_absolute_error",
    "mean_squared_error",
    "r2_score",
]
