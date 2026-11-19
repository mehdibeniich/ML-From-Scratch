"""Machine-learning algorithms implemented from scratch."""

from .decision_tree import best_split, gini_impurity
from .decision_tree_classifier import DecisionTreeClassifier
from .knn import KNeighborsClassifier
from .linear_regression import LinearRegression
from .logistic_regression import (
    LogisticRegression,
    binary_cross_entropy_with_logits,
    sigmoid,
)
from .metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from .standard_scaler import StandardScaler

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "StandardScaler",
    "sigmoid",
    "binary_cross_entropy_with_logits",
    "accuracy_score",
    "confusion_matrix",
    "mean_absolute_error",
    "mean_squared_error",
    "r2_score",
    "gini_impurity",
    "best_split",
]
