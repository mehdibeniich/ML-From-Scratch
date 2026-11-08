# ML From Scratch

A hands-on machine learning project built from scratch with Python and NumPy.

## Goals

Learn the mathematics behind machine learning by implementing algorithms step by step.

## Roadmap

- [x] Linear regression with gradient descent
- [ ] Logistic regression
- [ ] Data preprocessing and feature scaling
- [x] Regression evaluation metrics (MSE, MAE, and R²)
- [ ] K-nearest neighbors
- [ ] Decision trees

## Implemented

### Linear regression

A linear model trained with batch gradient descent. See **examples/linear_regression_demo.py** for a runnable example.

### Logistic regression building blocks

The **ml_from_scratch.logistic_regression** module currently provides:

- **sigmoid(z):** numerically stable conversion of logits to probabilities.
- **binary_cross_entropy_with_logits(y_true, logits, reduction="mean"):** numerically stable binary cross-entropy without explicitly calculating probabilities first. The reduction argument supports "none", "sum", and "mean".

Example:

~~~python
import numpy as np
from ml_from_scratch import binary_cross_entropy_with_logits, sigmoid

logits = np.array([-2.0, 0.0, 2.0])
probabilities = sigmoid(logits)
targets = np.array([0.0, 1.0, 1.0])
loss = binary_cross_entropy_with_logits(targets, logits)
print(probabilities)
print(loss)
~~~

### Regression evaluation metrics

The **ml_from_scratch.metrics** module provides:

- **Mean Squared Error (MSE):** average of squared prediction errors.
- **Mean Absolute Error (MAE):** average of absolute prediction errors.
- **R² score:** coefficient of determination; it can be negative when predictions are worse than the mean baseline.

All three functions require non-empty, equally sized, finite target and prediction values. For a constant target, **r2_score** returns 1.0 for perfect predictions and 0.0 otherwise.

Example:

~~~python
from ml_from_scratch.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0, 2, 8]

print(mean_squared_error(y_true, y_pred))
print(mean_absolute_error(y_true, y_pred))
print(r2_score(y_true, y_pred))
~~~

## Setup

~~~bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
~~~

## Run tests

From the repository root:

~~~bash
PYTHONPATH=src python -m unittest discover -s tests -v
~~~

On Windows PowerShell, set the source path first:

~~~powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
~~~

## Structure

~~~text
src/ml_from_scratch/  # implementations
examples/             # runnable demos
tests/                # unit tests
~~~

Each algorithm includes an explanation, implementation, example, and tests.
