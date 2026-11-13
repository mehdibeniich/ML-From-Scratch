# ML From Scratch

A hands-on machine learning project built from scratch with Python and NumPy.

## Goals

Learn the mathematics behind machine learning by implementing algorithms step by step.

## Roadmap

- [x] Linear regression with gradient descent
- [x] Logistic regression
- [x] Data preprocessing and feature scaling
- [x] Regression evaluation metrics (MSE, MAE, and R²)
- [x] K-nearest neighbors
- [ ] Decision trees

## Implemented

### Linear regression

A linear model trained with batch gradient descent. See **examples/linear_regression_demo.py** for a runnable example. If optimization diverges, `fit()` raises `FloatingPointError` and preserves any previously fitted model.

### Logistic regression

**LogisticRegression** is a binary classifier trained with batch gradient descent. It uses the stable sigmoid and binary cross-entropy utilities in **ml_from_scratch.logistic_regression**.

- **fit(X, y):** learns weights and bias from a 2D feature array and binary labels (0 and 1). If optimization diverges, it raises `FloatingPointError` without replacing a previously fitted model.
- **predict_proba(X):** returns two columns containing probabilities for classes 0 and 1.
- **predict(X):** returns class labels using a 0.5 probability threshold.
- **cost_history:** records the training loss after each iteration.

Example:

~~~python
import numpy as np
from ml_from_scratch import LogisticRegression

X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
y = np.array([0, 0, 1, 1])

model = LogisticRegression(learning_rate=0.1, n_iterations=1_000)
model.fit(X, y)

print(model.predict_proba([[1.5]]))
print(model.predict([[1.5]]))
~~~

### Logistic regression building blocks

The **ml_from_scratch.logistic_regression** module also provides:

- **sigmoid(z):** numerically stable conversion of logits to probabilities.
- **binary_cross_entropy_with_logits(y_true, logits, reduction="mean"):** numerically stable binary cross-entropy without explicitly calculating probabilities first. The reduction argument supports "none", "sum", and "mean". Targets must be in [0, 1], so both hard labels (0/1) and soft labels (probabilities) are supported.

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

### Data preprocessing: StandardScaler

**StandardScaler** learns each feature's mean and population standard deviation from training data, then applies `(X - mean_) / scale_` to new samples.

- **fit(X):** learns the statistics from a non-empty, finite 2D training array.
- **transform(X):** scales new data using the learned training statistics.
- **fit_transform(X):** fits the scaler and transforms the same data.
- Constant features use a scale of 1.0, so they transform to zeros instead of dividing by zero.

Fit the scaler only on training data, then reuse it for validation or test data to avoid data leakage.

Example:

~~~python
import numpy as np
from ml_from_scratch import StandardScaler

X_train = np.array([[1.0, 100.0], [2.0, 150.0], [3.0, 200.0]])
X_test = np.array([[4.0, 250.0]])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled)
print(X_test_scaled)
~~~

### K-nearest neighbors

**KNeighborsClassifier** predicts a class by majority vote among the `n_neighbors` closest training samples, using Euclidean distance.

- **fit(X, y):** stores a finite 2D training feature array and its class labels.
- **predict(X):** returns the majority-vote class. If vote counts tie, the closest tied neighbor decides.
- **predict_proba(X):** returns vote proportions, with columns ordered by the sorted `classes_` attribute.
- **n_neighbors:** must be a positive integer no larger than the number of training samples.

Euclidean distance is sensitive to feature scale. Use **StandardScaler** when features have substantially different units or ranges.

Example:

~~~python
from ml_from_scratch import KNeighborsClassifier

X_train = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
y_train = ["low", "low", "low", "high", "high", "high"]

model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
print(model.predict([[0.5], [4.5]]))
print(model.predict_proba([[0.5], [4.5]]))
print(model.classes_)  # probability-column order
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
