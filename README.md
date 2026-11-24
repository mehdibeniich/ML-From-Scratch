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
- [x] Decision trees
- [x] Random Forest classification
- [x] Gaussian Naive Bayes

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

### Decision trees

**DecisionTreeClassifier** builds a recursive binary classification tree using Gini impurity.

- **fit(X, y):** validates the training data and recursively builds the tree.
- **predict(X):** traverses the fitted tree and returns one class label per sample. A single 1D sample is also accepted.
- **predict_proba(X):** returns the class frequencies in each reached leaf, with columns ordered by sorted `classes_`. These empirical frequencies are not probability-calibrated.
- **feature_importances_:** normalized, sample-weighted reductions in Gini impurity attributed to each feature across the tree. Values sum to 1 when the tree makes an impurity-reducing split; an unsplit tree has all-zero importances.
- **max_depth:** optionally limits the number of splits from the root. The default, `None`, grows until nodes are pure or no improving split remains.
- **min_samples_split:** minimum number of samples required before a node is considered for splitting (default: 2).
- **classes_** contains the sorted class labels; majority-class ties are resolved by that order.

The lower-level **gini_impurity(y)** helper measures how mixed the class labels are in a node. The **best_split(X, y)** helper returns the feature index, threshold, and weighted child-node Gini impurity for the best split. Samples at or below the threshold go left; larger values go right. It returns **None** when no split improves the parent node.

Example:

~~~python
from ml_from_scratch import DecisionTreeClassifier

X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
y = ["low", "low", "middle", "middle", "high", "high"]

model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)
print(model.predict_proba([[0.5], [2.5], [4.5]]))
print(model.predict([[0.5], [2.5], [4.5]]))
~~~

### Gaussian Naive Bayes

**GaussianNB** is a multiclass classifier that models each numeric feature
with a Gaussian distribution conditioned on the class. It combines class
priors with feature log-likelihoods and normalizes the scores to produce
stable posterior probabilities.

- **fit(X, y):** estimates class counts, class priors, feature means, and
  class-feature variances.
- **predict(X):** returns the most likely class for each sample.
- **predict_proba(X):** returns normalized class probabilities, with columns
  ordered by `classes_`.
- **var_smoothing:** adds a global variance-scaled floor to prevent zero
  variance for constant features (default: `1e-9`).
- **classes_**, **class_count_**, **class_prior_**, **theta_**, and **var_**
  expose the learned class order and statistics.

The implementation evaluates Gaussian likelihoods in log space and uses
stable probability normalization. It supports numeric and string labels,
rejects missing labels and invalid feature arrays, and preserves an existing
fitted model if a refit fails.

Example:

~~~python
from ml_from_scratch import GaussianNB

X_train = [[-2.0], [-1.5], [-1.0], [1.0], [1.5], [2.0]]
y_train = ["left", "left", "left", "right", "right", "right"]

model = GaussianNB(var_smoothing=1e-9).fit(X_train, y_train)
print(model.predict([[1.8]]))
print(model.predict_proba([[1.8]]))
~~~

Run the complete multiclass example with
`PYTHONPATH=src python examples/gaussian_naive_bayes_demo.py`.
See [the Gaussian Naive Bayes guide](docs/gaussian_naive_bayes.md) for the
math, API, smoothing behavior, and input validation details.

### Randomized decision-tree splits

DecisionTreeClassifier also supports random candidate-feature selection at each
node, which is the tree-level building block for ensemble methods.

- max_features: None (all features), "sqrt", "log2", a positive integer, or
  a fraction in (0, 1]. Fractions select floor(fraction * n_features), with
  at least one feature.
- random_state: a non-negative integer seed or None for reproducible split
  feature selection.

Use max_features="sqrt" with a fixed random_state to experiment with feature
randomization. The default None preserves deterministic full-feature splitting.

### Random Forest

**RandomForestClassifier** combines randomized decision trees trained on
bootstrap samples. It averages tree probabilities in a consistent global
class order, including when a bootstrap sample omits a rare class.

- **n_estimators:** number of trees (default: 100).
- **max_features:** candidate features per split; defaults to \`"sqrt"\`.
- **bootstrap:** sample training rows with replacement (default: true).
- **random_state:** reproduces bootstrap samples and tree feature selection.
- **predict_proba(X):** averages each tree's probabilities after aligning
  columns to the forest's \`classes_\`.
- **feature_importances_:** normalized aggregate Gini impurity reductions.

Run \`PYTHONPATH=src python examples/random_forest_demo.py\` for a runnable
multiclass example. See [the Random Forest guide](docs/random_forest_classifier.md)
for the algorithm, API, bootstrap behavior, and edge cases.

### Classification evaluation metrics

**accuracy_score(y_true, y_pred)** returns the fraction of predicted class labels that match the true labels. It supports numeric and string labels, including multiclass targets, and requires non-empty, equally sized 1D arrays without missing or non-finite labels.

**confusion_matrix(y_true, y_pred, labels=None)** returns an integer matrix with true classes on rows and predicted classes on columns. By default, comparable labels are sorted; mixed, non-comparable label types retain first-occurrence order. Pass `labels` to choose an explicit order or include classes with zero observations. Every class present in the inputs must appear in an explicit `labels` sequence.

**precision_score**, **recall_score**, and **f1_score** support binary, micro, macro, weighted, and per-class (`average=None`) output. Binary scoring uses `pos_label` (default `1`); multiclass scoring should select an averaging strategy. `zero_division` controls the value used for a zero denominator. See [the classification metrics guide](docs/classification_metrics.md) and run `PYTHONPATH=src python examples/classification_metrics_demo.py` for a complete multiclass example.

Example:

~~~python
from ml_from_scratch import accuracy_score, confusion_matrix

y_true = ["cat", "dog", "bird"]
y_pred = ["cat", "bird", "bird"]

print(accuracy_score(y_true, y_pred))  # 0.666...
print(confusion_matrix(y_true, y_pred, labels=["bird", "cat", "dog"]))
~~~

### Regression evaluation metrics

The **ml_from_scratch.metrics** module provides:

- **Mean Squared Error (MSE):** average of squared prediction errors.
- **Mean Absolute Error (MAE):** average of absolute prediction errors.
- **R² score:** coefficient of determination; it can be negative when predictions are worse than the mean baseline.

All three functions require non-empty, equally sized, finite target and prediction values. They raise `FloatingPointError` if a metric calculation overflows. **r2_score** additionally requires at least two samples. For a constant target, **r2_score** returns 1.0 for perfect predictions and 0.0 otherwise.

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
