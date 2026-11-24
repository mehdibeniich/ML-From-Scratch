# Gaussian Naive Bayes

The `GaussianNB` classifier implements Gaussian Naive Bayes for numeric feature
arrays using NumPy. It assumes each feature follows a Gaussian distribution
within each class and treats features as conditionally independent given the
class.

For class \(c\) and feature \(j\), it estimates the mean and variance from
the training samples belonging to that class. Predictions use the joint
log-likelihood:

\[
\log P(c) - \frac{1}{2}\sum_j
\left(\log(2\pi\sigma_{cj}^2)
+ \frac{(x_j-\mu_{cj})^2}{\sigma_{cj}^2}\right)
\]

The largest score determines the predicted class. Probability output is
normalized with a log-sum-exp style shift. The log-probability API retains
small posterior values in log space instead of losing them to exponentiation
underflow.

## API

```python
from ml_from_scratch import GaussianNB

model = GaussianNB(var_smoothing=1e-9)
model.fit(X_train, y_train)
labels = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

- `fit(X, y)` learns class counts, priors, feature means, and variances.
- `predict(X)` returns one predicted label per sample.
- `predict_proba(X)` returns probabilities with columns in `classes_` order.
- `predict_log_proba(X)` returns normalized log probabilities in the same class order. It preserves very small class evidence even when exponentiating it in `predict_proba` underflows to zero.
- `classes_` contains sorted labels when they can be compared; otherwise
  labels retain first-occurrence order.
- `class_count_`, `class_prior_`, `theta_`, and `var_` expose learned
  statistics.
- `n_features_in_` records the feature count observed during fitting.

## Variance smoothing

`var_smoothing` must be a finite, non-negative number. The classifier adds a
small floor based on the largest global feature variance to each class-feature
variance. This prevents zero variance for constant features and helps avoid
division by zero. A tiny positive floating-point floor remains even when
smoothing is set to zero.

Scaling is not required by the Gaussian likelihood itself, but feature units
can still affect numerical behavior. Fit any preprocessing only on training
data and reuse it for evaluation data to avoid leakage.

## Input and failure behavior

- `X` must be a finite, non-empty 2D numeric array.
- `y` must be a 1D array with matching sample count and at least two classes.
- Missing or non-finite numeric labels are rejected.
- Prediction accepts either a 2D batch or one 1D feature vector.
- Prediction before fitting and mismatched feature counts raise clear errors.
- If fitting fails, a previously fitted model is left unchanged.
- If extreme inputs produce invalid log probabilities, prediction raises
  `FloatingPointError` rather than returning invalid probabilities.

Run the example from the repository root:

```bash
PYTHONPATH=src python examples/gaussian_naive_bayes_demo.py
```
