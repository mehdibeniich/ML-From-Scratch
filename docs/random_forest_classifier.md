# Random Forest Classifier

The \`RandomForestClassifier\` combines many decision trees trained on
bootstrap samples. Each tree also samples a subset of candidate features at
each split. The combination reduces the variance of a single decision tree and
can model nonlinear decision boundaries without feature scaling.

## How it works

For each estimator:

1. Draw a training sample of the original size with replacement when
   \`bootstrap=True\`.
2. Grow a decision tree using Gini impurity.
3. At every split, consider only the feature subset selected by
   \`max_features\`.
4. Repeat for \`n_estimators\` trees.

For a sample \`x\`, the forest averages the trees' class probabilities:

\[
\hat{P}(y=c\mid x)=\frac{1}{T}\sum_{t=1}^{T}P_t(y=c\mid x)
\]

The predicted class is the class with the largest mean probability. If a
bootstrap sample omits a class, that tree contributes zero to that class's
column; the forest always aligns every tree's output to the global
\`classes_\` order before averaging.

## API

~~~python
from ml_from_scratch import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
)
model.fit(X_train, y_train)

labels = model.predict(X_test)
probabilities = model.predict_proba(X_test)
importances = model.feature_importances_
~~~

- \`n_estimators\` must be a positive integer.
- \`max_depth\` and \`min_samples_split\` control each tree's growth.
- \`max_features\` supports \`None\`, \`"sqrt"\`, \`"log2"\`, a positive integer,
  or a fraction in \`(0, 1]\`. The default \`"sqrt"\` is a common classification
  setting.
- \`bootstrap\` controls sampling with replacement. When disabled, every tree
  receives all training rows, but feature sampling remains randomized.
- \`random_state\` controls both bootstrap draws and per-tree feature sampling.
  Reusing a seed reproduces the fitted forest.
- \`classes_\` stores the sorted global class labels.
- \`estimators_\` stores the fitted decision trees.
- \`estimators_samples_\` stores the row indices used to train each tree.
- \`feature_importances_\` aggregates the trees' normalized Gini-importance
  vectors and is normalized to sum to one when any tree makes a useful split.

A single one-dimensional feature vector is accepted by prediction methods.
Probability columns always follow \`classes_\`, including when some individual
bootstrap trees do not observe every class. A failed refit leaves an existing
fitted forest unchanged.

## Runnable example

From the repository root:

~~~bash
PYTHONPATH=src python examples/random_forest_demo.py
~~~

The implementation is intentionally single-process and educational; it does
not yet provide parallel fitting, out-of-bag scoring, sample weights, or
probability calibration.
