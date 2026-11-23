"""Demonstrate Gaussian Naive Bayes on a small synthetic multiclass dataset."""

import numpy as np

from ml_from_scratch import GaussianNB, accuracy_score

X_train = np.array([
    [-2.2, -1.0], [-1.8, -1.2], [-2.0, -0.7], [-1.6, -0.9],
    [0.0, 2.0], [0.3, 1.7], [-0.2, 2.3], [0.2, 2.1],
    [2.8, -1.0], [3.2, -0.8], [3.0, -1.3], [3.4, -1.1],
])
y_train = np.array([
    "blue", "blue", "blue", "blue",
    "red", "red", "red", "red",
    "green", "green", "green", "green",
])

X_test = np.array([[-2.0, -1.0], [0.1, 2.0], [3.1, -1.0]])
y_test = np.array(["blue", "red", "green"])

model = GaussianNB(var_smoothing=1e-9).fit(X_train, y_train)

print("Classes:", model.classes_)
print("Predictions:", model.predict(X_test))
print("Probabilities:")
print(np.round(model.predict_proba(X_test), 3))
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
