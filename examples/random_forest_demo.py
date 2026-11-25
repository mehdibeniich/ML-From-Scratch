"""Demonstrate Random Forest classification on a synthetic dataset."""

import numpy as np

from ml_from_scratch import RandomForestClassifier, accuracy_score

X_train = np.array([
    [1.0, 2.1, 0.1], [1.2, 1.9, 0.2], [0.8, 2.2, 0.0],
    [2.0, 0.9, 1.0], [2.2, 1.1, 0.9], [1.8, 0.8, 1.1],
    [3.0, 3.1, 2.0], [3.2, 2.9, 2.1], [2.8, 3.2, 1.9],
    [1.1, 2.0, 0.2], [2.1, 1.0, 1.2], [3.1, 3.0, 2.2],
])
y_train = np.array([
    "class-a", "class-a", "class-a",
    "class-b", "class-b", "class-b",
    "class-c", "class-c", "class-c",
    "class-a", "class-b", "class-c",
])

X_test = np.array([[1.0, 2.0, 0.1], [2.0, 1.0, 1.0], [3.0, 3.0, 2.0]])
y_test = np.array(["class-a", "class-b", "class-c"])

model = RandomForestClassifier(
    n_estimators=25,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
).fit(X_train, y_train)

predictions = model.predict(X_test)
print("Classes:", model.classes_)
print("Predictions:", predictions)
print("Probabilities:")
print(np.round(model.predict_proba(X_test), 3))
print("Feature importances:", np.round(model.feature_importances_, 3))
print("Accuracy:", accuracy_score(y_test, predictions))
