"""Evaluate a multiclass classifier with precision, recall, and F1."""

from ml_from_scratch import accuracy_score, f1_score, precision_score, recall_score

y_true = ["cat", "cat", "dog", "dog", "bird", "bird"]
y_pred = ["cat", "dog", "dog", "bird", "bird", "bird"]

print("Accuracy:", accuracy_score(y_true, y_pred))
for average in ("micro", "macro", "weighted"):
    print(
        f"{average.title()} precision:",
        precision_score(y_true, y_pred, average=average),
    )
    print(
        f"{average.title()} recall:",
        recall_score(y_true, y_pred, average=average),
    )
    print(
        f"{average.title()} F1:",
        f1_score(y_true, y_pred, average=average),
    )
