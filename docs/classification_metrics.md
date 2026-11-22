# Classification metrics: precision, recall, and F1

The `ml_from_scratch` package exposes `precision_score`, `recall_score`,
and `f1_score`. Each accepts the same arguments:

```python
score(y_true, y_pred, average="binary", pos_label=1, labels=None, zero_division=0)
```

- `average="binary"`: score the `pos_label` class. Exactly two classes must
  occur in the input. The default positive label is `1`.
- `average="micro"`: aggregate true positives, false positives, and false
  negatives across classes before calculating the score.
- `average="macro"`: calculate each class score and take the unweighted mean.
- `average="weighted"`: average per-class scores weighted by true class support.
- `average=None`: return a NumPy array of per-class scores.
- `labels`: optionally fixes class order and can include classes not observed
  in the input. Every observed true or predicted class must be included.
- `zero_division`: choose `0` or `1` for a score whose denominator is zero.

For single-label multiclass problems, micro precision, micro recall, and micro
F1 all equal accuracy. Macro averaging treats classes equally; weighted
averaging accounts for class frequency.

Run the complete example from the repository root after installing
`requirements.txt`:

```bash
PYTHONPATH=src python examples/classification_metrics_demo.py
```

Focused tests cover binary and string labels, multiclass averaging, custom
positive labels, class-order control, zero-denominator behavior, and invalid
inputs.
