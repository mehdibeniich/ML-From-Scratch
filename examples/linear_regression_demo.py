"""Run a tiny linear-regression example."""

import numpy as np
from ml_from_scratch import LinearRegression


# Synthetic data: y is approximately 4 + 3x.
X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
y = np.array([4, 7, 10, 13, 16, 19], dtype=float)

model = LinearRegression(learning_rate=0.05, n_iterations=2_000)
model.fit(X, y)

print(f"Weight: {model.weights[0]:.3f}")
print(f"Bias: {model.bias:.3f}")
print(f"Prediction for x=6: {model.predict([[6]])[0]:.3f}")
print(f"Final training loss: {model.cost_history[-1]:.6f}")
