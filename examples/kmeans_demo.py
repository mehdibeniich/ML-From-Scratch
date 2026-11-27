"""Runnable K-Means clustering example."""

import numpy as np

from ml_from_scratch import KMeans

X = np.array([
    [1.0, 1.2], [1.3, 0.8], [0.7, 1.0],
    [7.0, 7.2], [7.3, 6.8], [6.7, 7.0],
    [-5.0, 4.0], [-5.3, 4.2], [-4.7, 3.8],
])

model = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
print("Centers:")
print(model.cluster_centers_)
print("Labels:", model.labels_)
print("Inertia:", round(model.inertia_, 3))
print("New sample cluster:", model.predict([[7.1, 7.0]])[0])
