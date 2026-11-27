"""Runnable PCA dimensionality-reduction example."""

import numpy as np

from ml_from_scratch import PCA, StandardScaler

X = np.array([
    [2.5, 2.4, 1.0],
    [0.5, 0.7, 0.2],
    [2.2, 2.9, 0.9],
    [1.9, 2.2, 0.8],
    [3.1, 3.0, 1.1],
    [2.3, 2.7, 0.9],
    [2.0, 1.6, 0.6],
    [1.0, 1.1, 0.3],
])

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_scaled)

print("Reduced shape:", X_reduced.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative explained variance:", pca.explained_variance_ratio_.sum())
print("Reconstruction:")
print(pca.inverse_transform(X_reduced))