# K-Means Clustering

The repository's `KMeans` implements Lloyd's clustering algorithm using NumPy.

## Usage

```python
import numpy as np
from ml_from_scratch import KMeans

X = np.array([[1.0, 2.0], [1.5, 1.8], [8.0, 8.0], [8.5, 8.2]])
model = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
print(model.cluster_centers_)
print(model.labels_)
print(model.inertia_)
```

## Algorithm

Each initialization selects centers using K-Means++: the first center is sampled
uniformly, and later centers are sampled proportional to squared distance from
the nearest selected center. Lloyd iterations assign every sample to its nearest
center and recompute each center as its cluster mean. Empty clusters are
reinitialized at poorly represented samples. The lowest-inertia result across
`n_init` runs is retained.

## Parameters and attributes

- `n_clusters`: number of centers (1 through sample count)
- `max_iter`: iteration limit per initialization
- `tol`: squared center movement convergence threshold
- `n_init`: independent initializations
- `random_state`: seed for repeatable results
- `cluster_centers_`, `labels_`, `inertia_`, `n_iter_`

`predict(X)` assigns new samples, `transform(X)` returns distances to each
center, and `fit_predict(X)` fits and returns training labels.

K-Means minimizes within-cluster squared Euclidean distance and is sensitive to
feature scales and outliers. Scale features when their units differ; the model
does not scale data automatically. Cluster label numbers are arbitrary and may
permute between fits.
