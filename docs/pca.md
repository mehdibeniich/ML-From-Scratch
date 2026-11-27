# Principal Component Analysis (PCA)

PCA is an unsupervised dimensionality-reduction method. It centers the data and
uses singular value decomposition (SVD) to find orthogonal directions ordered
by captured variance. This implementation uses NumPy's SVD and does not depend
on a machine-learning library.

## API

- `PCA(n_components=None)`: retain all available components.
- `PCA(n_components=2)`: retain the first two principal directions.
- `PCA(n_components=0.95)`: retain the smallest number of components whose
  cumulative explained-variance ratio reaches 95%.
- `fit(X)`: learn the mean, principal axes, singular values, and variance.
- `transform(X)`: center and project samples into component coordinates.
- `fit_transform(X)`: fit and project the training samples.
- `inverse_transform(Z)`: reconstruct approximate samples from component scores.

Integer component counts must be between 1 and `min(X.shape)`. Fractional
targets must be finite values in (0, 1]. For all-constant or single-sample
data, explained variance ratios are zero and a fractional target keeps one
component. Failed fits do not overwrite previously learned state.

## Mathematics

For centered matrix X, SVD gives X = U S Vᵀ. The rows of Vᵀ are
principal directions. The variance of component i is sᵢ² / (n - 1) when
there are at least two samples. Projection is (X - mean) Vₖ, and the
reconstruction is Z Vₖᵀ + mean.

## Example

```python
import numpy as np
from ml_from_scratch import PCA

X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
pca = PCA(n_components=0.95)
Z = pca.fit_transform(X)

print(pca.explained_variance_ratio_)
print(Z)
print(pca.inverse_transform(Z))
```

## Notes

- Inputs must be finite numeric 2D arrays.
- New data must have the same feature count as the fit data.
- PCA is sensitive to feature scale; standardize heterogeneous features
  before fitting when appropriate.
- Principal-component signs are mathematically arbitrary; compare subspaces
  or reconstructed data rather than expecting a fixed sign across SVD backends.