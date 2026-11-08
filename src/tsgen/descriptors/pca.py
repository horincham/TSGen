from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

def pca_2d(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Two plot coordinates only; safe for empty, singleton and constant pools."""
    if n_components != 2:
        raise ValueError('PCA is visualization-only and requires n_components=2')
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError('PCA visualization needs a finite two-dimensional matrix')
    Y = np.zeros((len(X), 2))
    if len(X) < 2 or X.shape[1] == 0 or not np.any(X != X[0]):
        return Y
    count = min(2, *X.shape)
    Y[:, :count] = PCA(n_components=count, svd_solver='full', whiten=False).fit_transform(X)
    return Y
