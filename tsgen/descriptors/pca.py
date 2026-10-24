from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

def pca_2d(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    pca = PCA(n_components=n_components)
    Y = pca.fit_transform(X)
    return Y
