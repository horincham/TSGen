from __future__ import annotations
from typing import List
import numpy as np

def fps_indices(X: np.ndarray, k: int, start: int = 0) -> List[int]:
    n = X.shape[0]
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    chosen = [start % n]
    dist = np.full(n, np.inf, dtype=float)
    for _ in range(1, k):
        last = chosen[-1]
        d = np.linalg.norm(X - X[last:last+1], axis=1)
        dist = np.minimum(dist, d)
        # Exclude every selected row, including ties/identical descriptors.
        dist[chosen] = -np.inf
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
    return chosen
