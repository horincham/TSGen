from __future__ import annotations
import numpy as np

def minmax01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < eps:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo + eps)

def safe_std(a: np.ndarray, axis: int = 0) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return np.nanstd(a, axis=axis, ddof=0)
