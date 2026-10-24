from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray | None = None) -> dict:
    """Compute regression metrics used for AL convergence curves.

    Returns MAE, RMSE, and Gaussian NLL (if sigma is provided).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err))) if err.size else float("nan")
    rmse = float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan")
    out = {"mae": mae, "rmse": rmse}
    if sigma is not None:
        s = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
        nll = float(np.mean(0.5 * np.log(2.0 * np.pi * s ** 2) + 0.5 * (err ** 2) / (s ** 2)))
        out["nll"] = nll
    else:
        out["nll"] = None
    return out
