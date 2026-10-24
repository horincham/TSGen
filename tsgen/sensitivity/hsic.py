from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t


def gaussian_kernel(values: np.ndarray, sigma: float) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1, 1)
    distance_squared = (vector - vector.T) ** 2
    return np.exp(-distance_squared / (2.0 * sigma**2))


def conjunctive_gaussian_kernel(values: np.ndarray, gamma: float) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    feature = np.exp(-(vector**2) / (2.0 * gamma**2))
    return np.outer(feature, feature)


def unbiased_hsic(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] < 4:
        raise ValueError("HSIC kernels must be equal square matrices with at least four rows")
    k = np.array(first, dtype=float, copy=True)
    l = np.array(second, dtype=float, copy=True)
    np.fill_diagonal(k, 0.0)
    np.fill_diagonal(l, 0.0)
    n = k.shape[0]
    trace_term = float(np.sum(k * l.T))
    product_term = float(k.sum() * l.sum() / ((n - 1) * (n - 2)))
    cross_term = float(2.0 * np.dot(k.sum(axis=1), l.sum(axis=1)) / (n - 2))
    return max(0.0, (trace_term + product_term - cross_term) / (n * (n - 3)))


def transform_loss(values: np.ndarray) -> np.ndarray:
    loss = np.asarray(values, dtype=float)
    if not np.isfinite(loss).all() or np.any(loss < 0):
        raise ValueError("Loss values must be finite and non-negative")
    logged = np.log(np.maximum(loss, np.nextafter(0.0, 1.0)))
    span = float(logged.max() - logged.min())
    if span <= np.finfo(float).eps:
        raise ValueError("Loss vector has zero variance")
    return (logged - logged.min()) / span


def repeated_hsic(
    parameters: np.ndarray,
    loss: np.ndarray,
    parameter_ids: list[str],
    repeats: int,
    subsample_size: int,
    seed: int,
    sigma: float = 0.3,
    gamma: float = 0.3,
) -> pd.DataFrame:
    x = np.asarray(parameters, dtype=float)
    y = np.asarray(loss, dtype=float)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[1] != len(parameter_ids):
        raise ValueError("Parameter matrix, loss vector, and parameter IDs are inconsistent")
    if subsample_size > len(y):
        raise ValueError(f"Need {subsample_size} successful samples; observed {len(y)}")
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(repeats):
        indices = rng.choice(len(y), size=subsample_size, replace=False)
        transformed = transform_loss(y[indices])
        loss_kernel = conjunctive_gaussian_kernel(transformed, gamma)
        raw = np.array(
            [
                unbiased_hsic(gaussian_kernel(x[indices, column], sigma), loss_kernel)
                for column in range(x.shape[1])
            ],
            dtype=float,
        )
        total = float(raw.sum())
        normalized = raw / total if total > np.finfo(float).eps else np.zeros_like(raw)
        for parameter_id, raw_value, normalized_value in zip(
            parameter_ids, raw, normalized, strict=True
        ):
            rows.append(
                {
                    "repeat": repeat,
                    "parameter_id": parameter_id,
                    "raw_hsic": raw_value,
                    "normalized_hsic": normalized_value,
                }
            )
    return pd.DataFrame(rows)


def student_t_interval(values: np.ndarray) -> tuple[float, float, float, float]:
    vector = np.asarray(values, dtype=float)
    if len(vector) < 2:
        raise ValueError("At least two repeated HSIC estimates are required")
    mean = float(vector.mean())
    standard_deviation = float(vector.std(ddof=1))
    half_width = float(t.ppf(0.975, df=len(vector) - 1) * standard_deviation / np.sqrt(len(vector)))
    return (
        mean,
        standard_deviation,
        max(0.0, mean - half_width),
        min(1.0, mean + half_width),
    )

