from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ALMetrics:
    iter: int
    timestamp: str
    stage: str
    dft_used: int
    dft_budget: int
    n_pool: int
    labeled_train: int
    labeled_holdout: int
    n_species_labeled: int

    # Holdout evaluation (optional)
    mae_holdout: float | None = None
    rmse_holdout: float | None = None
    nll_holdout: float | None = None
    cov_1sigma: float | None = None
    cov_2sigma: float | None = None

    # Uncertainty summaries (optional)
    mean_sigma_selected: float | None = None
    mean_sigma_pool: float | None = None
    max_sigma_pool: float | None = None

    # Operational stats (optional)
    submitted_this_iter: int | None = None
    collected_ok_this_iter: int | None = None
    collected_fail_this_iter: int | None = None


def _safe_mean(x: np.ndarray) -> float | None:
    if x is None or x.size == 0:
        return None
    return float(np.mean(x))


def _safe_max(x: np.ndarray) -> float | None:
    if x is None or x.size == 0:
        return None
    return float(np.max(x))


def evaluate_holdout(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.maximum(sigma, 1e-6)

    err = mu - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    nll = float(np.mean(0.5 * np.log(2.0 * np.pi * sigma**2) + 0.5 * (err**2) / (sigma**2)))

    # coverage
    z = np.abs(err) / sigma
    cov_1 = float(np.mean(z <= 1.0))
    cov_2 = float(np.mean(z <= 2.0))
    return {"mae": mae, "rmse": rmse, "nll": nll, "cov_1sigma": cov_1, "cov_2sigma": cov_2}


def append_metrics_csv(path: Path, m: ALMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(m)
    df = pd.DataFrame([row])
    if path.exists():
        prev = pd.read_csv(path)
        out = pd.concat([prev, df], ignore_index=True)
        out.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def update_plots(metrics_csv: Path, plots_dir: Path) -> None:
    """Generate/update lightweight ML-style training curves."""
    if not metrics_csv.exists():
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_csv)
    if len(df) == 0:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _plot(x, y, out_path: Path, xlabel: str, ylabel: str, title: str):
        plt.figure()
        plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    x = df["iter"].to_numpy()

    if "mean_sigma_pool" in df.columns:
        _plot(x, df["mean_sigma_pool"].to_numpy(), plots_dir / "sigma_mean_pool_vs_iter.png",
              "Iteration", "Mean sigma (kcal/mol)", "Mean surrogate uncertainty (pool)")

    if "n_species_labeled" in df.columns:
        _plot(x, df["n_species_labeled"].to_numpy(), plots_dir / "species_vs_iter.png",
              "Iteration", "# Species labeled", "Labeled species growth")

    if "rmse_holdout" in df.columns and df["rmse_holdout"].notna().any():
        _plot(x, df["rmse_holdout"].to_numpy(), plots_dir / "holdout_rmse_vs_iter.png",
              "Iteration", "RMSE (kcal/mol)", "Holdout RMSE")

    if "dft_used" in df.columns:
        _plot(x, df["dft_used"].to_numpy(), plots_dir / "dft_used_vs_iter.png",
              "Iteration", "DFT used", "Budget consumption")
