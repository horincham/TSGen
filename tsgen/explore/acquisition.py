from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from tsgen.explore.config import AcquisitionCfg
from tsgen.selection.fps import fps_indices


def _min_dist_to_set(X: np.ndarray, Xref: np.ndarray) -> np.ndarray:
    """Compute min Euclidean distance from each row of X to any row of Xref."""
    if Xref is None or Xref.size == 0:
        return np.ones((X.shape[0],), dtype=float)
    out = np.empty((X.shape[0],), dtype=float)
    chunk = 512
    for i in range(0, X.shape[0], chunk):
        Xi = X[i : i + chunk]
        d2 = ((Xi[:, None, :] - Xref[None, :, :]) ** 2).sum(axis=2)
        out[i : i + chunk] = np.sqrt(np.min(d2, axis=1) + 1e-12)
    return out


def _fps_on_subset(X: np.ndarray, k: int, start_idx: int = 0) -> List[int]:
    """Light farthest-point sampling on rows of X. Returns indices in [0, n)."""
    return fps_indices(X, k, start=start_idx)


@dataclass
class Acquisition:
    """Score and select candidates for labeling.

    Supports:
      - method='sigma' : pure uncertainty sampling (sigma)
      - method='ucb'   : UCB score = mu + kappa*sigma
    and an optional two-stage selection:
      1) preselect top-M by score (M = preselect_factor*k)
      2) diversity rerank via FPS in descriptor space
    """

    cfg: AcquisitionCfg

    def compute(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        labeled_X: np.ndarray,
        labeled_species_counts: Dict[str, int],
    ) -> pd.DataFrame:
        out = df.copy()

        mu = out.get("mu", out.get("mu_kcalmol", 0.0))
        sigma = out.get("sigma", out.get("sigma_kcalmol", 0.0))

        out["mu"] = np.asarray(mu, dtype=float)
        out["sigma"] = np.asarray(sigma, dtype=float)

        # base acquisition
        if self.cfg.method == "sigma":
            out["acq"] = out["sigma"]
        elif self.cfg.method == "ucb":
            out["acq"] = out["mu"] + float(self.cfg.ucb_kappa) * out["sigma"]
        else:
            raise ValueError(f"Unsupported acquisition method: {self.cfg.method}")

        # uncertainty component (still useful for reporting)
        out["u"] = out["sigma"]

        # diversity / novelty: min distance to labeled set (higher is more novel)
        if self.cfg.diversity.method == "farthest_in_descriptor":
            out["d"] = _min_dist_to_set(X, labeled_X)
        elif self.cfg.diversity.method in ("fps", "none"):
            out["d"] = 0.0
        else:
            raise ValueError(f"Unsupported diversity method: {self.cfg.diversity.method}")

        # reaction relevance placeholder
        out["r"] = 0.0

        # new-species bonus
        ns = []
        for sp in out.get("species_id", "").tolist():
            c = labeled_species_counts.get(str(sp), 0)
            ns.append(1.0 if c == 0 else 1.0 / np.sqrt(c))
        out["new_species"] = np.asarray(ns, dtype=float)

        # robust normalization (MAD) to keep scoring stable
        for col in ["acq", "u", "d", "r", "new_species"]:
            v = out[col].to_numpy(dtype=float)
            denom = np.median(np.abs(v - np.median(v))) + 1e-12
            out[col] = v / denom

        w = self.cfg.weights
        out["score"] = (
            float(w.u) * out["u"].to_numpy(dtype=float)
            + float(w.d) * out["d"].to_numpy(dtype=float)
            + float(w.r) * out["r"].to_numpy(dtype=float)
            + float(w.new_species) * out["new_species"].to_numpy(dtype=float)
            + 1.0 * out["acq"].to_numpy(dtype=float)
        )
        return out

    def select_indices(self, df_scored: pd.DataFrame, X_scored: np.ndarray) -> List[int]:
        """Return selected indices into df_scored (row positions), not pool indices."""
        k = int(self.cfg.k_select)
        if len(df_scored) == 0 or k <= 0:
            return []

        df = df_scored.copy()
        df["_original_pos"] = np.arange(len(df), dtype=int)
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

        M = int(max(k, min(len(df), int(self.cfg.preselect_factor) * k)))
        pre = df.head(M).reset_index(drop=True)
        original_positions = pre["_original_pos"].to_numpy(dtype=int)

        if self.cfg.diversity.method != "farthest_in_descriptor" or len(pre) <= k:
            return original_positions[: min(k, len(pre))].tolist()

        # diversity rerank within preselected set using FPS; start from the top-score point
        X_pre = X_scored[original_positions]
        sel_local = _fps_on_subset(X_pre, k=k, start_idx=0)
        return original_positions[np.asarray(sel_local, dtype=int)].tolist()
