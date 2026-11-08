from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from tsgen.explore.config import AnchorsCfg
from tsgen.selection.fps import fps_indices
from tsgen.explore.surrogate.metrics import regression_metrics


@dataclass
class AnchorManager:
    cfg: AnchorsCfg

    def init_anchors(self, df_pool: pd.DataFrame, X_pool: np.ndarray, run_dir: Path) -> pd.DataFrame:
        """Create a fixed anchor set and persist it."""
        anchors_file = run_dir / "anchors.csv"
        if anchors_file.exists():
            return pd.read_csv(anchors_file)

        if not self.cfg.enabled or len(df_pool) == 0:
            anchors = df_pool.iloc[0:0].copy()
            anchors.to_csv(anchors_file, index=False)
            return anchors

        n = min(self.cfg.n_anchors, len(df_pool))
        if self.cfg.strategy == "random":
            anchors = df_pool.sample(n=n, random_state=0).reset_index(drop=True)
        elif self.cfg.strategy == "fps":
            # a light farthest-point sampling in descriptor space
            idx = self._fps(X_pool, n)
            anchors = df_pool.iloc[idx].reset_index(drop=True)
        else:  # stratified_by_species
            anchors = self._stratified(df_pool, n).reset_index(drop=True)

        anchors.to_csv(anchors_file, index=False)
        return anchors

    def evaluate(self, anchors: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray) -> dict:
        """Evaluate surrogate on anchors; requires oracle_energy present."""
        y = anchors["oracle_energy"].to_numpy(dtype=float)
        err = mu - y
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        # Gaussian NLL (calibration proxy)
        s = np.maximum(sigma, 1e-6)
        nll = float(np.mean(0.5 * np.log(2.0 * np.pi * s ** 2) + 0.5 * (err ** 2) / (s ** 2)))
        return {"mae": mae, "rmse": rmse, "nll": nll}

    @staticmethod
    def _fps(X: np.ndarray, k: int) -> list[int]:
        return fps_indices(X, k)

    @staticmethod
    def _stratified(df: pd.DataFrame, n: int) -> pd.DataFrame:
        if "species_id" not in df.columns:
            return df.sample(n=n, random_state=0)
        parts = []
        buckets = {sp: g.copy() for sp, g in df.groupby("species_id")}
        keys = list(buckets.keys())
        i = 0
        while len(parts) < n and keys:
            sp = keys[i % len(keys)]
            g = buckets[sp]
            if len(g) == 0:
                keys.remove(sp)
                continue
            pick = g.sample(n=1, random_state=len(parts))
            parts.append(pick)
            buckets[sp] = g.drop(pick.index)
            i += 1
        out = pd.concat(parts, axis=0) if parts else df.iloc[0:0]
        return out.reset_index(drop=True).head(n)
