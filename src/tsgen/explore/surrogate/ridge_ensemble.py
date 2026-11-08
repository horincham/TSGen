from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from tsgen.explore.config import RidgeEnsembleCfg
from tsgen.explore.surrogate.base import SurrogateModel


@dataclass
class RidgeEnsemble(SurrogateModel):
    cfg: RidgeEnsembleCfg
    scaler: StandardScaler | None = None
    models: List[Ridge] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = self.scaler.fit_transform(X)
        n = Xs.shape[0]
        rng = np.random.default_rng(0)
        self.models = []
        for m in range(int(self.cfg.n_models)):
            if self.cfg.bootstrap and n > 1:
                idx = rng.integers(0, n, size=n)
                Xm = Xs[idx]
                ym = y[idx]
            else:
                Xm = Xs
                ym = y
            reg = Ridge(alpha=float(self.cfg.alpha), random_state=m)
            reg.fit(Xm, ym)
            self.models.append(reg)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.scaler is None or self.models is None or len(self.models) == 0:
            mu = np.zeros((X.shape[0],), dtype=float)
            sigma = np.ones((X.shape[0],), dtype=float)
            return mu, sigma
        Xs = self.scaler.transform(X)
        preds = np.stack([m.predict(Xs) for m in self.models], axis=0)
        mu = preds.mean(axis=0)
        sigma = preds.std(axis=0) + 1e-12
        return mu, sigma

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"cfg": self.cfg.model_dump(), "scaler": self.scaler, "models": self.models}, f)

    @classmethod
    def load(cls, path: Path) -> "RidgeEnsemble":
        with open(path, "rb") as f:
            d = pickle.load(f)
        cfg = RidgeEnsembleCfg.model_validate(d.get("cfg", {}))
        obj = cls(cfg=cfg)
        obj.scaler = d.get("scaler")
        obj.models = d.get("models")
        return obj
