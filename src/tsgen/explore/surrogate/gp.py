from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel

from tsgen.explore.config import GPCfg
from tsgen.explore.surrogate.base import SurrogateModel


@dataclass
class GaussianProcessSurrogate(SurrogateModel):
    """Sklearn GPR wrapper providing (mu, sigma) for acquisition.

    This is intended for small/medium labeled sets. For larger sets, limit training
    via cfg.max_train (uniform subsampling).
    """

    cfg: GPCfg
    model: GaussianProcessRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.shape[0] == 0:
            return
        if X.shape[0] > int(self.cfg.max_train):
            # deterministic subsample
            idx = np.linspace(0, X.shape[0] - 1, int(self.cfg.max_train)).astype(int)
            X = X[idx]
            y = y[idx]

        if self.cfg.kernel == "matern32":
            kern = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5)
        else:
            kern = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
        self.model = GaussianProcessRegressor(
            kernel=kern,
            alpha=float(self.cfg.alpha),
            normalize_y=bool(self.cfg.normalize_y),
            n_restarts_optimizer=1,
            random_state=0,
        )
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if self.model is None or X.shape[0] == 0:
            return np.zeros((X.shape[0],), dtype=float), np.ones((X.shape[0],), dtype=float)
        mu, std = self.model.predict(X, return_std=True)
        return np.asarray(mu, dtype=float), np.asarray(std, dtype=float)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"cfg": self.cfg.model_dump(), "model": self.model}, f)

    @staticmethod
    def load(path: Path) -> "GaussianProcessSurrogate":
        with open(path, "rb") as f:
            d = pickle.load(f)
        cfg = GPCfg.model_validate(d.get("cfg", {}))
        m = GaussianProcessSurrogate(cfg=cfg)
        m.model = d.get("model")
        return m
