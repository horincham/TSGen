from __future__ import annotations

from pathlib import Path

from tsgen.explore.config import SurrogateCfg
from tsgen.explore.surrogate.ridge_ensemble import RidgeEnsemble
from tsgen.explore.surrogate.gp import GaussianProcessSurrogate


def build_surrogate(cfg: SurrogateCfg, snapshot: Path | None = None):
    """Create (or load) a surrogate model from config."""
    if cfg.kind == "ridge_ensemble":
        if snapshot is not None and snapshot.exists():
            return RidgeEnsemble.load(snapshot)
        return RidgeEnsemble(cfg=cfg.ridge_ensemble)
    if cfg.kind == "gp":
        if snapshot is not None and snapshot.exists():
            return GaussianProcessSurrogate.load(snapshot)
        return GaussianProcessSurrogate(cfg=cfg.gp)
    raise ValueError(f"Unsupported surrogate kind: {cfg.kind}")
