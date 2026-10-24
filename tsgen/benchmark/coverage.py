from __future__ import annotations

"""Deterministic benchmark metrics based on descriptor-space coverage.

This is a *model-free* proxy suitable for early, publication-facing MVP runs:
it measures how well the current training set covers a fixed benchmark pool B
in a fixed descriptor space.

Importantly, it does not fabricate any DFT error curve; it is a reproducible
geometric statistic that can legitimately stabilize/decline with iteration.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

from tsgen.data.xyz import Frame
from tsgen.descriptors.features import compute_descriptor_matrix


class _CoulombCompatCfg:
    backend = "coulomb_eigs"
    def __init__(self, max_atoms: int):
        self.max_atoms = int(max_atoms)


@dataclass
class CoverageStats:
    n_train: int
    n_bench: int
    nn_mean: float
    nn_median: float
    nn_p95: float
    nn_max: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_train": self.n_train,
            "n_bench": self.n_bench,
            "bench_nn_mean": self.nn_mean,
            "bench_nn_median": self.nn_median,
            "bench_nn_p95": self.nn_p95,
            "bench_nn_max": self.nn_max,
        }


def descriptor_coverage(train_frames: List[Frame], bench_frames: List[Frame], *, max_atoms: int = 256, descriptor_cfg=None) -> CoverageStats:
    if len(train_frames) == 0 or len(bench_frames) == 0:
        return CoverageStats(len(train_frames), len(bench_frames), float("nan"), float("nan"), float("nan"), float("nan"))

    cfg = descriptor_cfg if descriptor_cfg is not None else _CoulombCompatCfg(max_atoms)
    Xtr = compute_descriptor_matrix(train_frames, cfg)
    Xb = compute_descriptor_matrix(bench_frames, cfg)
    # NN distance for each benchmark point to training set
    d2 = ((Xb[:, None, :] - Xtr[None, :, :]) ** 2).sum(axis=2)
    nn = np.sqrt(d2.min(axis=1))
    return CoverageStats(
        n_train=int(Xtr.shape[0]),
        n_bench=int(Xb.shape[0]),
        nn_mean=float(np.mean(nn)),
        nn_median=float(np.median(nn)),
        nn_p95=float(np.quantile(nn, 0.95)),
        nn_max=float(np.max(nn)),
    )
