from __future__ import annotations

from typing import List

import numpy as np

from tsgen.data.xyz import Frame
from tsgen.descriptors.coulomb import coulomb_matrix_eigs
from tsgen.descriptors.soap import soap_feature_matrix


def _coulomb_max_atoms(cfg) -> int:
    if hasattr(cfg, "max_atoms"):
        return int(getattr(cfg, "max_atoms"))
    if hasattr(cfg, "coulomb_eigs"):
        return int(getattr(getattr(cfg, "coulomb_eigs"), "max_atoms"))
    return 256


def compute_descriptor_matrix(frames: List[Frame], cfg) -> np.ndarray:
    if not frames:
        return np.zeros((0, 0), dtype=float)
    backend = str(getattr(cfg, "backend", "coulomb_eigs"))
    if backend == "soap":
        return soap_feature_matrix(frames, cfg)
    if backend == "coulomb_eigs":
        max_atoms = _coulomb_max_atoms(cfg)
        return np.vstack([coulomb_matrix_eigs(fr, max_atoms=max_atoms) for fr in frames]).astype(float)
    raise ValueError(f"Unsupported descriptor backend: {backend}")
