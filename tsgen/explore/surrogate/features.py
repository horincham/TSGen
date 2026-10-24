from __future__ import annotations

from typing import List

import numpy as np

from tsgen.data.xyz import Frame
from tsgen.explore.config import DescriptorCfg
from tsgen.descriptors.features import compute_descriptor_matrix


def compute_features(frames: List[Frame], cfg: DescriptorCfg) -> np.ndarray:
    """Compute feature matrix for surrogate training and acquisition.

    Default uses SOAP, which is standard for descriptor-space diversity analysis
    in MLIP/active-learning workflows.  Coulomb eigenvalues remain available as
    a dependency-light fallback for quick checks and legacy runs.
    """
    if not frames:
        return np.zeros((0, 0), dtype=float)
    return compute_descriptor_matrix(frames, cfg)
