from __future__ import annotations

from typing import List

import numpy as np

from tsgen.data.xyz import Frame
from tsgen.explore.config import DescriptorCfg
from tsgen.descriptors.features import compute_descriptor_matrix


def compute_features(frames: List[Frame], cfg: DescriptorCfg) -> np.ndarray:
    """Compute the configured descriptor matrix for training and acquisition."""
    if not frames:
        return np.zeros((0, 0), dtype=float)
    return compute_descriptor_matrix(frames, cfg)
