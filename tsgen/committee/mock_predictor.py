from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
from tsgen.data.xyz import Frame
from tsgen.utils.math import safe_std

def _simple_energy(frame: Frame, w: float) -> float:
    coords = frame.coords
    n = coords.shape[0]
    e = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = float(np.linalg.norm(coords[i] - coords[j]) + 1e-6)
            e += 1.0 / r
    bias = 0.1 * sum(1.0 for s in frame.symbols if s == "C") + 0.02 * sum(1.0 for s in frame.symbols if s == "H")
    return w * e + bias

@dataclass
class MockCommittee:
    size: int
    jitter_sigma: float
    seed: int = 0

    def energies_per_member(self, frames: List[Frame]) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        ws = 1.0 + rng.normal(0.0, self.jitter_sigma, size=self.size)
        Em = np.zeros((self.size, len(frames)), dtype=float)
        for m in range(self.size):
            for i, fr in enumerate(frames):
                Em[m, i] = _simple_energy(fr, w=float(ws[m]))
        return Em

    def uncertainty_uE(self, frames: List[Frame]) -> np.ndarray:
        Em = self.energies_per_member(frames)
        return safe_std(Em, axis=0)
