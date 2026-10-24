from __future__ import annotations
from typing import Protocol, List
import numpy as np
from tsgen.data.xyz import Frame

class EnergyPredictor(Protocol):
    def predict_energy(self, frames: List[Frame]) -> np.ndarray:
        ...
