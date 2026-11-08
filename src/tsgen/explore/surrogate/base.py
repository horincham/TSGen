from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class SurrogateModel(ABC):
    """Abstract surrogate interface."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mu, sigma)."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "SurrogateModel":
        ...
