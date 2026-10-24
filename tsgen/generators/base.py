from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, List, Dict, Any, Optional
import numpy as np
from tsgen.data.xyz import Frame

@dataclass
class Proposal:
    frame: Frame
    source: str
    propose_params: Dict[str, Any]
    parent_id: Optional[str]

class Generator(Protocol):
    def generate(self, parents: List[Frame], parent_ids: List[str], rng: np.random.Generator) -> List[Proposal]:
        ...
