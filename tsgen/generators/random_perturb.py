from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
from tsgen.data.xyz import Frame
from tsgen.generators.base import Proposal

@dataclass
class RandomPerturb:
    n_per_parent: int
    sigma_angstrom: float

    def generate(self, parents: List[Frame], parent_ids: List[str], rng: np.random.Generator) -> List[Proposal]:
        out: List[Proposal] = []
        for fr, pid in zip(parents, parent_ids):
            for t in range(self.n_per_parent):
                disp = rng.normal(0.0, self.sigma_angstrom, size=fr.coords.shape)
                new = Frame(
                    symbols=fr.symbols,
                    coords=fr.coords + disp,
                    comment=f"id=TO_BE_SET source=random_perturb parent={pid}",
                )
                out.append(
                    Proposal(
                        frame=new,
                        source="g_random",
                        propose_params={"sigma": self.sigma_angstrom, "trial": t},
                        parent_id=pid,
                    )
                )
        return out
