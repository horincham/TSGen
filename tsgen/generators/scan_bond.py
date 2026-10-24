from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
from tsgen.data.xyz import Frame
from tsgen.generators.base import Proposal

@dataclass
class BondScan:
    n_per_parent: int
    i: int
    j: int
    r_min: float
    r_max: float

    def generate(self, parents: List[Frame], parent_ids: List[str], rng: np.random.Generator) -> List[Proposal]:
        out: List[Proposal] = []
        for fr, pid in zip(parents, parent_ids):
            ri = fr.coords[self.i].copy()
            rj = fr.coords[self.j].copy()
            v = rj - ri
            norm = float(np.linalg.norm(v) + 1e-12)
            e = v / norm
            rs = np.linspace(self.r_min, self.r_max, num=self.n_per_parent)
            for idx, r_target in enumerate(rs):
                new_coords = fr.coords.copy()
                new_coords[self.j] = ri + e * r_target
                new = Frame(
                    symbols=fr.symbols,
                    coords=new_coords,
                    comment=f"id=TO_BE_SET source=bond_scan parent={pid}",
                )
                out.append(
                    Proposal(
                        frame=new,
                        source="g1_bond_scan",
                        propose_params={"i": self.i, "j": self.j, "r": float(r_target), "seq": idx},
                        parent_id=pid,
                    )
                )
        return out
