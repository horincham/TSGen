from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
from tsgen.data.xyz import Frame
from tsgen.generators.base import Proposal

def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    ba /= (np.linalg.norm(ba) + 1e-12)
    bc /= (np.linalg.norm(bc) + 1e-12)
    cosv = np.clip(float(np.dot(ba, bc)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))

def rotate_around_axis(p: np.ndarray, origin: np.ndarray, axis: np.ndarray, theta_rad: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    v = p - origin
    v_rot = (
        v * np.cos(theta_rad)
        + np.cross(axis, v) * np.sin(theta_rad)
        + axis * (np.dot(axis, v)) * (1.0 - np.cos(theta_rad))
    )
    return origin + v_rot

@dataclass
class AngleDihedralScan:
    n_per_parent: int
    i: int
    j: int
    k: int
    angle_min_deg: float
    angle_max_deg: float
    dihedral_enabled: bool = False
    l: int = 3
    phi_min_deg: float = -60
    phi_max_deg: float = 60

    def generate(self, parents: List[Frame], parent_ids: List[str], rng: np.random.Generator) -> List[Proposal]:
        out: List[Proposal] = []
        for fr, pid in zip(parents, parent_ids):
            angles = np.linspace(self.angle_min_deg, self.angle_max_deg, num=self.n_per_parent)
            for seq, ang_t in enumerate(angles):
                coords = fr.coords.copy()
                pi, pj, pk = coords[self.i], coords[self.j], coords[self.k]
                v1 = pi - pj
                v2 = pk - pj
                axis = np.cross(v2, v1)
                if np.linalg.norm(axis) < 1e-8:
                    axis = rng.normal(size=3)
                cur = angle_deg(pi, pj, pk)
                dtheta = np.radians(ang_t - cur)
                coords[self.k] = rotate_around_axis(pk, origin=pj, axis=axis, theta_rad=dtheta)

                if self.dihedral_enabled and self.l < len(coords):
                    phis = np.linspace(self.phi_min_deg, self.phi_max_deg, num=2)
                    for pseq, phi in enumerate(phis):
                        coords2 = coords.copy()
                        axis2 = coords2[self.k] - coords2[self.j]
                        coords2[self.l] = rotate_around_axis(coords2[self.l], origin=coords2[self.k], axis=axis2, theta_rad=np.radians(phi))
                        new = Frame(fr.symbols, coords2, f"id=TO_BE_SET source=angle_dihedral parent={pid}")
                        out.append(
                            Proposal(
                                frame=new,
                                source="g2_angle_dihedral",
                                propose_params={"i": self.i, "j": self.j, "k": self.k, "angle_deg": float(ang_t), "phi_deg": float(phi), "seq": f"{seq}_{pseq}"},
                                parent_id=pid,
                            )
                        )
                else:
                    new = Frame(fr.symbols, coords, f"id=TO_BE_SET source=angle_scan parent={pid}")
                    out.append(
                        Proposal(
                            frame=new,
                            source="g2_angle",
                            propose_params={"i": self.i, "j": self.j, "k": self.k, "angle_deg": float(ang_t), "seq": seq},
                            parent_id=pid,
                        )
                    )
        return out
