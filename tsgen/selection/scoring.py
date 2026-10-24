from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd
from tsgen.utils.math import minmax01

def reaction_relevance_geometry(frames, rule: Dict[str, Any] | None = None) -> np.ndarray:
    if rule is None:
        rule = {"pairs": [(0, 1, 1.54, 0.3)]}
    rs = []
    for fr in frames:
        val = 0.0
        for (i, j, r0, scale) in rule["pairs"]:
            if i < len(fr.coords) and j < len(fr.coords):
                r = float(np.linalg.norm(fr.coords[i] - fr.coords[j]))
                val += abs(r - r0) / (scale + 1e-12)
        rs.append(val)
    return np.array(rs, dtype=float)

@dataclass
class ScoreWeights:
    lambda_u: float
    lambda_r: float
    lambda_d: float
    use_u: bool = True
    use_r: bool = True
    use_d: bool = True

def compute_total_score(df: pd.DataFrame, w: ScoreWeights) -> pd.DataFrame:
    u = df["score_u"].to_numpy()
    r = df["score_r"].to_numpy()
    d = df["score_d"].to_numpy()

    u01 = minmax01(u)
    r01 = minmax01(r)
    d01 = minmax01(d)

    S = 0.0
    if w.use_u:
        S += w.lambda_u * u01
    if w.use_r:
        S += w.lambda_r * r01
    if w.use_d:
        S += w.lambda_d * d01

    out = df.copy()
    out["u_norm"] = u01
    out["r_norm"] = r01
    out["d_norm"] = d01
    out["S"] = S
    return out
