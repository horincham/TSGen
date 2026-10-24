from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from tsgen.data.xyz import Frame
from tsgen.filters.physics import PhysicsFilter
from tsgen.explore.config import GatingCfg


@dataclass
class PhysicsGater:
    cfg: GatingCfg
    angle_limits_deg: Tuple[float, float] = (50.0, 180.0)

    def filter(self, frames: List[Frame], df: pd.DataFrame) -> tuple[List[Frame], pd.DataFrame]:
        """Apply a light physical gate.

        Args:
            frames: frames aligned with df rows
            df: candidate dataframe

        Returns:
            kept_frames, df_out (with keep/reasons columns)
        """
        if len(frames) == 0 or len(df) == 0:
            return [], df.iloc[0:0].copy()
        if not self.cfg.enabled:
            df2 = df.copy()
            df2["keep"] = True
            df2["reasons"] = ""
            return frames, df2

        pf = PhysicsFilter(min_distance_angstrom=float(self.cfg.min_dist), angle_limits_deg=self.angle_limits_deg)
        kept, flags = pf.apply(frames)
        flags_df = pd.DataFrame(flags)
        df2 = df.reset_index(drop=True).copy()
        df2 = pd.concat([df2, flags_df[["keep", "reasons", "min_dist", "extreme_angle"]]], axis=1)
        kept_frames = [fr for fr, keep in zip(frames, df2["keep"].tolist()) if keep]
        df_kept = df2[df2["keep"]].reset_index(drop=True)
        return kept_frames, df_kept
