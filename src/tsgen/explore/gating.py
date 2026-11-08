from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import hashlib
import json
from pathlib import Path

import pandas as pd

from tsgen.data.xyz import Frame
from tsgen.filters.physics import PhysicsFilter
from tsgen.explore.config import GatingCfg
from tsgen.filters.reference_bonds import load_reference_bonds, resolve_reference_bonds


@dataclass
class PhysicsGater:
    cfg: GatingCfg
    angle_limits_deg: Tuple[float, float] | None = None

    def signature(self) -> str:
        payload = {'version': 2, 'config': self.cfg.model_dump(mode='json'), 'angle_override': self.angle_limits_deg}
        if self.cfg.enabled and self.cfg.reference_bonds_file:
            payload['reference_sha256'] = hashlib.sha256(Path(self.cfg.reference_bonds_file).read_bytes()).hexdigest()
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def filter(self, frames: List[Frame], df: pd.DataFrame) -> tuple[List[Frame], pd.DataFrame]:
        """Apply a light physical gate.

        Args:
            frames: frames aligned with df rows
            df: candidate dataframe

        Returns:
            kept_frames, df_out (with keep/reasons columns)
        """
        if len(frames) != len(df):
            raise ValueError('Geometry screening requires aligned frames and table rows')
        if len(frames) == 0:
            return [], df.iloc[0:0].copy()
        if not self.cfg.enabled:
            df2 = df.copy()
            df2["keep"] = True
            df2["reasons"] = ""
            for column in ('min_dist', 'rho_max', 'theta_min', 'reference_bond_count'):
                df2[column] = float('nan')
            df2['extreme_angle'] = False
            df2['rho_status'] = 'disabled'
            df2['gating_reference_id'] = ''
            df2['gating_signature'] = self.signature()
            return frames, df2

        references, reference_ids = None, [''] * len(frames)
        if self.cfg.reference_bonds_file:
            references, reference_ids = resolve_reference_bonds(frames, df, load_reference_bonds(self.cfg.reference_bonds_file))
        elif self.cfg.max_bond_ratio is not None:
            raise ValueError('max_bond_ratio requires reference_bonds_file with fixed reference connectivity')
        pf = PhysicsFilter(min_distance_angstrom=float(self.cfg.min_dist),
                           angle_limits_deg=self.angle_limits_deg or (self.cfg.min_angle_deg, 180.),
                           max_bond_ratio=self.cfg.max_bond_ratio, bond_scale=self.cfg.bond_scale,
                           max_bond_angstrom=self.cfg.max_bond_angstrom)
        kept, flags = pf.apply(frames, reference_bonds=references)
        flags_df = pd.DataFrame(flags)
        df2 = df.reset_index(drop=True).copy()
        for column in flags_df.columns:
            df2[column] = flags_df[column].values
        df2['gating_reference_id'] = reference_ids
        df2['gating_signature'] = self.signature()
        kept_frames = [fr for fr, keep in zip(frames, df2["keep"].tolist()) if keep]
        # Runner requires one audit row per original candidate to build its mask.
        return kept_frames, df2
