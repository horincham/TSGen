from __future__ import annotations
from typing import Dict
import pandas as pd

def apply_quotas(df: pd.DataFrame, by_field: str, quotas: Dict[str, int], score_col: str = "S") -> pd.DataFrame:
    picks = []
    for key, sub in df.groupby(by_field):
        q = quotas.get(str(key), 0)
        if q <= 0:
            continue
        picks.append(sub.sort_values(score_col, ascending=False).head(q))
    if not picks:
        return df.iloc[0:0]
    return pd.concat(picks, axis=0)
