from __future__ import annotations

from pathlib import Path
import pandas as pd

def write_features_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def read_features_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
