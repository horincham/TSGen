from __future__ import annotations

"""Utilities for assembling publication-ready Figure 3 datasets.

The workflow writes per-iteration artifacts (features, filter report, selection table,
and reaction-coordinate metrics). This module merges them into a single table that
can be directly consumed by downstream statistics / plotting scripts.
"""

from pathlib import Path
from typing import Optional
import pandas as pd


def build_iter_table(iter_dir: Path) -> pd.DataFrame:
    features = pd.read_csv(iter_dir / "features.csv") if (iter_dir / "features.csv").exists() else pd.DataFrame()
    selection = pd.read_csv(iter_dir / "selection.csv") if (iter_dir / "selection.csv").exists() else pd.DataFrame()
    geom = pd.read_csv(iter_dir / "geom_metrics.csv") if (iter_dir / "geom_metrics.csv").exists() else pd.DataFrame()

    df = features.copy()
    if not selection.empty:
        df = df.merge(selection, on=["id", "iter"], how="left", suffixes=("", "_sel"))
    if not geom.empty:
        df = df.merge(geom, on=["id", "iter"], how="left")

    return df


def append_run_dataset(run_dir: Path, iter_dir: Path, *, out_name: str = "figure3_dataset.csv") -> Path:
    out_path = run_dir / out_name
    df_iter = build_iter_table(iter_dir)
    if df_iter.empty:
        return out_path

    if out_path.exists():
        old = pd.read_csv(out_path)
        df = pd.concat([old, df_iter], axis=0, ignore_index=True)
    else:
        df = df_iter

    df.to_csv(out_path, index=False)
    return out_path
