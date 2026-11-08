from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def write_diagnostic_tables(tables_dir: Path) -> None:
    coordinates = _read_table(tables_dir / "selected_coordinates.csv")
    if not coordinates.empty:
        coverage = coordinates.groupby(["scan_type", "family"]).size().reset_index(name="n_coordinates")
        coverage.to_csv(tables_dir / "coordinate_family_coverage.csv", index=False)

    jobs = _read_table(tables_dir / "scan_jobs.csv")
    if not jobs.empty and "status" in jobs.columns:
        outcomes = jobs.groupby(["scan_type", "status"]).size().reset_index(name="count")
        outcomes.to_csv(tables_dir / "gaussian_job_outcome_by_type.csv", index=False)

    curves = _read_table(tables_dir / "scan_energy_curves.csv")
    if not curves.empty and "e_kcalmol_rel" in curves.columns:
        curves = curves.loc[
            curves["scan_type"].isin(["bond", "angle", "torsion"])
            & curves["e_kcalmol_rel"].notna()
        ].copy()
        if not curves.empty:
            curves["abs_e_kcalmol_rel"] = curves["e_kcalmol_rel"].abs()
            curves.to_csv(tables_dir / "pes_curve_points.csv", index=False)
            summary = curves.groupby(["scan_type", "family"]).agg(
                n_curves=("coord_id", "nunique"),
                n_points=("id", "count"),
                e_min=("e_kcalmol_rel", "min"),
                e_max=("e_kcalmol_rel", "max"),
                e_mean=("e_kcalmol_rel", "mean"),
                e_abs_p50=("abs_e_kcalmol_rel", "median"),
                e_abs_p95=("abs_e_kcalmol_rel", lambda values: values.quantile(0.95)),
            ).reset_index()
            summary["e_range"] = summary["e_max"] - summary["e_min"]
            summary.to_csv(tables_dir / "pes_family_energy_summary.csv", index=False)

    targets = _read_table(tables_dir / "energy_targets.csv")
    if not targets.empty and {"target_kcalmol", "weight", "scan_type"}.issubset(targets.columns):
        targets["abs_target_kcalmol"] = targets["target_kcalmol"].abs()
        targets.to_csv(tables_dir / "energy_target_weight_profile.csv", index=False)


def write_report(report_path: Path, summary: dict) -> None:
    lines = [
        "# TSGen ReaxFF training set",
        "",
        "## Summary",
        "",
    ]
    lines.extend(f"- **{key}**: {value}" for key, value in summary.items())
    lines.extend([
        "",
        "## Outputs",
        "",
        "- `geo` contains the equilibrium and accepted scan geometries.",
        "- `trainset.in` contains the energy, geometry, and charge targets.",
        "- `tables/selected_coordinates.csv` lists the bond, angle, and torsion scan coordinates.",
        "- `tables/scan_energy_curves.csv` lists the parsed DFT scan energies.",
        "- `tables/energy_targets.csv` lists the exported ENERGY entries.",
        "- `tables/energy_target_weight_profile.csv` records energy targets and their tolerances.",
        "- `tables/pes_family_energy_summary.csv` summarizes scan coverage by coordinate family.",
        "",
        "ENERGY targets are relative to the corresponding optimized equilibrium structures.",
        "Each ReaxFF contribution is the squared residual divided by the squared tolerance.",
        "Exact zero targets can be represented by a small nonzero epsilon for standalone ReaxFF compatibility.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
