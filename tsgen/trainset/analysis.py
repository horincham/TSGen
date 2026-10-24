from __future__ import annotations

from pathlib import Path

import pandas as pd


def _safe_series(df: pd.DataFrame, col: str):
    return df[col] if col in df.columns else pd.Series(dtype=float)


def write_figures(tables_dir: Path, figures_dir: Path) -> None:
    """Write publication-oriented diagnostic data and figures.

    The figures are intended to be useful before force-field optimization:
    (1) internal-coordinate coverage, (2) PES curve coverage by family,
    (3) high-energy/downweight profile for the generated trainset.in, and
    (4) Gaussian success/failure funnel.  When later ReaxFF validation data are
    available, the same tables can be joined with fort.99/fort.13 to make the
    error histograms used in ReaxFF papers.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    # Coordinate family coverage table and horizontal bar chart.
    p = tables_dir / "selected_coordinates.csv"
    if p.exists():
        df = pd.read_csv(p)
        if not df.empty:
            s = df.groupby(["scan_type", "family"]).size().reset_index(name="n_coordinates")
            s.to_csv(tables_dir / "coordinate_family_coverage.csv", index=False)
            top = s.sort_values(["scan_type", "n_coordinates", "family"], ascending=[True, False, True]).head(35)
            plt.figure(figsize=(9.0, max(4.0, 0.30 * len(top))))
            labels = [f"{a}: {b}" for a, b in zip(top["scan_type"], top["family"])]
            plt.barh(labels, top["n_coordinates"])
            plt.xlabel("Selected scan coordinates")
            plt.ylabel("Internal-coordinate family")
            plt.title("Internal-coordinate coverage of the ReaxFF scan set")
            plt.tight_layout()
            plt.savefig(figures_dir / "coordinate_family_coverage.png", dpi=300)
            plt.close()

    # Job outcome funnel.
    p = tables_dir / "scan_jobs.csv"
    if p.exists():
        df = pd.read_csv(p)
        if not df.empty and "status" in df.columns:
            funnel = df.groupby(["scan_type", "status"]).size().reset_index(name="count")
            funnel.to_csv(tables_dir / "gaussian_job_outcome_by_type.csv", index=False)
            pivot = funnel.pivot(index="scan_type", columns="status", values="count").fillna(0)
            plt.figure(figsize=(7.0, 4.5))
            bottom = None
            for col in sorted(pivot.columns):
                vals = pivot[col].values
                if bottom is None:
                    plt.bar(pivot.index.astype(str), vals, label=str(col))
                    bottom = vals
                else:
                    plt.bar(pivot.index.astype(str), vals, bottom=bottom, label=str(col))
                    bottom = bottom + vals
            plt.ylabel("Gaussian jobs")
            plt.xlabel("Job type")
            plt.title("Gaussian scan-job outcome")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(figures_dir / "gaussian_job_outcome_by_type.png", dpi=300)
            plt.close()

    # Scan PES curves and summary statistics.
    p = tables_dir / "scan_energy_curves.csv"
    if p.exists():
        df = pd.read_csv(p)
        if not df.empty and "e_kcalmol_rel" in df.columns:
            ok = df[df["scan_type"].isin(["bond", "angle", "torsion"])].copy()
            ok = ok[pd.notna(ok["e_kcalmol_rel"])]
            if not ok.empty:
                ok["abs_e_kcalmol_rel"] = ok["e_kcalmol_rel"].abs()
                ok.to_csv(tables_dir / "pes_curve_points.csv", index=False)

                summary = ok.groupby(["scan_type", "family"]).agg(
                    n_curves=("coord_id", "nunique"),
                    n_points=("id", "count"),
                    e_min=("e_kcalmol_rel", "min"),
                    e_max=("e_kcalmol_rel", "max"),
                    e_mean=("e_kcalmol_rel", "mean"),
                    e_abs_p50=("abs_e_kcalmol_rel", "median"),
                    e_abs_p95=("abs_e_kcalmol_rel", lambda x: x.quantile(0.95)),
                ).reset_index()
                summary["e_range"] = summary["e_max"] - summary["e_min"]
                summary.to_csv(tables_dir / "pes_family_energy_summary.csv", index=False)

                # One family-annotated PES curve panel per scan type.  Thin curves
                # show individual scans; the legend is deliberately omitted for
                # readability, with the CSV carrying exact identities.
                for stype in ["bond", "angle", "torsion"]:
                    sub = ok[ok["scan_type"] == stype].copy()
                    if sub.empty:
                        continue
                    plt.figure(figsize=(7.5, 5.0))
                    for _coord_id, g in list(sub.groupby("coord_id"))[:40]:
                        g = g.sort_values("target_value")
                        plt.plot(g["target_value"], g["e_kcalmol_rel"], marker="o", linewidth=1.0, markersize=2.2, alpha=0.80)
                    xlabel = "Bond distance (Å)" if stype == "bond" else ("Angle (deg)" if stype == "angle" else "Torsion angle (deg)")
                    plt.xlabel(xlabel)
                    plt.ylabel("DFT relative energy (kcal/mol)")
                    plt.title(f"{stype.capitalize()} relaxed-scan PES curves")
                    plt.tight_layout()
                    plt.savefig(figures_dir / f"pes_curves_{stype}.png", dpi=300)
                    plt.close()

                # Energy distribution by type, useful as a trainset difficulty map.
                plt.figure(figsize=(7.0, 4.8))
                groups = [ok[ok["scan_type"] == st]["abs_e_kcalmol_rel"].values for st in ["bond", "angle", "torsion"]]
                labels = [st for st, vals in zip(["bond", "angle", "torsion"], groups) if len(vals)]
                groups = [vals for vals in groups if len(vals)]
                if groups:
                    plt.boxplot(groups, labels=labels, showfliers=False)
                    plt.ylabel("|DFT relative energy| (kcal/mol)")
                    plt.xlabel("Scan type")
                    plt.title("Energy-scale distribution of training targets")
                    plt.tight_layout()
                    plt.savefig(figures_dir / "target_energy_scale_by_type.png", dpi=300)
                    plt.close()

    # Trainset weights: essential because in ReaxFF larger weights reduce contribution.
    p = tables_dir / "energy_targets.csv"
    if p.exists():
        df = pd.read_csv(p)
        if not df.empty and {"target_kcalmol", "weight", "scan_type"}.issubset(df.columns):
            df["abs_target_kcalmol"] = df["target_kcalmol"].abs()
            df.to_csv(tables_dir / "energy_target_weight_profile.csv", index=False)
            plt.figure(figsize=(7.0, 4.8))
            for stype, g in df.groupby("scan_type"):
                plt.scatter(g["abs_target_kcalmol"], g["weight"], s=18, alpha=0.75, label=str(stype))
            plt.xlabel("|DFT relative energy| (kcal/mol)")
            plt.ylabel("ReaxFF trainset weight / tolerance")
            plt.title("Energy-target weighting profile")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(figures_dir / "energy_target_weight_profile.png", dpi=300)
            plt.close()


def write_report(exports_dir: Path, tables_dir: Path, report_path: Path, summary: dict) -> None:
    lines = [
        "# TSGen ReaxFF Trainset Builder Report",
        "",
        "## Summary",
        "",
    ]
    for k, v in summary.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Core output files")
    lines.extend([
        "",
        "- `geo`: final multi-structure ReaxFF/BGF geometry file containing equilibrium and successful scan structures",
        "- `trainset.in`: final standalone ReaxFF optimization target file",
        "- `tables/selected_coordinates.csv`: selected bond/angle/torsion coordinates",
        "- `tables/scan_energy_curves.csv`: parsed DFT scan energies for plotting",
        "- `tables/energy_targets.csv`: ENERGY entries actually written to trainset.in",
        "- `tables/energy_target_weight_profile.csv`: target energy and ReaxFF weight/tolerance audit",
        "- `tables/pes_family_energy_summary.csv`: publication-oriented PES family summary",
        "- `figures/pes_curves_*.png`: PES curve plots grouped by scan type",
        "",
    ])
    lines.append("## Notes")
    lines.append("")
    lines.append("- ENERGY targets are DFT relative energies against the corresponding equilibrium parent structure.")
    lines.append("- Standalone ReaxFF uses error = ((ReaxFF - QC/Lit) / Weight)^2, so larger Weight downweights a data point.")
    lines.append("- Exact zero QC/Lit targets are replaced by a small non-zero epsilon because ReaxFF trainset.in cannot safely use 0.0000 targets.")
    lines.append("- Input structures may be supplied directly as an XYZ directory; Explore GEO is no longer required.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
