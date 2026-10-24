from __future__ import annotations

import re
from pathlib import Path


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return cleaned or "sensitivity"


class SensitivityPaths:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.config_file = self.run_dir / "config.yaml"
        self.state_file = self.run_dir / "state.json"
        self.provenance_file = self.run_dir / "provenance.json"
        self.inputs_dir = self.run_dir / "inputs"
        self.tables_dir = self.run_dir / "tables"
        self.samples_dir = self.run_dir / "samples"
        self.baseline_dir = self.samples_dir / "baseline"
        self.figures_dir = self.run_dir / "figures"
        self.figure_source_dir = self.run_dir / "figure_source_data"
        self.exports_dir = self.run_dir / "exports"
        self.logs_dir = self.run_dir / "logs"
        self.slurm_dir = self.run_dir / "slurm"
        self.parameter_catalog = self.tables_dir / "parameter_catalog.csv"
        self.candidate_parameter_audit = self.tables_dir / "candidate_parameter_audit.csv"
        self.parameter_samples = self.tables_dir / "parameter_samples.csv"
        self.sample_status = self.tables_dir / "sample_status.csv"
        self.sample_losses = self.tables_dir / "sample_losses.csv"
        self.sample_item_losses = self.tables_dir / "sample_item_losses.csv"
        self.hsic_bootstrap_results = self.tables_dir / "hsic_bootstrap_results.csv"
        self.parameter_sensitivity_summary = self.tables_dir / "parameter_sensitivity_summary.csv"
        self.parameter_group_summary = self.tables_dir / "parameter_group_summary.csv"
        self.rank_convergence = self.tables_dir / "rank_convergence.csv"
        self.selection_decisions = self.tables_dir / "selection_decisions.csv"
        self.panel_a_source = self.figure_source_dir / "panel_a_parameter_ranking.csv"
        self.panel_b_source = self.figure_source_dir / "panel_b_group_sensitivity.csv"
        self.panel_c_source = self.figure_source_dir / "panel_c_target_sensitivity.csv"
        self.panel_d_source = self.figure_source_dir / "panel_d_cumulative_sensitivity.csv"
        self.selected_params_ml = self.exports_dir / "params-ml.selected"
        self.indeedopt_bundle = self.exports_dir / "indeedopt_bundle"

    @classmethod
    def next(cls, out_dir: str | Path, name: str) -> "SensitivityPaths":
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        prefix = f"{_safe_name(name)}_run_"
        indices = []
        for child in root.glob(f"{prefix}*"):
            if child.is_dir():
                try:
                    indices.append(int(child.name[len(prefix):]))
                except ValueError:
                    continue
        return cls(root / f"{prefix}{max(indices, default=0) + 1:04d}")

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> "SensitivityPaths":
        return cls(Path(run_dir).resolve())

    def ensure(self) -> None:
        for path in (
            self.run_dir,
            self.inputs_dir,
            self.tables_dir,
            self.samples_dir,
            self.figures_dir,
            self.figure_source_dir,
            self.exports_dir,
            self.logs_dir,
            self.slurm_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def sample_dir(self, sample_id: str) -> Path:
        return self.samples_dir / sample_id

