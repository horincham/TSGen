from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class SensitivityRunCfg(BaseModel):
    out_dir: Path = Path("outputs_sensitivity")
    name: str = "reaxff_parameter_sensitivity"
    seed: int = 20260724


class SensitivityInputsCfg(BaseModel):
    ffield: Path
    geo: Path | None = None
    trainset_in: Path | None = None
    control: Path | None = None


class BoundRuleCfg(BaseModel):
    section: int
    item: int
    delta: float
    lower_bound: float
    upper_bound: float


class ParameterSpaceCfg(BaseModel):
    mode: Literal["auto_from_ffield", "from_params_ml", "from_rules"] = "auto_from_ffield"
    params_ml: Path | None = None
    allowed_sections: list[int] = Field(default_factory=lambda: [3, 4, 5, 6, 7])
    section_item_allowlist: dict[int, list[int]] = Field(default_factory=dict)
    bound_rules: list[BoundRuleCfg] = Field(default_factory=list)
    bounds_overrides: dict[str, tuple[float, float, float]] = Field(default_factory=dict)
    include_charge_parameters: bool = True
    charge_atom_items: list[int] = Field(default_factory=lambda: [14, 15])
    relative_range: float = Field(default=0.20, gt=0, lt=1)
    delta_fraction: float = Field(default=0.01, gt=0)
    minimum_delta: float = Field(default=1.0e-4, gt=0)
    include_zero_parameters: bool = False
    zero_tolerance: float = Field(default=1.0e-12, ge=0)
    zero_absolute_range: float = Field(default=1.0, gt=0)
    force_include: list[str] = Field(default_factory=list)
    force_exclude: list[str] = Field(default_factory=list)
    require_initial_inside_bounds: bool = True


class SamplingCfg(BaseModel):
    method: Literal["latin_hypercube"] = "latin_hypercube"
    samples: int = Field(default=2000, ge=2)
    center_initial_ffield: bool = True


class ReaxFFCfg(BaseModel):
    command: str | list[str] = "./reac"
    executable_source: Path | None = None
    timeout_seconds: int = Field(default=1800, ge=1)
    expected_total_output: str = "fort.13"
    expected_item_output: str = "fort.99"
    stage_mode: Literal["copy", "symlink"] = "symlink"


class SensitivitySlurmCfg(BaseModel):
    partition: str | None = None
    time: str = "24:00:00"
    cpus_per_task: int = Field(default=1, ge=1)
    mem: str = "2G"
    array_parallelism: int = Field(default=100, ge=1)
    max_array_size: int = Field(default=1000, ge=1)
    sequential_batches: bool = True
    extra_lines: list[str] = Field(default_factory=list)


class SensitivityExecutionCfg(BaseModel):
    mode: Literal["local", "manual", "slurm"] = "manual"
    workers: int = Field(default=1, ge=1)
    submit: bool = False
    wait_for_completion: bool = False
    poll_seconds: int = Field(default=60, ge=1)
    max_wait_seconds: int = Field(default=86400, ge=1)
    reuse_completed: bool = True
    slurm: SensitivitySlurmCfg = Field(default_factory=SensitivitySlurmCfg)


class SensitivityAnalysisCfg(BaseModel):
    targets: list[Literal["TOTAL", "ENERGY", "GEOMETRY", "CHARGE"]] = Field(
        default_factory=lambda: ["TOTAL", "ENERGY", "GEOMETRY", "CHARGE"]
    )
    parameter_sigma: float = Field(default=0.3, gt=0)
    loss_gamma: float = Field(default=0.3, gt=0)
    repeats: int = Field(default=10, ge=2)
    subsample_size: int = Field(default=2000, ge=4)
    minimum_success_fraction: float = Field(default=0.95, gt=0, le=1)
    cumulative_threshold: float = Field(default=0.90, gt=0, le=1)
    rank_top_k: int = Field(default=20, ge=1)
    convergence_sizes: list[int] = Field(default_factory=lambda: [500, 1000, 2000, 5000, 10000])


class SensitivityOutputsCfg(BaseModel):
    write_figures: bool = True
    export_indeedopt_bundle: bool = True
    indeedopt_root: Path | None = None


class SensitivityConfig(BaseModel):
    run: SensitivityRunCfg = Field(default_factory=SensitivityRunCfg)
    inputs: SensitivityInputsCfg
    parameter_space: ParameterSpaceCfg = Field(default_factory=ParameterSpaceCfg)
    sampling: SamplingCfg = Field(default_factory=SamplingCfg)
    reaxff: ReaxFFCfg = Field(default_factory=ReaxFFCfg)
    execution: SensitivityExecutionCfg = Field(default_factory=SensitivityExecutionCfg)
    analysis: SensitivityAnalysisCfg = Field(default_factory=SensitivityAnalysisCfg)
    outputs: SensitivityOutputsCfg = Field(default_factory=SensitivityOutputsCfg)

    @model_validator(mode="after")
    def _validate_cross_fields(self):
        if self.parameter_space.mode == "from_params_ml" and self.parameter_space.params_ml is None:
            raise ValueError("parameter_space.params_ml is required for mode=from_params_ml")
        if self.analysis.subsample_size > self.sampling.samples:
            raise ValueError("analysis.subsample_size cannot exceed sampling.samples")
        return self


def _resolve(base: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_sensitivity_config(path: str | Path) -> SensitivityConfig:
    source = Path(path).resolve()
    data: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    base = source.parent
    run = data.setdefault("run", {})
    run["out_dir"] = str(_resolve(base, run.get("out_dir", "outputs_sensitivity")))
    inputs = data.setdefault("inputs", {})
    for key in ("ffield", "geo", "trainset_in", "control"):
        if inputs.get(key) is not None:
            inputs[key] = str(_resolve(base, inputs[key]))
    parameter_space = data.setdefault("parameter_space", {})
    if parameter_space.get("params_ml") is not None:
        parameter_space["params_ml"] = str(_resolve(base, parameter_space["params_ml"]))
    reaxff = data.setdefault("reaxff", {})
    if reaxff.get("executable_source") is not None:
        reaxff["executable_source"] = str(_resolve(base, reaxff["executable_source"]))
    outputs = data.setdefault("outputs", {})
    if outputs.get("indeedopt_root") is not None:
        outputs["indeedopt_root"] = str(_resolve(base, outputs["indeedopt_root"]))
    return SensitivityConfig.model_validate(data)


def save_sensitivity_config(cfg: SensitivityConfig, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
