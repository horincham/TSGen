from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator


class RunCfg(BaseModel):
    out_dir: str = "outputs_trainset"
    name: str = "reaxff_scan_trainset"
    seed: int = 0


class InputCfg(BaseModel):
    # Standalone scan/trainset construction should not depend on Explore GEO.
    # Default input is now a directory of optimized equilibrium .xyz files.
    # ``geo_file`` is retained only for legacy input compatibility.
    source: Literal["xyz_dir", "explore_final", "geo_file"] = "xyz_dir"
    xyz_dir: Optional[str] = None
    geo_file: Optional[str] = None
    run_dir: Optional[str] = None
    structures_dir: str = "exports/final/structures"
    geo_index: Optional[str] = "exports/final/tables/geo_index.csv"
    pattern: str = "*.xyz"
    recursive: bool = False
    deduplicate_structures: bool = True
    dedup_fingerprint_tol: float = 0.03


class ChemistryCfg(BaseModel):
    allowed_elements: Optional[List[str]] = Field(default_factory=lambda: ["C", "H", "O"])
    required_elements: List[str] = Field(default_factory=list)
    exclude_elements: List[str] = Field(default_factory=list)
    min_element_counts: Dict[str, int] = Field(default_factory=dict)
    max_element_counts: Dict[str, int] = Field(default_factory=lambda: {"C": 9, "O": 2})
    exclude_open_shell: bool = True
    charge: Union[int, Literal["auto"]] = "auto"
    multiplicity: Union[int, Literal["auto"]] = "auto"


class BondScanSelectionCfg(BaseModel):
    enabled: bool = True
    max_per_family: int = 2
    max_total: Optional[int] = 40
    families: List[str] = Field(default_factory=lambda: [
        "C-H", "C-C_single", "C-C_double", "C-C_triple",
        "C-O_single", "C-O_double", "C-O_triple", "O-H", "O-O_single", "H-H",
    ])


class AngleScanSelectionCfg(BaseModel):
    enabled: bool = True
    max_per_family: int = 2
    max_total: Optional[int] = 50
    min_angle_deg: float = 45.0
    max_angle_deg: float = 175.0


class TorsionScanSelectionCfg(BaseModel):
    enabled: bool = True
    max_per_family: int = 2
    max_total: Optional[int] = 40
    exclude_ring_torsions: bool = True
    exclude_double_bond_center: bool = True
    exclude_triple_bond_related: bool = True
    reject_near_linear_angle_deg: float = 165.0
    reject_near_zero_angle_deg: float = 15.0
    exclude_terminal_hydrogen_rotation: bool = True


class CoordinateSelectionCfg(BaseModel):
    bond_scans: BondScanSelectionCfg = Field(default_factory=BondScanSelectionCfg)
    angle_scans: AngleScanSelectionCfg = Field(default_factory=AngleScanSelectionCfg)
    torsion_scans: TorsionScanSelectionCfg = Field(default_factory=TorsionScanSelectionCfg)
    max_coordinates_per_parent: int = 12
    prefer_small_molecules: bool = True
    protected_formulas: List[str] = Field(default_factory=lambda: [
        "H2", "O2", "H2O", "CO", "CO2", "CH4", "C2H2", "C2H4", "C2H6", "CH2O", "CH4O", "H2O2",
    ])


class BondScanGenerationCfg(BaseModel):
    # Fallback absolute bond scan range in Angstrom.  For production relaxed
    # scans, family_ranges below is used preferentially because a uniform
    # 1.0--5.0 Å relaxed scan is often too aggressive for X-H and multiple bonds.
    start: float = 1.0
    stop: float = 5.0
    step: float = 0.25
    include_equilibrium_if_off_grid: bool = False
    min_distance: float = 0.55
    use_family_ranges: bool = True
    family_ranges: Dict[str, Dict[str, float]] = Field(default_factory=lambda: {
        # Near-equilibrium + dissociation region.  These are intentionally more
        # conservative than a global 1--5 Å relaxed scan to avoid Gaussian
        # ModRedundant failures in CH/O-H and multiple-bond scans.
        "H-H": {"start": 0.50, "stop": 3.00, "step": 0.10},
        "C-H": {"start": 0.80, "stop": 3.00, "step": 0.10},
        "O-H": {"start": 0.70, "stop": 3.00, "step": 0.10},
        "O-O_single": {"start": 0.90, "stop": 3.50, "step": 0.15},
        "C-C_single": {"start": 1.00, "stop": 4.00, "step": 0.15},
        "C-C_double": {"start": 1.00, "stop": 3.50, "step": 0.15},
        "C-C_triple": {"start": 0.90, "stop": 3.00, "step": 0.10},
        "C-O_single": {"start": 0.90, "stop": 3.50, "step": 0.15},
        "C-O_double": {"start": 0.90, "stop": 3.00, "step": 0.10},
        "C-O_triple": {"start": 0.90, "stop": 3.00, "step": 0.10},
    })


class AngleScanGenerationCfg(BaseModel):
    # Unified relaxed angle scan range, in degrees.
    start_deg: float = 70.0
    stop_deg: float = 140.0
    step_deg: float = 10.0
    include_equilibrium_if_off_grid: bool = False


class TorsionScanGenerationCfg(BaseModel):
    # Unified torsion scan over one full rotation. include_endpoint=False avoids duplicating 0 and 360 deg.
    start_deg: float = 0.0
    stop_deg: float = 360.0
    step_deg: float = 30.0
    include_endpoint: bool = False
    include_equilibrium_if_off_grid: bool = False


class ScanGenerationCfg(BaseModel):
    # relaxed_modredundant generates one Gaussian Opt=ModRedundant scan job per selected internal coordinate.
    # rigid_sp is retained only as a fallback/debug option.
    mode: Literal["relaxed_modredundant", "rigid_sp"] = "relaxed_modredundant"
    bond: BondScanGenerationCfg = Field(default_factory=BondScanGenerationCfg)
    angle: AngleScanGenerationCfg = Field(default_factory=AngleScanGenerationCfg)
    torsion: TorsionScanGenerationCfg = Field(default_factory=TorsionScanGenerationCfg)
    min_nonbonded_distance: float = 0.50
    write_input_guess_xyz: bool = True


class SlurmCfg(BaseModel):
    partition: Optional[str] = None
    time: Optional[str] = None
    cpus_per_task: Optional[int] = None
    mem: Optional[str] = None
    extra_lines: List[str] = Field(default_factory=list)


class GaussianCfg(BaseModel):
    mode: Literal["manual", "local", "slurm"] = "manual"
    # Must contain Opt=ModRedundant for relaxed_modredundant; it will be appended automatically if missing.
    route: str = "# M062X/6-311G(d) Opt=ModRedundant"
    reference_route: str = "# M062X/6-311G(d) SP"
    write_chk: bool = True
    chk_name: str = "input.chk"
    auto_formchk: bool = True
    formchk_cmd: str = "formchk"
    charge: Union[int, Literal["auto"]] = "auto"
    multiplicity: Union[int, Literal["auto"]] = "auto"
    nproc: int = 8
    mem: str = "16GB"
    g16_cmd: str = "g16"
    slurm: SlurmCfg = Field(default_factory=SlurmCfg)


class ExecutionCfg(BaseModel):
    submit: bool = False
    wait_for_completion: bool = False
    poll_seconds: int = 60
    max_wait_seconds: int = 86400
    require_all_ok: bool = False
    min_ok_frac: float = 0.95
    reuse_completed: bool = True


class GeometryWeightsCfg(BaseModel):
    bond: float = 0.02
    angle: float = 1.0
    torsion: float = 5.0


class GeometryTrainsetCfg(BaseModel):
    enabled: bool = True
    include_equilibrium_bonds: bool = True
    include_equilibrium_angles: bool = True
    include_equilibrium_torsions: bool = True
    zero_epsilon: float = 0.0001
    weights: GeometryWeightsCfg = Field(default_factory=GeometryWeightsCfg)


class EnergyWeightsCfg(BaseModel):
    # ReaxFF computes error as ((ReaxFF - QC/Lit) / weight)^2.
    # Therefore larger weights mean *less* contribution, not more importance.
    # Defaults are modest tolerances; high-energy points can be downweighted dynamically.
    bond_scan: float = 1.0
    angle_scan: float = 1.0
    torsion_scan: float = 2.0


class EnergyTrainsetCfg(BaseModel):
    enabled: bool = True
    reference: Literal["equilibrium_parent"] = "equilibrium_parent"
    unit: Literal["kcalmol"] = "kcalmol"
    # Exact zero QC/Lit values are unsafe for standalone ReaxFF trainset.in.
    # Keep the data point but replace 0.0000 by zero_epsilon_kcalmol.
    skip_zero_targets: bool = False
    zero_epsilon_kcalmol: float = 0.0001
    max_rel_kcalmol: Optional[float] = 400.0
    min_rel_kcalmol: Optional[float] = None
    weights: EnergyWeightsCfg = Field(default_factory=EnergyWeightsCfg)
    dynamic_weight_by_energy: bool = True
    dynamic_weight_scale_kcalmol: float = 50.0
    max_dynamic_weight: float = 25.0
    min_dynamic_weight: float = 0.5


class ChargeTrainsetCfg(BaseModel):
    # ADCH is preferred when a Multiwfn charge export is present.  Gaussian does
    # not natively print ADCH, so Hirshfeld is the robust automated fallback.
    enabled: bool = True
    method: Literal["adch", "hirshfeld"] = "adch"
    fallback_method: Literal["hirshfeld", "none"] = "hirshfeld"
    # Optional command template for site-specific Multiwfn ADCH postprocessing.
    # Common placeholders: {wavefunction_file_q}, {job_dir_q}, {output_file_q}, {natoms}.
    adch_command: Optional[str] = None
    adch_wavefunction_patterns: List[str] = Field(default_factory=lambda: ["*.fchk", "*.wfn", "*.wfx", "*.molden", "*.mwfn"])
    adch_output_files: List[str] = Field(default_factory=lambda: [
        "adch_charges.txt",
        "adch.out",
        "multiwfn_adch.out",
        "charges.adch",
        "atomic_charges.txt",
    ])
    adch_timeout_seconds: int = 600
    weight: float = 0.1
    include_equilibrium: bool = True
    # Scan-point charges require per-point population output.  Keep disabled by
    # default to avoid accidentally assigning a final scan-job charge to every
    # relaxed scan step.
    include_scan_points: bool = False


class TrainsetCfg(BaseModel):
    geometry: GeometryTrainsetCfg = Field(default_factory=GeometryTrainsetCfg)
    energy: EnergyTrainsetCfg = Field(default_factory=EnergyTrainsetCfg)
    charge: ChargeTrainsetCfg = Field(default_factory=ChargeTrainsetCfg)


class GeoOutputCfg(BaseModel):
    # Molecular BGF-style ReaxFF geo is the default. Periodic XTLGRF/CRYSTX
    # output is only written when explicitly requested.
    periodic: bool = False
    cell: List[float] = Field(default_factory=lambda: [80.0, 80.0, 80.0, 90.0, 90.0, 90.0])
    include_format_lines: bool = True
    remark: str = "TSGen ReaxFF training geometry"
    equilibrium_rutype: str = "NORMAL RUN"
    scan_rutype: str = "SINGLE POINT"


class OutputsCfg(BaseModel):
    write_geo: bool = True
    write_bgf: bool = True
    write_trainset_in: bool = True
    # models.in is optional because a single combined geo + trainset.in is usually enough.
    write_models_in: bool = False
    write_figures: bool = True
    write_report: bool = True
    geo: GeoOutputCfg = Field(default_factory=GeoOutputCfg)


class TrainsetBuildCfg(BaseModel):
    run: RunCfg = Field(default_factory=RunCfg)
    input: InputCfg = Field(default_factory=InputCfg)
    chemistry: ChemistryCfg = Field(default_factory=ChemistryCfg)
    coordinate_selection: CoordinateSelectionCfg = Field(default_factory=CoordinateSelectionCfg)
    scan_generation: ScanGenerationCfg = Field(default_factory=ScanGenerationCfg)
    gaussian: GaussianCfg = Field(default_factory=GaussianCfg)
    execution: ExecutionCfg = Field(default_factory=ExecutionCfg)
    trainset: TrainsetCfg = Field(default_factory=TrainsetCfg)
    outputs: OutputsCfg = Field(default_factory=OutputsCfg)

    @model_validator(mode="after")
    def _normalize_routes(self):
        if self.scan_generation.mode == "relaxed_modredundant":
            r = str(self.gaussian.route)
            if "opt" not in r.lower():
                self.gaussian.route = r.rstrip() + " Opt=ModRedundant"
            elif "modredundant" not in r.lower():
                # Opt without ModRedundant would ignore the trailing freeze constraints.
                self.gaussian.route = r.replace("Opt", "Opt=ModRedundant") if "Opt" in r else r.rstrip() + " Opt=ModRedundant"
        if self.trainset.charge.enabled and self.trainset.charge.fallback_method == "hirshfeld":
            self.gaussian.reference_route = _ensure_pop_hirshfeld(self.gaussian.reference_route)
            self.gaussian.route = _ensure_pop_hirshfeld(self.gaussian.route)
        return self


def _ensure_pop_hirshfeld(route: str) -> str:
    r = str(route).strip()
    if "pop=" in r.lower() and "hirshfeld" in r.lower():
        return r
    if "pop=" in r.lower():
        return r
    return r + " Pop=Hirshfeld"


def load_trainset_config(path: str | Path) -> TrainsetBuildCfg:
    p = Path(path)
    data: Dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if "trainset" in data and isinstance(data["trainset"], dict) and "run" not in data:
        data = data["trainset"]
    return TrainsetBuildCfg.model_validate(data)


def save_trainset_config(cfg: TrainsetBuildCfg, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.model_dump(mode="python"), sort_keys=False, allow_unicode=True), encoding="utf-8")
