from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple, Dict, Any, List, Optional

import yaml
from pydantic import BaseModel, Field

class RunConfig(BaseModel):
    output_root: str = "outputs"
    run_id: int = 1
    seed_xyz: str
    benchmark_xyz: str

class WorkflowConfig(BaseModel):
    max_iters: int = 5
    dft_budget: int = 200
    select_k: int = 20
    gate: Literal["off", "kit"] = "off"
    resume: bool = True

class BondScanCfg(BaseModel):
    enabled: bool = True
    n_per_parent: int = 10
    i: int = 0
    j: int = 1
    r_min: float = 1.0
    r_max: float = 2.5

class DihedralCfg(BaseModel):
    enabled: bool = False
    l: int = 3
    phi_min_deg: float = -60
    phi_max_deg: float = 60

class AngleDihedralScanCfg(BaseModel):
    enabled: bool = True
    n_per_parent: int = 8
    i: int = 0
    j: int = 1
    k: int = 2
    angle_min_deg: float = 70
    angle_max_deg: float = 150
    dihedral: DihedralCfg = Field(default_factory=DihedralCfg)

class RandomPerturbCfg(BaseModel):
    enabled: bool = True
    n_per_parent: int = 10
    sigma_angstrom: float = 0.05
    seed: int = 0

class GeneratorsConfig(BaseModel):
    bond_scan: BondScanCfg = Field(default_factory=BondScanCfg)
    angle_dihedral_scan: AngleDihedralScanCfg = Field(default_factory=AngleDihedralScanCfg)
    random_perturb: RandomPerturbCfg = Field(default_factory=RandomPerturbCfg)

class FiltersConfig(BaseModel):
    min_distance_angstrom: float = 0.75
    max_bondlike_distance_angstrom: float = 3.2
    angle_limits_deg: Tuple[float, float] = (45.0, 170.0)

class CoulombEigsCfg(BaseModel):
    max_atoms: int = 32

class PCACfg(BaseModel):
    n_components: int = 2

class DescriptorConfig(BaseModel):
    backend: Literal["soap", "coulomb_eigs"] = "soap"
    coulomb_eigs: CoulombEigsCfg = Field(default_factory=CoulombEigsCfg)
    pca: PCACfg = Field(default_factory=PCACfg)
    soap_species: List[str] = Field(default_factory=lambda: ["H", "C", "N", "O", "F", "Si", "P", "S", "Cl"])
    soap_r_cut: float = 5.0
    soap_n_max: int = 8
    soap_l_max: int = 6
    soap_sigma: float = 0.5
    soap_periodic: bool = False
    soap_pooling: Literal["mean", "mean_std"] = "mean_std"

class CommitteeConfig(BaseModel):
    size: int = 5
    # "reaxff" enables a publication-grade uncertainty estimate based on
    # an ensemble of ReaxFF parameter sets (ffield_*) evaluated on the same x.
    source: Literal["bootstrap", "jitter", "snapshot", "reaxff"] = "jitter"
    jitter_sigma: float = 0.03

class ReaxFFBackendConfig(BaseModel):
    # Path to ReaxFF executable (or wrapper script). Required when committee.source == "reaxff".
    reaxff_exe: Optional[str] = None
    # Directory containing a committee of ffield parameter files.
    # Accepted patterns: ffield_*.ffield, ffield-*/ffield, or user-provided exact file names.
    ffield_dir: Optional[str] = None
    ffield_glob: str = "**/ffield*"
    # Optional working directory under each iter for SP evaluations.
    work_subdir: str = "reaxff_sp"
    # How many ffield members to use (<= available). None means use all.
    max_members: Optional[int] = None

class ReactionCoordBond(BaseModel):
    name: str
    i: int
    j: int

class ReactionCoordAngle(BaseModel):
    name: str
    i: int
    j: int
    k: int

class ReactionCoordDihedral(BaseModel):
    name: str
    i: int
    j: int
    k: int
    l: int

class ReactionCoordsConfig(BaseModel):
    # User-defined reaction coordinates for Panel C.
    bonds: List[ReactionCoordBond] = Field(default_factory=list)
    angles: List[ReactionCoordAngle] = Field(default_factory=list)
    dihedrals: List[ReactionCoordDihedral] = Field(default_factory=list)

class AblationCfg(BaseModel):
    u: bool = True
    r: bool = True
    d: bool = True

class QuotasCfg(BaseModel):
    enabled: bool = False
    by_field: str = "source"
    quotas: Dict[str, int] = Field(default_factory=dict)

class ScoringConfig(BaseModel):
    lambda_u: float = 1.0
    lambda_r: float = 0.5
    lambda_d: float = 0.5
    ablation: AblationCfg = Field(default_factory=AblationCfg)
    quotas: QuotasCfg = Field(default_factory=QuotasCfg)

class Layer1Cfg(BaseModel):
    do_sp_energy: bool = True
    do_charge: bool = False

class Layer2Cfg(BaseModel):
    n_anchors: int = 2
    do_opt_freq: bool = False


class GaussianSlurmCfg(BaseModel):
    partition: str = "compute"
    time: str = "02:00:00"
    cpus_per_task: int = 8
    mem: str = "16G"
    extra_lines: List[str] = Field(default_factory=list)


class GaussianCfg(BaseModel):
    """Gaussian16 labeling configuration.

    TSGen supports three execution modes:
      - slurm: submit and poll using squeue (default)
      - local: run Gaussian directly on a workstation
      - manual: only write inputs (user submits externally); TSGen parses when logs exist
    """

    mode: Literal["slurm", "local", "manual"] = "slurm"
    route: str = "# M062X/6-311G(d) SP"
    # charge / multiplicity can be set explicitly (int) or auto-guessed from electron parity
    charge: int | Literal["auto"] = "auto"
    multiplicity: int | Literal["auto"] = "auto"
    nproc: int = 8
    mem: str = "8GB"
    g16_cmd: str = "g16"
    slurm: GaussianSlurmCfg = Field(default_factory=GaussianSlurmCfg)

class LabelingConfig(BaseModel):
    backend: Literal["mock", "gaussian_stub", "gaussian"] = "mock"
    layer1: Layer1Cfg = Field(default_factory=Layer1Cfg)
    layer2: Layer2Cfg = Field(default_factory=Layer2Cfg)
    gaussian: GaussianCfg = Field(default_factory=GaussianCfg)

class TSGenConfig(BaseModel):
    run: RunConfig
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    generators: GeneratorsConfig = Field(default_factory=GeneratorsConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    descriptor: DescriptorConfig = Field(default_factory=DescriptorConfig)
    committee: CommitteeConfig = Field(default_factory=CommitteeConfig)
    reaxff: ReaxFFBackendConfig = Field(default_factory=ReaxFFBackendConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    reaction_coords: ReactionCoordsConfig = Field(default_factory=ReactionCoordsConfig)

def load_config(path: Path) -> TSGenConfig:
    data: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TSGenConfig.model_validate(data)
