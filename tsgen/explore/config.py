# src/tsgen/explore/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, Annotated

import yaml
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Run / general
# =============================================================================

class RunCfg(BaseModel):
    out_dir: str = "outputs_explore"
    name: str = "explore"
    seed: int = 0
    max_iters: int = 5
    dft_budget: int = 50



# =============================================================================
# Long-running loop control (auto vs step)
# =============================================================================

class LoopCfg(BaseModel):
    mode: Literal["step", "auto"] = "step"  # default: keep existing behavior
    poll_seconds: int = 60
    max_wait_seconds: int = 86400  # 24h
    require_all_ok: bool = False
    min_ok_frac: float = 0.8


class StopCfg(BaseModel):
    # Hard stops
    max_iters: Optional[int] = None  # if None, use run.max_iters
    dft_budget: Optional[int] = None  # if None, use run.dft_budget

    # Soft stops (optional; evaluated only when values provided)
    sigma_mean_below: Optional[float] = None  # kcal/mol
    no_new_species_rounds: Optional[int] = None


class StagesCfg(BaseModel):
    # Logical stage thresholds (runner decides stage each iteration)
    bootstrap_min_labels: int = 20
    bootstrap_min_species: int = 5

    # Stage-specific acquisition overrides
    explore_method: Optional[Literal["sigma", "ucb"]] = None
    explore_ucb_kappa: Optional[float] = None

    refine_method: Optional[Literal["sigma", "ucb"]] = None
    refine_ucb_kappa: Optional[float] = None

# =============================================================================
# Physics gating (filters)
# =============================================================================

class GatingCfg(BaseModel):
    enabled: bool = True
    min_dist: float = 0.7  # Å
    angle_local_only: bool = True
    bonded_only: bool = True
    extra: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Candidate pool sources
# =============================================================================

class TrajXYZSourceCfg(BaseModel):
    type: Literal["traj_xyz"] = "traj_xyz"
    path: str
    stride: int = 1
    max_frames: Optional[int] = None
    use_comment_energy: bool = False


class SeedXYZSourceCfg(BaseModel):
    type: Literal["seed_xyz"] = "seed_xyz"
    path: str
    use_comment_energy: bool = False


class SeedDirSourceCfg(BaseModel):
    type: Literal["seed_dir"] = "seed_dir"
    path: str
    pattern: str = "*.xyz"
    recursive: bool = False
    use_comment_energy: bool = False


class TrajDirSourceCfg(BaseModel):
    type: Literal["traj_dir"] = "traj_dir"
    path: str
    pattern: str = "*.xyz"
    recursive: bool = False
    stride: int = 1
    max_frames: Optional[int] = None
    use_comment_energy: bool = False


PoolSourceCfg = Union[
    TrajXYZSourceCfg,
    SeedXYZSourceCfg,
    SeedDirSourceCfg,
    TrajDirSourceCfg,
]

DiscriminatedPoolSourceCfg = Annotated[PoolSourceCfg, Field(discriminator="type")]



class PoolChemistryFilterCfg(BaseModel):
    # Optional pre-active-learning chemistry filter.  This is applied while
    # building the candidate pool, so DFT budget is not spent on structures that
    # can never enter the curated ReaxFF GEO under the intended composition.
    enabled: bool = False
    allowed_elements: Optional[List[str]] = None
    required_elements: List[str] = Field(default_factory=list)
    exclude_elements: List[str] = Field(default_factory=list)
    min_element_counts: Dict[str, int] = Field(default_factory=dict)
    max_element_counts: Dict[str, int] = Field(default_factory=dict)
    exclude_odd_electron: bool = False


class PoolBalanceCfg(BaseModel):
    # Optional pool balancing to prevent long-lived trajectory species from
    # dominating active-learning selection.  Frames are sampled evenly across
    # each group in trajectory order.
    enabled: bool = False
    by: Literal["species", "formula"] = "species"
    max_per_group: int = Field(default=50, ge=1)
    method: Literal["evenly_spaced", "fps"] = "evenly_spaced"


class PoolDeduplicationCfg(BaseModel):
    # Invariant, element-pair distance heuristic; not a graph-isomorphism proof.
    enabled: bool = False
    fingerprint_tol: float = Field(default=0.01, ge=0.0)


class CandidatePoolCfg(BaseModel):
    sources: List[DiscriminatedPoolSourceCfg] = Field(default_factory=list)

    # used by pool.py
    split_fragments: bool = True
    min_fragment_atoms: int = 2

    # BACKWARD-COMPAT: some code expects cfg.candidate_pool.gating
    gating: Optional[GatingCfg] = None

    # Optional pre-AL controls; export.selection remains the authoritative
    # final trainset/GEO filter, but these reduce wasted DFT budget.
    chemistry_filter: PoolChemistryFilterCfg = Field(default_factory=PoolChemistryFilterCfg)
    balance: PoolBalanceCfg = Field(default_factory=PoolBalanceCfg)
    deduplication: PoolDeduplicationCfg = Field(default_factory=PoolDeduplicationCfg)


# =============================================================================
# Descriptor / species
# =============================================================================

class PCACfg(BaseModel):
    enabled: bool = False
    variance_ratio: float = Field(default=0.999, gt=0.0, lt=1.0)


class DescriptorCfg(BaseModel):
    # descriptor backend used by surrogate/features.py
    # SOAP is the mainstream default for descriptor-space MLIP/AL analysis.
    backend: Literal["soap", "coulomb_eigs"] = "soap"
    # fixed-length descriptor size control for Coulomb fallback
    max_atoms: int = 256
    # DScribe SOAP settings.  Species should cover the intended chemistry.
    soap_species: List[str] = Field(default_factory=lambda: ["H", "C", "N", "O", "F", "Si", "P", "S", "Cl"])
    soap_r_cut: float = 5.0
    soap_n_max: int = 8
    soap_l_max: int = 6
    soap_sigma: float = 0.5
    soap_periodic: bool = False
    soap_pooling: Literal["mean", "mean_std"] = "mean_std"
    normalization: Literal["none", "l2"] = "none"
    pca: PCACfg = Field(default_factory=PCACfg)


class SpeciesQuotaCfg(BaseModel):
    # runner.py uses cfg.species.quota.min_per_species
    min_per_species: int = 0
    new_species_boost: float = 1.5


class SpeciesCfg(BaseModel):
    enable: bool = True
    bond_tol: float = 1.2
    quota: SpeciesQuotaCfg = Field(default_factory=SpeciesQuotaCfg)


# =============================================================================
# Surrogate
# =============================================================================

class RidgeEnsembleCfg(BaseModel):
    n_models: int = 16
    bootstrap: bool = True
    alpha: float = 1e-6


class GPCfg(BaseModel):
    kernel: Literal["rbf", "matern", "matern32"] = "matern32"
    alpha: float = 1e-6
    max_train: int = 2000
    normalize_y: bool = True


class SurrogateCfg(BaseModel):
    kind: Literal["ridge_ensemble", "gp"] = "gp"
    # NOTE: do NOT use Literal here.
    # Users frequently want to train on different label columns (e.g., e_kcalmol_rel).
    # If we used Literal, pydantic would reject those values before validators run.
    # Supported values (runner resolves aliases):
    #   - dft_energy (default) -> e_kcalmol_abs
    #   - e_kcalmol_abs
    #   - e_kcalmol_rel
    #   - e_hartree
    #   - energy (alias of dft_energy)
    target: str = "dft_energy"
    ridge_ensemble: RidgeEnsembleCfg = Field(default_factory=RidgeEnsembleCfg)
    gp: GPCfg = Field(default_factory=GPCfg)

    @field_validator("target")
    @classmethod
    def _norm_target(cls, v: str) -> str:
        vv = str(v).strip()
        # Backward/forward compatible aliases
        if vv == "energy":
            return "dft_energy"
        # Common synonyms
        if vv in {"dft_energy_abs", "dft_energy_kcal", "dft_energy_kcalmol"}:
            return "dft_energy"
        if vv in {"dft_energy_rel", "rel", "relative"}:
            return "e_kcalmol_rel"
        if vv in {"hartree", "energy_hartree"}:
            return "e_hartree"
        return vv


# =============================================================================
# Acquisition / selection
# =============================================================================

class AcquisitionWeightsCfg(BaseModel):
    u: float = 1.0
    d: float = 0.3
    r: float = 0.0
    new_species: float = 0.5
    penalty: float = 0.2


class DiversityCfg(BaseModel):
    method: Literal["farthest_in_descriptor", "fps", "none"] = "farthest_in_descriptor"
    k_ref: int = 2000


class AcquisitionCfg(BaseModel):
    k_select: int = 20
    method: Literal["sigma", "ucb"] = "sigma"
    ucb_kappa: float = 2.0
    preselect_factor: int = 20
    weights: AcquisitionWeightsCfg = Field(default_factory=AcquisitionWeightsCfg)
    diversity: DiversityCfg = Field(default_factory=DiversityCfg)




# =============================================================================
# Export / trainset post-processing
# =============================================================================


class PostOptimizationDedupCfg(BaseModel):
    # Final mother-molecule curation after optimization, not AL/scan-point curation.
    enabled: bool = True
    # One conformer per final connectivity/stereo/electronic identity by default.
    # Explicit "geometry" restores the old near-identical-geometry filter.
    method: Literal["molecular_identity", "geometry"] = "molecular_identity"
    # Compare only inside the same formula by default.  Use "global" only for
    # very small datasets; different formulas can never be duplicates in normal
    # chemistry.
    scope: Literal["within_formula", "within_family", "global"] = "within_formula"
    fingerprint_tol: float = 0.03  # Used as a cutoff only with method="geometry".
    keep: Literal["lowest_energy", "first"] = "lowest_energy"


class ChemicalFamilyFilterCfg(BaseModel):
    # Heuristic chemistry-family representative selection for final GEO only.
    # It is deliberately post-optimization by default: it should curate the final
    # ReaxFF GEO, not prematurely suppress useful AL/DFT exploration.
    enabled: bool = True
    stage: Literal["post_opt", "off"] = "post_opt"
    method: Literal["heuristic_bond_environment"] = "heuristic_bond_environment"
    max_per_family: int = 3
    # Optional cap per formula after duplicate removal.  Keep null if isomers or
    # distinct bonding environments with the same formula should be retained.
    max_per_formula: Optional[int] = None
    protected_formulas: List[str] = Field(default_factory=lambda: [
        "H2", "O2", "H2O", "CO", "CO2", "CH4", "C2H2", "C2H4", "C2H6"
    ])
    protected_families: List[str] = Field(default_factory=list)


class ExportSelectionCfg(BaseModel):
    # Maximum representative structures kept per species for trainset candidates.
    max_per_species: int = 5
    # Optional global cap after species balancing.
    max_total: Optional[int] = None
    # Drop structures whose per-species relative energy is above this threshold.
    # Set to null in YAML to disable.
    max_rel_kcalmol: Optional[float] = 250.0
    # Within each species, select representatives across energy bins.
    energy_bins: int = 3
    # Same-species structures with pair-distance fingerprint distance <= this
    # threshold are treated as redundant for export.
    fingerprint_tol: float = 0.05

    # -------------------------------------------------------------------------
    # Chemistry filters for curated trainset/GEO export.
    # These filters affect post-processing only; they do not change the active-
    # learning candidate pool or the already completed Gaussian labels.
    # -------------------------------------------------------------------------
    # Optional filter for final trainset/GEO export.  This does not change the
    # active-learning selection itself; it only prevents open-shell/radical
    # structures from entering the curated ReaxFF GEO products.
    exclude_open_shell: bool = False
    # If provided, only these spin multiplicities are allowed into the curated
    # trainset/GEO export.  Example: [1] keeps only singlet/closed-shell jobs.
    allowed_multiplicities: Optional[List[int]] = None
    # Optional species_id regex filters.  Any matching species is excluded from
    # the curated trainset/GEO export.
    exclude_species_regex: List[str] = Field(default_factory=list)

    # Element/composition constraints. Examples:
    #   allowed_elements: [C, H] + max_element_counts: {C: 9}
    #       keeps hydrocarbons up to C9, and also C0/H-only species such as H2.
    #   allowed_elements: [C, H, O] + max_element_counts: {C: 9, O: 2}
    #       keeps C/H/O combustion-style molecules up to C9 and O2.
    #   required_elements: [C]
    #       requires at least one carbon atom, excluding H2/H2O even if allowed.
    allowed_elements: Optional[List[str]] = None
    required_elements: List[str] = Field(default_factory=list)
    exclude_elements: List[str] = Field(default_factory=list)
    min_element_counts: Dict[str, int] = Field(default_factory=dict)
    max_element_counts: Dict[str, int] = Field(default_factory=dict)
    include_formula_regex: List[str] = Field(default_factory=list)
    exclude_formula_regex: List[str] = Field(default_factory=list)

    # Final GEO curation after optimization.
    post_opt_dedup: PostOptimizationDedupCfg = Field(default_factory=PostOptimizationDedupCfg)
    chemical_family_filter: ChemicalFamilyFilterCfg = Field(default_factory=ChemicalFamilyFilterCfg)


class ExportOptimizationCfg(BaseModel):
    enabled: bool = False
    # Usually an optimization route. Existing AL labeling may remain SP.
    route: str = "# M062X/6-311G(d) Opt"
    # If null, reuse labeling.gaussian.mode.
    mode: Optional[Literal["manual", "local", "slurm"]] = None
    # In slurm mode, submit optimization jobs automatically.
    submit: bool = True
    # If true, export_trainset_products blocks and polls optimization logs until
    # all opt jobs finish or max_wait_seconds is reached. This gives a one-shot
    # user experience for small/medium trainset exports.
    wait_for_completion: bool = False
    poll_seconds: int = 60
    max_wait_seconds: int = 86400
    require_all_ok: bool = True


class ExportGeoCfg(BaseModel):
    enabled: bool = False
    filename: str = "trainset.geo"
    prefer_optimized: bool = True
    # If optimization is not enabled or opt jobs are incomplete, write GEO from
    # the AL-selected single-point geometries instead of writing no GEO.
    allow_unoptimized_fallback: bool = True


class ExportCfg(BaseModel):
    # validated: keep current behavior only.
    # trainset: additionally write de-duplicated trainset candidates.
    # geo / opt_geo / trainset_geo: additionally write ReaxFF GEO products.
    scheme: Literal["validated", "trainset", "geo", "opt_geo", "trainset_geo"] = "validated"
    # labeled_all: include seed and non-seed labeled records.
    # non_seed_labeled: only non-seed records.
    source: Literal["labeled_all", "non_seed_labeled"] = "non_seed_labeled"
    selection: ExportSelectionCfg = Field(default_factory=ExportSelectionCfg)
    optimization: ExportOptimizationCfg = Field(default_factory=ExportOptimizationCfg)
    geo: ExportGeoCfg = Field(default_factory=ExportGeoCfg)


# =============================================================================
# Labeling (Gaussian)
# =============================================================================

class SlurmCfg(BaseModel):
    partition: Optional[str] = None
    time: Optional[str] = None
    cpus_per_task: Optional[int] = None
    mem: Optional[str] = None
    extra_lines: List[str] = Field(default_factory=list)


class GaussianCfg(BaseModel):
    mode: Literal["manual", "local", "slurm"] = "manual"
    route: str = "# M062X/6-311G(d) SP"
    charge: Union[int, Literal["auto"]] = "auto"
    multiplicity: Union[int, Literal["auto"]] = "auto"
    nproc: int = 8
    mem: str = "16GB"
    g16_cmd: str = "g16"
    slurm: SlurmCfg = Field(default_factory=SlurmCfg)


class LabelingCfg(BaseModel):
    backend: Literal["gaussian", "oracle_traj", "mock"] = "gaussian"
    gaussian: GaussianCfg = Field(default_factory=GaussianCfg)


# =============================================================================
# Anchors / evaluation
# =============================================================================

class AnchorsCfg(BaseModel):
    enabled: bool = True
    n_anchors: int = 50
    min_labeled: int = 20  # ensure at least this many holdout labels for evaluation
    strategy: Literal["random", "fps", "stratified_by_species"] = "stratified_by_species"
    holdout_frac: float = 0.2


# =============================================================================
# Top-level Explore config
# =============================================================================

class ExploreCfg(BaseModel):
    run: RunCfg = Field(default_factory=RunCfg)
    candidate_pool: CandidatePoolCfg = Field(default_factory=CandidatePoolCfg)
    descriptor: DescriptorCfg = Field(default_factory=DescriptorCfg)
    species: SpeciesCfg = Field(default_factory=SpeciesCfg)

    # RECOMMENDED: gating at top-level
    gating: GatingCfg = Field(default_factory=GatingCfg)

    surrogate: SurrogateCfg = Field(default_factory=SurrogateCfg)
    acquisition: AcquisitionCfg = Field(default_factory=AcquisitionCfg)
    labeling: LabelingCfg = Field(default_factory=LabelingCfg)
    anchors: AnchorsCfg = Field(default_factory=AnchorsCfg)
    loop: LoopCfg = Field(default_factory=LoopCfg)
    stop: StopCfg = Field(default_factory=StopCfg)
    stages: StagesCfg = Field(default_factory=StagesCfg)
    export: ExportCfg = Field(default_factory=ExportCfg)

    extra: Dict[str, Any] = Field(default_factory=dict)


def load_explore_config(path: str | Path) -> ExploreCfg:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Explore config not found: {p}")

    data = yaml.safe_load(p.read_text())
    if data is None:
        data = {}

    # optional alias
    if isinstance(data, dict) and "explore" in data and isinstance(data["explore"], dict):
        data = data["explore"]

    cfg = ExploreCfg.model_validate(data)

    # BACKWARD-COMPAT:
    # If user put gating under candidate_pool.gating, merge/override into top-level gating.
    if cfg.candidate_pool.gating is not None:
        cfg.gating = cfg.candidate_pool.gating

    return cfg


def save_explore_config(cfg: ExploreCfg, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    obj = cfg.model_dump(mode="python")
    p.write_text(yaml.safe_dump(obj, sort_keys=False))
