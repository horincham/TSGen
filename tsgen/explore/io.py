from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ExploreState:
    iter: int = 0
    dft_used: int = 0

    # cumulative counters
    labeled: int = 0
    selected: int = 0
    submitted: int = 0
    failed: int = 0

    # bookkeeping lists (ids)
    labeled_ids: list[str] = None  # populated in load_state
    holdout_ids: list[str] = None  # populated in load_state

    # long-running semantics
    last_iter_submitted: int | None = None
    last_iter_collected: int | None = None
    stage: str = "BOOTSTRAP"
    recent_new_species: list[int] | None = None  # sliding window counts


@dataclass
class ExplorePaths:
    run_dir: Path

    # ---------
    # Workspace / cache dirs (runner-internal)
    # ---------
    @property
    def pool_dir(self) -> Path:
        # Candidate pool workspace (cache/index/features, etc.)
        return self.run_dir / "pool"

    @property
    def state_file(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def summary_csv(self) -> Path:
        return self.run_dir / "summary.csv"

    @property
    def labels_jsonl(self) -> Path:
        return self.run_dir / "labels.jsonl"

    @property
    def features_npy(self) -> Path:
        return self.run_dir / "features.npy"

    @property
    def pool_index_csv(self) -> Path:
        return self.run_dir / "pool_index.csv"

    # ---------
    # Exports (user-facing XYZ outputs)
    # ---------
    @property
    def exports_dir(self) -> Path:
        return self.run_dir / "exports"

    
    # ---------
    # Oracle / labels / jobs
    # ---------
    @property
    def labels_dir(self) -> Path:
        return self.run_dir / "labels"

    @property
    def labels_jsonl(self) -> Path:
        return self.labels_dir / "labels.jsonl"

    @property
    def metrics_csv(self) -> Path:
        return self.run_dir / "metrics.csv"

    def jobs_dir(self, it: int) -> Path:
        return self.iter_dir(it) / "gaussian" / "jobs"

    @property
    def filtered_pool_xyz(self) -> Path:
        """All candidates that pass physics gating (fragment-level)."""
        return self.exports_dir / "filtered_pool.xyz"

    @property
    def selected_xyz(self) -> Path:
        """All structures that were selected for DFT labeling (cumulative)."""
        return self.exports_dir / "selected_for_label.xyz"

    @property
    def labeled_ok_xyz(self) -> Path:
        """All structures with successful DFT labels (cumulative)."""
        return self.exports_dir / "labeled_ok.xyz"

    @property
    def trainset_xyz(self) -> Path:
        """Exported training set structures (seed + labeled_ok, unique by id)."""
        return self.exports_dir / "trainset_structures.xyz"

    @property
    def new_species_xyz(self) -> Path:
        """All selected structures that belong to species not present in the seed set."""
        return self.exports_dir / "selected_new_species.xyz"

    @property
    def new_species_xyz_dir(self) -> Path:
        """Per-structure XYZ exports for validated new (non-seed) structures."""
        return self.exports_dir / "new_species_xyz"


    @property
    def trainset_candidates_dir(self) -> Path:
        """De-duplicated, representative DFT-labeled structures for trainset construction."""
        return self.exports_dir / "trainset_candidates"

    @property
    def reaxff_geo_dir(self) -> Path:
        """ReaxFF/BIOGRF-style GEO export directory."""
        return self.exports_dir / "reaxff_geo"

    def iter_dir(self, it: int) -> Path:
        return self.run_dir / f"iter_{it:03d}"

    def ensure(self) -> None:
        (self.run_dir / "surrogate").mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.trainset_candidates_dir.mkdir(parents=True, exist_ok=True)
        self.reaxff_geo_dir.mkdir(parents=True, exist_ok=True)


def init_run_dir(out_root: Path, name: str) -> ExplorePaths:
    """Initialize a run directory.

    Default behavior (experiment-friendly): create a fresh, incrementing
    ``{name}_run_000X`` directory for each **new** run. This matches typical
    ML/AL software semantics where `run` starts a new experiment and `resume`
    continues an existing one.

    Set environment variable ``TSGEN_STABLE_RUNDIR=1`` to instead use a stable
    directory ``out_root/name`` (useful for service-like deployments).
    """
    import os

    out_root.mkdir(parents=True, exist_ok=True)

    if os.environ.get("TSGEN_STABLE_RUNDIR", "0") not in ("0", "false", "False", ""):
        run_dir = out_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        return ExplorePaths(run_dir=run_dir)

    existing = sorted([p for p in out_root.glob(f"{name}_run_*") if p.is_dir()])
    next_id = 1
    if existing:
        try:
            next_id = max(int(p.name.split("_")[-1]) for p in existing) + 1
        except Exception:
            next_id = len(existing) + 1
    run_dir = out_root / f"{name}_run_{next_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return ExplorePaths(run_dir=run_dir)


def load_state(path: Path) -> ExploreState:
    if not path.exists():
        return ExploreState()
    data = json.loads(path.read_text(encoding="utf-8"))
    # backward compatible defaults
    if data.get('labeled_ids') is None:
        data['labeled_ids'] = []
    if data.get('holdout_ids') is None:
        data['holdout_ids'] = []
    if data.get('recent_new_species') is None:
        data['recent_new_species'] = []
    if data.get('stage') is None:
        data['stage'] = 'BOOTSTRAP'
    return ExploreState(**data)


def save_state(path: Path, state: ExploreState) -> None:
    path.write_text(json.dumps(asdict(state), indent=2) + "\n", encoding="utf-8")


def write_config_snapshot(run_dir: Path, cfg_text: str) -> None:
    (run_dir / "config.yaml").write_text(cfg_text, encoding="utf-8")