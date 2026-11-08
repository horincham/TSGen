from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tsgen.explore.config import DescriptorCfg


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class InitialConfig(StrictModel):
    ffield: Path
    labels: Path
    geo: Path
    trainset: Path
    control: Path
    params: Path | None = None
    test_labels: Path | None = None
    holdout_ids: Path | None = None


class ExistingMDConfig(StrictModel):
    ffield: Path
    trajectories: list[Path] = Field(min_length=1)


class MDConfig(StrictModel):
    inputs: dict[str, Path] = Field(default_factory=dict)
    coordinates: Literal['unwrapped']
    split_fragments: bool = True
    min_fragment_atoms: int = Field(default=2, ge=1)
    bond_tol: float = Field(default=1.2, gt=0)
    fingerprint_version: Literal['legacy', 'wl_v2'] = 'wl_v2'
    min_distance: float = Field(default=.60, gt=0)
    min_angle_deg: float = Field(default=45., ge=0, le=180)

    @model_validator(mode='after')
    def has_inputs(self):
        if any(not k.replace('_', '').isalnum() for k in self.inputs):
            raise ValueError('MD input keys must be simple names')
        return self


class AdapterConfig(StrictModel):
    command: list[str] | None = None
    timeout_seconds: int = Field(default=3600, ge=1)
    inputs: dict[str, Path] = Field(default_factory=dict)
    native: dict | None = None

    @model_validator(mode='after')
    def one_backend(self):
        if self.command and self.native:raise ValueError('Choose command or native adapter, not both')
        return self


class StopConfig(StrictModel):
    max_cycles: int = Field(default=5, ge=1)
    acquire_on_final_cycle: bool = False
    dft_budget: int = Field(default=1000, ge=1)
    labels_per_cycle: int = Field(default=40, ge=1)
    patience: int = Field(default=2, ge=1)
    new_species_max: int | None = Field(default=None, ge=0)
    novelty_median_max: float | None = Field(default=None, ge=0)
    delta_ff_max: float | None = Field(default=None, ge=0)


class OuterConfig(StrictModel):
    mode: Literal['legacy','manual','automatic'] = 'legacy'
    project_root: Path | None = None
    run_dir: Path | None = None
    out_dir: Path = Path('output')
    initial: InitialConfig
    md: MDConfig
    existing_md: ExistingMDConfig | None = None
    descriptor: DescriptorCfg = Field(default_factory=lambda: DescriptorCfg(normalization='l2'))
    adapters: dict[str, AdapterConfig] = Field(default_factory=dict)
    stop: StopConfig = Field(default_factory=StopConfig)
    explore_config: Path | None = None
    trainset_config: Path | None = None
    scan_job_budget: int = Field(default=10000,ge=1)
    poll_seconds: float = Field(default=60,gt=0)
    wait_timeout_seconds: float = Field(default=172800,gt=0)
    validation_limits: dict[str, float] = Field(default_factory=dict)
    fragment_charge: int = 0
    fragment_multiplicity: int | Literal['auto'] = 'auto'

    @model_validator(mode='after')
    def fixed_metric(self):
        if self.existing_md is not None:
            if self.mode != 'manual':
                raise ValueError('existing_md requires mode: manual')
        elif self.initial.params is None:
            raise ValueError('initial.params is required when the outer loop fits a force field')
        if self.descriptor.normalization != 'l2':
            raise ValueError('Outer metrics require fixed L2-normalized descriptors')
        unknown = set(self.adapters) - {'optimize', 'md', 'evaluate', 'acquire', 'build'}
        if unknown:
            raise ValueError(f'Unknown adapters: {unknown}')
        if any(v < 0 for v in self.validation_limits.values()):
            raise ValueError('Validation upper limits must be nonnegative')
        if isinstance(self.fragment_multiplicity,int) and self.fragment_multiplicity<1:
            raise ValueError('Fragment multiplicity must be positive')
        if self.adapters.get('md',AdapterConfig()).command and not self.md.inputs:
            raise ValueError('Automatic MD requires initial-condition inputs; manual MD may import them with its result')
        if self.mode!='legacy':
            if not self.explore_config or not self.trainset_config:
                raise ValueError('manual/automatic modes require explore_config and trainset_config for real dataset updating')
            for stage in ['acquire','build']:
                adapter=self.adapters.get(stage,AdapterConfig())
                if adapter.command or adapter.native:
                    raise ValueError(f'{stage} is performed by TSGen internally in manual/automatic mode')
            if self.mode=='manual':
                if any(self.adapters.get(s,AdapterConfig()).command or self.adapters.get(s,AdapterConfig()).native for s in ['optimize','md']):
                    raise ValueError('Manual mode imports optimizer and MD results; use automatic for executable backends')
            else:
                for stage in ['optimize','md','evaluate']:
                    adapter=self.adapters.get(stage,AdapterConfig())
                    if not (adapter.command or adapter.native):
                        raise ValueError(f'Automatic mode requires a native or command {stage} adapter')
        return self


def load_config(path: Path) -> OuterConfig:
    path = Path(path).resolve()
    cfg = OuterConfig.model_validate(yaml.safe_load(path.read_text(encoding='utf-8')))
    cfg.project_root=(path.parent/cfg.project_root).resolve() if cfg.project_root is not None else None
    base=cfg.project_root or path.parent
    def absolute(p):
        return (base / p).resolve() if p is not None else None
    cfg.run_dir = absolute(cfg.run_dir)
    cfg.out_dir = (base / cfg.out_dir).resolve()
    for key in InitialConfig.model_fields:
        setattr(cfg.initial, key, absolute(getattr(cfg.initial, key)))
    cfg.md.inputs = {k: absolute(p) for k, p in cfg.md.inputs.items()}
    if cfg.existing_md is not None:
        cfg.existing_md.ffield = absolute(cfg.existing_md.ffield)
        cfg.existing_md.trajectories = [absolute(p) for p in cfg.existing_md.trajectories]
    cfg.explore_config = absolute(cfg.explore_config)
    cfg.trainset_config = absolute(cfg.trainset_config)
    for adapter in cfg.adapters.values():
        adapter.inputs = {k:absolute(p) for k,p in adapter.inputs.items()}
        if adapter.native and adapter.native.get('template_dir'):
            adapter.native['template_dir']=str(absolute(Path(adapter.native['template_dir'])))
    return cfg
