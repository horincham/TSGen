from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True, order=True)
class ParameterCoordinate:
    section: int
    block: int
    item: int

    @property
    def parameter_id(self) -> str:
        return f"{self.section}:{self.block}:{self.item}"

    @classmethod
    def parse(cls, value: str) -> "ParameterCoordinate":
        try:
            section, block, item = (int(part) for part in value.split(":"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid parameter coordinate {value!r}; expected section:block:item") from exc
        return cls(section, block, item)


@dataclass(frozen=True)
class ParameterDefinition:
    coordinate: ParameterCoordinate
    section_name: str
    block_key: tuple[str, ...]
    parameter_name: str
    initial_value: float
    line_index: int
    token_start: int
    token_end: int


@dataclass(frozen=True)
class ParameterBound:
    coordinate: ParameterCoordinate
    delta: float
    lower_bound: float
    upper_bound: float
    source: str = ""


@dataclass(frozen=True)
class TrainingItem:
    item_index: int
    section: Literal["ENERGY", "GEOMETRY", "CHARGE"]
    structure_ids: tuple[str, ...]
    item_label: str
    reference_value: float
    weight: float


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    normalized_values: tuple[float, ...]
    physical_values: tuple[float, ...]
    is_initial: bool = False


@dataclass(frozen=True)
class ItemLoss:
    item_index: int
    section: str
    structure_id: str
    item_label: str
    reaxff_value: float
    reference_value: float
    weight: float
    squared_error: float


@dataclass(frozen=True)
class EvaluationResult:
    sample_id: str
    status: Literal["success", "failed", "timeout", "parse_error"]
    total_loss: float | None
    section_losses: dict[str, float] = field(default_factory=dict)
    item_losses: tuple[ItemLoss, ...] = ()
    elapsed_seconds: float = 0.0
    failure_reason: str = ""
    return_code: int | None = None
    ffield_sha256: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SensitivityResult:
    parameter_id: str
    target: str
    hsic_mean: float
    hsic_std: float
    hsic_ci_low: float
    hsic_ci_high: float

