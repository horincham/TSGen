from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, List, Dict, Any
import json

Stage = Literal[
    "INIT",
    "GENERATE",
    "FILTER",
    "FEATURIZE",
    "SCORE_SELECT",
    "GATE_WAIT",
    "LABEL_TASKS",
    "LABEL_WAIT",
    "LABEL_PARSE",
    "EVAL_BENCH",
    "DONE",
]

@dataclass
class RunState:
    stage: Stage = "INIT"
    iteration: int = 0
    dft_used: int = 0
    selected_ids_total: List[str] = None
    labeled_ids_total: List[str] = None

    def __post_init__(self):
        if self.selected_ids_total is None:
            self.selected_ids_total = []
        if self.labeled_ids_total is None:
            self.labeled_ids_total = []

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "RunState":
        return RunState(
            stage=d.get("stage", "INIT"),
            iteration=int(d.get("iteration", 0)),
            dft_used=int(d.get("dft_used", 0)),
            selected_ids_total=list(d.get("selected_ids_total", [])),
            labeled_ids_total=list(d.get("labeled_ids_total", [])),
        )

def save_state(path: Path, state: RunState) -> None:
    path.write_text(json.dumps(state.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")

def load_state(path: Path) -> RunState:
    return RunState.from_json(json.loads(path.read_text(encoding="utf-8")))
