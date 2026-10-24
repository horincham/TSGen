from __future__ import annotations
from typing import Protocol, List, Dict, Any
from pathlib import Path
from tsgen.data.xyz import Frame

class LabelBackend(Protocol):
    def generate_tasks(self, task_dir: Path, selected: List[Frame], ids: List[str], layer_cfg: Dict[str, Any]) -> None:
        ...
    def run_and_wait(self, task_dir: Path) -> None:
        ...
    def parse_results(self, task_dir: Path) -> Dict[str, Dict[str, Any]]:
        ...
