from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import json
import numpy as np
from tsgen.data.xyz import Frame

@dataclass
class MockLabelBackend:
    seed: int = 0

    def generate_tasks(self, task_dir: Path, selected: List[Frame], ids: List[str], layer_cfg: Dict[str, Any]) -> None:
        task_dir.mkdir(parents=True, exist_ok=True)
        meta = {"n": len(ids), "ids": ids, "layer_cfg": layer_cfg}
        (task_dir / "tasks.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def run_and_wait(self, task_dir: Path) -> None:
        return

    def parse_results(self, task_dir: Path) -> Dict[str, Dict[str, Any]]:
        meta = json.loads((task_dir / "tasks.json").read_text(encoding="utf-8"))
        ids = meta["ids"]
        rng = np.random.default_rng(self.seed)
        results = {}
        for cid in ids:
            results[cid] = {"energy": float(rng.normal(loc=-10.0, scale=0.2))}
        (task_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        return results
