from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class RunPaths:
    root: Path
    run_id: int

    @property
    def run_dir(self) -> Path:
        return self.root / f"run_{self.run_id:04d}"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def state_file(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def summary_csv(self) -> Path:
        return self.run_dir / "summary.csv"

    def iter_dir(self, it: int) -> Path:
        return self.run_dir / f"iter_{it:03d}"

    def ensure_base(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def from_run_dir(run_dir: Path) -> "RunPaths":
        name = run_dir.name
        if not name.startswith("run_"):
            raise ValueError(f"Invalid run_dir: {run_dir}")
        run_id = int(name.split("_")[1])
        return RunPaths(root=run_dir.parent, run_id=run_id)
