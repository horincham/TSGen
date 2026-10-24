from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
import json

@dataclass
class ManifestRow:
    id: str
    iter: int
    source: str
    propose_params: Dict[str, Any]
    hash: str
    status: str
    xyz_frame_index: int
    parent_id: Optional[str] = None

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

def write_manifest(path: Path, rows: Iterable[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.to_jsonl() + "\n")

def read_manifest(path: Path) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        rows.append(ManifestRow(**d))
    return rows
