from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional
import json

@dataclass
class LineageRecord:
    id: str
    parent_id: Optional[str]
    source: str
    propose_params: Dict[str, Any]
    structure_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def write_lineage_json(path: Path, recs: Dict[str, "LineageRecord"]) -> None:
    out = {k: v.to_dict() for k, v in recs.items()}
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
