from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import pandas as pd
from tsgen.review_bundle.templates import README_TEMPLATE

@dataclass(frozen=True)
class KitInput:
    candidates_xyz: str
    manifest_jsonl: str
    features_csv: str
    required_columns: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates_xyz": self.candidates_xyz,
            "manifest_jsonl": self.manifest_jsonl,
            "features_csv": self.features_csv,
            "required_columns": self.required_columns,
        }

def write_review_bundle(bundle_dir: Path, candidates_xyz: Path, manifest_jsonl: Path, features_csv: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    kit = KitInput(
        candidates_xyz=str(candidates_xyz),
        manifest_jsonl=str(manifest_jsonl),
        features_csv=str(features_csv),
        required_columns={
            "id": "Unique candidate id (must match candidates.xyz comment id=...)",
            "score_u": "Committee uncertainty u_E(x)",
            "score_r": "Reaction relevance r(x)",
            "score_d": "Diversity proxy d(x)",
            "S": "Final score S(x)",
            "PCA1": "PCA coordinate 1 for visualization",
            "PCA2": "PCA coordinate 2 for visualization",
            "filter_flags": "Filter status / reasons",
        },
    )
    (bundle_dir / "kit_input.json").write_text(json.dumps(kit.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    (bundle_dir / "README.md").write_text(README_TEMPLATE, encoding="utf-8")

def read_decision_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out

def apply_decisions(features: pd.DataFrame, decisions: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    keep, drop = [], []
    dmap = {d["id"]: d for d in decisions if "id" in d and "action" in d}
    for cid in features["id"].astype(str).tolist():
        if cid not in dmap:
            continue
        action = str(dmap[cid]["action"]).lower()
        if action == "keep":
            keep.append(cid)
        elif action == "drop":
            drop.append(cid)
    return keep, drop
