from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable, Optional
import numpy as np
import re

@dataclass(frozen=True)
class Frame:
    symbols: List[str]
    coords: np.ndarray  # (N,3)
    comment: str

    def get_id(self) -> Optional[str]:
        m = re.search(r"\bid=([^\s]+)", self.comment)
        return m.group(1) if m else None

def read_xyz_multiframe(path: Path) -> List[Frame]:
    text = path.read_text(encoding="utf-8").strip().splitlines()
    frames: List[Frame] = []
    i = 0
    while i < len(text):
        n = int(text[i].strip())
        comment = text[i + 1].rstrip("\n")
        symbols: List[str] = []
        coords: List[List[float]] = []
        for k in range(n):
            parts = text[i + 2 + k].split()
            symbols.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        frames.append(Frame(symbols=symbols, coords=np.array(coords, dtype=float), comment=comment))
        i += 2 + n
    return frames

def write_xyz_multiframe(path: Path, frames: Iterable[Frame]) -> None:
    lines: List[str] = []
    for fr in frames:
        n = len(fr.symbols)
        lines.append(str(n))
        lines.append(fr.comment)
        for s, (x, y, z) in zip(fr.symbols, fr.coords):
            lines.append(f"{s:2s} {x: .6f} {y: .6f} {z: .6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
