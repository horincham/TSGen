from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from tsgen.data.xyz import Frame, read_xyz_multiframe, write_xyz_multiframe
from tsgen.trainset.config import TrainsetBuildCfg, save_trainset_config


BAD_ID_CHARS = re.compile(r"[^A-Za-z0-9_]")


def sanitize_id(s: str, max_len: int = 80) -> str:
    out = BAD_ID_CHARS.sub("_", str(s).strip())
    out = re.sub(r"_+", "_", out).strip("_")
    if not out:
        out = "structure"
    # ReaxFF trainset.in uses +, -, / as operators; keep IDs simple and unique.
    if out[0].isdigit():
        out = "m_" + out
    return out[:max_len]


def next_run_dir(out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    prefix = f"{sanitize_id(name)}_run_"
    for p in out_dir.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            try:
                existing.append(int(p.name.split("_")[-1]))
            except Exception:
                pass
    idx = max(existing, default=0) + 1
    return out_dir / f"{sanitize_id(name)}_run_{idx:04d}"


class TrainsetPaths:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.config_file = self.run_dir / "config.yaml"
        self.exports_dir = self.run_dir / "exports" / "reaxff_trainset"
        self.structures_dir = self.exports_dir / "structures"
        self.eq_structures_dir = self.structures_dir / "equilibrium"
        self.bond_structures_dir = self.structures_dir / "bond_scans"
        self.angle_structures_dir = self.structures_dir / "angle_scans"
        self.torsion_structures_dir = self.structures_dir / "torsion_scans"
        self.jobs_dir = self.exports_dir / "gaussian_jobs"
        self.tables_dir = self.exports_dir / "tables"
        self.figures_dir = self.exports_dir / "figures"
        self.geo_file = self.exports_dir / "geo"
        self.bgf_dir = self.exports_dir / "bgf"
        self.trainset_in = self.exports_dir / "trainset.in"
        self.models_in = self.exports_dir / "models.in"
        self.report_md = self.exports_dir / "report.md"
        self.summary_json = self.exports_dir / "summary.json"

    def ensure(self) -> None:
        for p in [
            self.run_dir, self.exports_dir, self.structures_dir, self.eq_structures_dir,
            self.bond_structures_dir, self.angle_structures_dir, self.torsion_structures_dir,
            self.jobs_dir, self.tables_dir, self.figures_dir, self.bgf_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


def resolve_input_structures(cfg: TrainsetBuildCfg) -> List[Path]:
    icfg = cfg.input
    if icfg.source == "explore_final":
        if not icfg.run_dir:
            raise ValueError("input.run_dir is required when input.source == explore_final")
        base = Path(icfg.run_dir)
        sdir = Path(icfg.structures_dir)
        if not sdir.is_absolute():
            sdir = base / sdir
    elif icfg.source == "xyz_dir":
        sdir = Path(icfg.xyz_dir or icfg.structures_dir)
    else:
        sdir = Path(icfg.structures_dir)
    if not sdir.exists():
        raise FileNotFoundError(f"Input structures directory does not exist: {sdir}")
    paths = sorted(sdir.rglob(icfg.pattern) if icfg.recursive else sdir.glob(icfg.pattern))
    return [p for p in paths if p.is_file()]


def read_first_frame(path: Path) -> Frame:
    frames = read_xyz_multiframe(path)
    if not frames:
        raise ValueError(f"No xyz frame in {path}")
    return frames[0]


def write_single_xyz(path: Path, frame: Frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_xyz_multiframe(path, [frame])


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_dataframe(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def read_frames_from_reaxff_geo(path: Path) -> List[tuple[str, Frame]]:
    """Read common standalone ReaxFF geometry inputs.

    Supports BIOGRF/XTLGRF BGF-like blocks written by TSGen/ReaxFF and simple legacy .geo
    Cartesian blocks such as ``C Ethyl_radical`` followed by ``idx sym x y z`` lines.
    Returns ``[(identifier, Frame), ...]``.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    out: List[tuple[str, Frame]] = []

    # BGF/BIOGRF style: one or more blocks delimited by END.
    if any(line.strip().startswith(("BIOGRF", "XTLGRF")) for line in lines):
        block: list[str] = []
        for line in lines:
            if not block and not line.strip():
                continue
            block.append(line)
            if line.strip() == "END":
                rec = _parse_bgf_block(block, default_name=f"model_{len(out)+1:04d}")
                if rec is not None:
                    out.append(rec)
                block = []
        if block:
            rec = _parse_bgf_block(block, default_name=f"model_{len(out)+1:04d}")
            if rec is not None:
                out.append(rec)
        return out

    # Legacy simple .geo. Blocks begin with a control character and an identifier.
    cur_name: Optional[str] = None
    cur_symbols: list[str] = []
    cur_coords: list[list[float]] = []
    def flush():
        nonlocal cur_name, cur_symbols, cur_coords
        if cur_name and cur_symbols:
            import numpy as np
            ident = sanitize_id(cur_name, max_len=80)
            out.append((ident, Frame(symbols=list(cur_symbols), coords=np.array(cur_coords, dtype=float), comment=f"id={ident} source={path.name}")))
        cur_name = None; cur_symbols = []; cur_coords = []

    for raw in lines + [""]:
        line = raw.strip()
        if not line:
            flush(); continue
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 1 and parts[0] in {"C", "F", "1", "D", "H", "5", "I"} and not parts[1].replace('.', '', 1).isdigit():
            flush()
            cur_name = parts[1]
            continue
        if len(parts) >= 5 and parts[0].lstrip("+-").isdigit():
            try:
                sym = parts[1]
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                cur_symbols.append(sym); cur_coords.append([x, y, z])
            except Exception:
                pass
    return out


def _is_float(x: str) -> bool:
    try:
        float(x); return True
    except Exception:
        return False


def _parse_bgf_block(block: list[str], default_name: str) -> Optional[tuple[str, Frame]]:
    import numpy as np
    name = default_name
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in block:
        st = line.strip()
        if st.startswith("DESCRP"):
            parts = st.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                name = parts[1].strip()
        elif st.startswith(("HETATM", "ATOM")):
            parts = st.split()
            if len(parts) >= 6:
                try:
                    sym = parts[2]
                    # Three common variants:
                    #   HETATM idx C x y z C ...
                    #   HETATM idx C MOL res x y z C ...
                    #   HETATM idx C RES A res x y z C ... (TSGen writer)
                    if len(parts) >= 9 and not _is_float(parts[3]) and not _is_float(parts[4]):
                        x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
                    elif len(parts) >= 8 and not _is_float(parts[3]) and _is_float(parts[4]):
                        x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
                    else:
                        x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                    symbols.append(sym); coords.append([x, y, z])
                except Exception:
                    continue
    if not symbols:
        return None
    ident = sanitize_id(name, max_len=80)
    return ident, Frame(symbols=symbols, coords=np.array(coords, dtype=float), comment=f"id={ident} source=geo")


def load_input_frames(cfg: TrainsetBuildCfg) -> List[tuple[str, Path, Frame]]:
    """Resolve input into named frames, regardless of source type."""
    icfg = cfg.input
    if icfg.source == "geo_file":
        if not icfg.geo_file:
            raise ValueError("input.geo_file is required when input.source == geo_file")
        gp = Path(icfg.geo_file)
        if not gp.exists():
            raise FileNotFoundError(f"Input geo file does not exist: {gp}")
        return [(name, gp, fr) for name, fr in read_frames_from_reaxff_geo(gp)]
    paths = resolve_input_structures(cfg)
    return [(p.stem, p, read_first_frame(p)) for p in paths]
