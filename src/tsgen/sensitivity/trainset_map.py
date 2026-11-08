from __future__ import annotations

import math
from pathlib import Path
import re

from tsgen.sensitivity.models import TrainingItem
from tsgen.trainset.reaxff_io import LEGACY_BGF_ATOM_FORMAT, LEGACY_BGF_CONECT_FORMAT


ENERGY_ID_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*/")
SECTION_ENDS = {
    "ENERGY": "ENDENERGY",
    "GEOMETRY": "ENDGEOMETRY",
    "CHARGE": "ENDCHARGE",
}


def _fixed_column_bgf_atom(
    record: str,
    atom_index: int,
    atom_label: str,
    x: float,
    y: float,
    z: float,
    ff_type: str,
    flag1: int = 0,
    flag2: int = 0,
    charge: float = 0.0,
) -> str:
    """Format the exact pre-400 BIOGRF atom layout read by legacy ReaxFF."""
    if not 0 <= atom_index <= 99999:
        raise ValueError(f"BIOGRF atom index is outside the i5 field: {atom_index}")
    return (
        f"{record[:6]:6s} {atom_index:5d} {atom_label[:5]:<5s} "
        f"RES A {1:<5d}{x:10.5f}{y:10.5f}{z:10.5f} "
        f"{ff_type[:5]:<5s}{flag1:3d}{flag2:2d} {charge:8.5f}"
    )


def write_reaxff_compatible_geo(source: str | Path, destination: str | Path) -> int:
    """Normalize TSGen geo blocks for the fixed-column legacy ReaxFF reader."""
    source_path = Path(source)
    output_path = Path(destination)
    text = source_path.read_text(encoding="utf-8")
    converted = 0
    output_lines: list[str] = []
    has_atom_format = False
    has_conect_format = False
    has_atoms = False
    for line_number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if stripped.startswith(("BIOGRF", "XTLGRF")):
            has_atom_format = False
            has_conect_format = False
            has_atoms = False
        elif stripped.startswith("FORMAT ATOM"):
            raw = LEGACY_BGF_ATOM_FORMAT
            has_atom_format = True
        elif stripped.startswith("FORMAT CONECT"):
            raw = LEGACY_BGF_CONECT_FORMAT
            has_conect_format = True

        parts = raw.split()
        is_atom = parts and parts[0] in {"ATOM", "HETATM"}
        is_compact = is_atom and len(parts) >= 10
        if is_compact:
            try:
                atom_index = int(parts[1])
                x, y, z = (_finite(value, f"{source_path}:{line_number}") for value in parts[3:6])
                flag1, flag2 = int(parts[7]), int(parts[8])
                charge = _finite(parts[9], f"{source_path}:{line_number}")
            except (ValueError, IndexError):
                is_compact = False
        if is_atom and not has_atom_format:
            output_lines.append(LEGACY_BGF_ATOM_FORMAT)
            has_atom_format = True
        if is_compact:
            output_lines.append(
                _fixed_column_bgf_atom(
                    "HETATM",
                    atom_index,
                    parts[2],
                    x,
                    y,
                    z,
                    parts[6],
                    flag1,
                    flag2,
                    charge,
                )
            )
            converted += 1
            has_atoms = True
        elif is_atom:
            output_lines.append(raw)
            has_atoms = True
        elif stripped == "END":
            if has_atoms and not has_conect_format:
                output_lines.append(LEGACY_BGF_CONECT_FORMAT)
                has_conect_format = True
            output_lines.append(raw)
        else:
            output_lines.append(raw)
    suffix = "\n" if text.endswith(("\n", "\r")) else ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines) + suffix, encoding="utf-8")
    return converted


def parse_geo_inventory(path: str | Path) -> dict[str, set[str]]:
    inventory: dict[str, set[str]] = {}
    current: str | None = None
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        stripped = raw.strip()
        if stripped.startswith("DESCRP"):
            parts = stripped.split(maxsplit=1)
            if len(parts) != 2 or not parts[1].strip():
                raise ValueError(f"{path}:{line_number}: blank DESCRP identifier")
            current = parts[1].strip()
            if current in inventory:
                raise ValueError(f"{path}:{line_number}: duplicate DESCRP identifier {current}")
            inventory[current] = set()
        elif stripped.startswith(("ATOM", "HETATM")) and current is not None:
            parts = stripped.split()
            if len(parts) >= 3:
                inventory[current].add(parts[2])
        elif stripped == "END":
            current = None
    if not inventory:
        raise ValueError(f"{path}: no DESCRP structures found")
    return inventory


def _finite(value: str, context: str) -> float:
    try:
        parsed = float(value.replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise ValueError(f"{context}: invalid numeric value {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{context}: non-finite numeric value {value!r}")
    return parsed


def parse_trainset(path: str | Path) -> list[TrainingItem]:
    source = Path(path)
    section: str | None = None
    items: list[TrainingItem] = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        stripped = raw.split("#", 1)[0].split("!", 1)[0].strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper in SECTION_ENDS:
            section = upper
            continue
        if section is not None and upper == SECTION_ENDS[section]:
            section = None
            continue
        if upper.startswith("END"):
            section = None
            continue
        if section is None:
            continue
        parts = stripped.split()
        context = f"{source}:{line_number}"
        if section == "CHARGE":
            if len(parts) < 4:
                raise ValueError(f"{context}: malformed CHARGE row")
            structure_ids = (parts[0],)
            weight = _finite(parts[1], context)
            reference = _finite(parts[-1], context)
            label = f"Charge {parts[0]} atom {parts[2]}"
        elif section == "GEOMETRY":
            if len(parts) < 4:
                raise ValueError(f"{context}: malformed GEOMETRY row")
            structure_ids = (parts[0],)
            weight = _finite(parts[1], context)
            reference = _finite(parts[-1], context)
            label = f"Geo {stripped}"
        else:
            if len(parts) < 3:
                raise ValueError(f"{context}: malformed ENERGY row")
            weight = _finite(parts[0], context)
            reference = _finite(parts[-1], context)
            structure_ids = tuple(ENERGY_ID_RE.findall(stripped))
            if not structure_ids:
                raise ValueError(f"{context}: ENERGY row has no structure_id/divisor term")
            label = f"Energy {stripped}"
        if weight == 0:
            raise ValueError(f"{context}: ReaxFF training weight cannot be zero")
        items.append(
            TrainingItem(
                item_index=len(items),
                section=section,
                structure_ids=structure_ids,
                item_label=label,
                reference_value=reference,
                weight=weight,
            )
        )
    if not items:
        raise ValueError(f"{source}: no ENERGY, GEOMETRY, or CHARGE training rows found")
    output_order = {"CHARGE": 0, "ENERGY": 1, "GEOMETRY": 2}
    ordered = sorted(items, key=lambda item: output_order[item.section])
    return [
        TrainingItem(
            item_index=index,
            section=item.section,
            structure_ids=item.structure_ids,
            item_label=item.item_label,
            reference_value=item.reference_value,
            weight=item.weight,
        )
        for index, item in enumerate(ordered)
    ]


def validate_training_inputs(
    geo_inventory: dict[str, set[str]],
    items: list[TrainingItem],
) -> None:
    missing = sorted(
        {
            identifier
            for item in items
            for identifier in item.structure_ids
            if identifier not in geo_inventory
        }
    )
    if missing:
        raise ValueError(f"trainset.in identifiers absent from geo: {', '.join(missing)}")
