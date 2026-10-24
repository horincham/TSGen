from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from tsgen.data.xyz import Frame


_REAXFF_BAD_ID_CHARS = re.compile(r"[^A-Za-z0-9_.]")

# The supplied legacy ReaxFF source stores structure keys in CHARACTER*20 and
# copies at most characters 1:19 from BGF DESCRP records.  Longer keys are
# silently truncated and then fail to match ENERGY references.
LEGACY_REAXFF_ID_MAX_LEN = 19

LEGACY_BGF_ATOM_FORMAT = (
    "FORMAT ATOM   (a6,1x,i5,1x,a5,1x,a3,1x,a1,1x,a5,"
    "3f10.5,1x,a5,i3,i2,1x,f8.5)"
)
LEGACY_BGF_CONECT_FORMAT = "FORMAT CONECT (a6,12i6)"


def sanitize_reaxff_id(s: str, max_len: int = LEGACY_REAXFF_ID_MAX_LEN) -> str:
    """Sanitize a DESCRP/trainset identifier for standalone ReaxFF.

    The ReaxFF manual treats ``+``, ``-`` and ``/`` as operators in the
    ENERGY section, so identifiers must not contain them.  Dots are retained
    because legacy ReaxFF training sets commonly use names such as
    ``HClO_Cl_O_4.8`` to encode a scan value.
    """
    out = _REAXFF_BAD_ID_CHARS.sub("_", str(s).strip())
    out = re.sub(r"_+", "_", out).strip("_")
    if not out:
        out = "structure"
    if out[0].isdigit():
        out = "m_" + out
    max_len = int(max_len)
    if max_len < 8:
        raise ValueError("ReaxFF identifier max_len must be at least 8")
    if len(out) <= max_len:
        return out
    digest = hashlib.sha1(out.encode("utf-8")).hexdigest()[:6]
    return f"{out[: max_len - len(digest) - 1]}_{digest}"


def format_scan_value(value: float, scan_type: str | None = None) -> str:
    """Human-readable scan value used in DESCRP/trainset identifiers."""
    x = float(value)
    if scan_type == "bond":
        return f"{x:.2f}"
    if scan_type in {"angle", "torsion"}:
        # Keep one decimal so Gaussian integer grids still read as scan values.
        return f"{x:.1f}"
    return f"{x:.2f}"


def _fmt_weight(weight: float) -> str:
    w = float(weight)
    if abs(w - round(w)) < 1e-9:
        return str(int(round(w)))
    return f"{w:.6g}"


def _safe_lit_value(value: float, epsilon: float = 0.0001) -> float:
    """Return a ReaxFF-safe QC/Lit target value.

    Standalone ReaxFF has a documented bug/limitation with exact zero
    literature/QC values in trainset.in.  Preserve the sign when possible and
    replace exact or near-exact zero with a small non-zero value.
    """
    v = float(value)
    eps = abs(float(epsilon)) if float(epsilon) != 0 else 0.0001
    if abs(v) < eps:
        return eps if v >= 0 else -eps
    return v


def atom_label(symbol: str, index0: int) -> str:
    return f"{symbol}{index0 + 1}"


def coordinate_label(symbols: Sequence[str], atom_indices0: Sequence[int], with_indices: bool = True) -> str:
    if with_indices:
        return "_".join(atom_label(symbols[i], i) for i in atom_indices0)
    return "_".join(symbols[i] for i in atom_indices0)


def make_scan_step_id(scan_prefix: str, value: float, scan_type: str) -> str:
    return sanitize_reaxff_id(f"{scan_prefix}_{format_scan_value(value, scan_type)}")


def write_bgf_structure(
    path: Path,
    ident: str,
    frame: Frame,
    rutype: str = "SINGLE POINT",
    remark: str | None = "TSGen ReaxFF training geometry",
    include_format_lines: bool = True,
    periodic: bool = False,
    cell: Sequence[float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        bgf_block(
            ident,
            frame,
            rutype=rutype,
            remark=remark,
            include_format_lines=include_format_lines,
            periodic=periodic,
            cell=cell,
        ) + "\n",
        encoding="utf-8",
    )


def bgf_block(
    ident: str,
    frame: Frame,
    rutype: str = "SINGLE POINT",
    remark: str | None = "TSGen ReaxFF training geometry",
    include_format_lines: bool = True,
    periodic: bool = False,
    cell: Sequence[float] | None = None,
) -> str:
    """Return a ReaxFF-readable BGF block.

    Defaults target isolated molecular systems: ``BIOGRF`` without ``CRYSTX``.
    Periodic ``XTLGRF``/``CRYSTX`` output is only written when explicitly enabled.
    """
    ident = sanitize_reaxff_id(ident)
    lines: list[str] = []
    lines.append("XTLGRF 200" if periodic else "BIOGRF 200")
    lines.append(f"DESCRP {ident}")
    if remark:
        lines.append(f"REMARK {str(remark).strip()[:160]}")
    if rutype:
        lines.append(f"RUTYPE {rutype}")
    if periodic:
        c = list(cell or [80.0, 80.0, 80.0, 90.0, 90.0, 90.0])
        if len(c) != 6:
            raise ValueError("periodic cell must have six values: a b c alpha beta gamma")
        lines.append("CRYSTX " + " ".join(f"{float(v):10.5f}" for v in c))
    if include_format_lines:
        lines.append(LEGACY_BGF_ATOM_FORMAT)
    for idx, (sym, (x, y, z)) in enumerate(zip(frame.symbols, frame.coords), start=1):
        lines.append(
            f"HETATM {idx:5d} {sym:<5s} RES A {1:<5d}"
            f"{float(x):10.5f}{float(y):10.5f}{float(z):10.5f} "
            f"{sym:<5s}{0:3d}{0:2d} {0.0:8.5f}"
        )
    if include_format_lines:
        lines.append(LEGACY_BGF_CONECT_FORMAT)
    lines.append("END")
    return "\n".join(lines)


def write_geo_file(
    path: Path,
    rows: list[dict],
    *,
    include_format_lines: bool = True,
    periodic: bool = False,
    cell: Sequence[float] | None = None,
    remark: str | None = "TSGen ReaxFF training geometry",
    equilibrium_rutype: str = "NORMAL RUN",
    scan_rutype: str = "SINGLE POINT",
) -> None:
    blocks: list[str] = []
    seen: set[str] = set()
    for r in rows:
        fr: Frame = r["frame"]
        ident = sanitize_reaxff_id(str(r.get("reaxff_id") or r.get("id")))
        if ident in seen:
            raise ValueError(
                f"Duplicate legacy ReaxFF DESCRP identifier {ident!r}; "
                "source identifiers must be unique before GEO export"
            )
        seen.add(ident)
        stype = str(r.get("scan_type", ""))
        rutype = equilibrium_rutype if stype == "equilibrium" else scan_rutype
        blocks.append(
            bgf_block(
                ident,
                fr,
                rutype=rutype,
                remark=remark,
                include_format_lines=include_format_lines,
                periodic=periodic,
                cell=cell,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def write_models_in(path: Path, bgf_paths: Sequence[Path]) -> None:
    lines = [str(p) for p in bgf_paths]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_trainset_in(
    path: Path,
    geometry_rows: list[dict],
    energy_rows: list[dict],
    charge_rows: list[dict] | None = None,
    skip_zero_targets: bool = False,
    zero_epsilon: float = 0.0001,
    geometry_zero_epsilon: float | None = None,
) -> None:
    """Write canonical standalone ReaxFF trainset.in.

    Only the sections requested by TSGen are written.  Comments are kept concise
    and grouped by structure/scan coordinate so trainset.in mirrors the DESCRP
    identifiers in the generated geo file.
    """
    lines: list[str] = []
    lines.append("# TSGen-generated ReaxFF trainset.in")
    lines.append("# Identifiers below correspond exactly to DESCRP fields in geo.")
    lines.append("# ReaxFF error term is ((ReaxFF - QC/Lit) / Weight)^2; larger Weight downweights a data point.")
    lines.append("# Exact zero QC/Lit targets are replaced by a small non-zero epsilon to avoid the documented ReaxFF zero-value bug.")
    lines.append("")

    charge_rows = charge_rows or []
    if charge_rows:
        lines.append("CHARGE")
        lines.append("#Iden Weight Atom Lit")
        last_ident = None
        for r in charge_rows:
            ident = sanitize_reaxff_id(str(r["id"]))
            if ident != last_ident:
                lines.append(f"#{ident}")
                last_ident = ident
            weight = float(r["weight"])
            atom = int(r["atom_index_1based"])
            charge = float(r["charge"])
            lines.append(f"{ident:<24s} {_fmt_weight(weight):>8s} {atom:4d} {charge:12.6f}")
        lines.append("ENDCHARGE")
        lines.append("")

    if geometry_rows:
        lines.append("GEOMETRY")
        lines.append("#Iden Weight At1 At2 At3 At4 Lit")
        last_parent = None
        for r in geometry_rows:
            parent = str(r.get("id", ""))
            if parent != last_parent:
                lines.append(f"#{parent}")
                last_parent = parent
            stype = str(r["target_type"])
            ident = sanitize_reaxff_id(str(r["id"]))
            weight = float(r["weight"])
            value = _safe_lit_value(float(r["value"]), geometry_zero_epsilon if geometry_zero_epsilon is not None else zero_epsilon)
            atoms = [int(x) for x in r["atom_indices_1based"].split()] if isinstance(r["atom_indices_1based"], str) else list(r["atom_indices_1based"])
            if stype == "bond":
                lines.append(f"{ident:<24s} {_fmt_weight(weight):>8s} {atoms[0]:4d} {atoms[1]:4d} {value:12.5f}")
            elif stype == "angle":
                lines.append(f"{ident:<24s} {_fmt_weight(weight):>8s} {atoms[0]:4d} {atoms[1]:4d} {atoms[2]:4d} {value:12.5f}")
            elif stype == "torsion":
                lines.append(f"{ident:<24s} {_fmt_weight(weight):>8s} {atoms[0]:4d} {atoms[1]:4d} {atoms[2]:4d} {atoms[3]:4d} {value:12.5f}")
        lines.append("ENDGEOMETRY")
        lines.append("")

    if energy_rows:
        lines.append("ENERGY")
        lines.append("#Weight op1 Ide1 n1 op2 Ide2 n2 Lit")
        last_group = None
        for r in energy_rows:
            value = float(r["target_kcalmol"])
            if abs(value) < abs(float(zero_epsilon)):
                if skip_zero_targets:
                    continue
                value = _safe_lit_value(value, zero_epsilon)
            group = str(r.get("scan_group_comment") or r.get("coord_id") or r.get("family") or "scan")
            if group != last_group:
                lines.append(f"#{group}")
                last_group = group
            ident = sanitize_reaxff_id(str(r["id"]))
            ref = sanitize_reaxff_id(str(r["reference_id"]))
            weight = float(r["weight"])
            lines.append(f"{_fmt_weight(weight):>8s} + {ident}/1 - {ref}/1 {value:12.4f}")
        lines.append("ENDENERGY")
        lines.append("")

    if not charge_rows and not geometry_rows and not energy_rows:
        lines.append("# No trainset entries available yet. Run/resume after Gaussian jobs finish.")

    output = "\n".join(lines)
    issues = validate_legacy_trainset_text(output)
    if issues:
        raise ValueError("Invalid legacy ReaxFF trainset:\n" + "\n".join(issues))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding="utf-8")


def validate_legacy_trainset_text(text: str) -> list[str]:
    """Return compatibility errors for the supplied legacy ReaxFF parser.

    This intentionally validates the concrete behaviour of the bundled 2008
    executable, which is stricter than the generic trainset format described in
    newer manuals.
    """
    issues: list[str] = []
    section: str | None = None
    start_sections = {"CHARGE", "GEOMETRY", "ENERGY"}
    end_sections = {f"END{name}" for name in start_sections}

    def check_id(value: str, line_number: int) -> None:
        if len(value) > LEGACY_REAXFF_ID_MAX_LEN:
            issues.append(
                f"line {line_number}: identifier {value!r} exceeds "
                f"{LEGACY_REAXFF_ID_MAX_LEN} characters"
            )

    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper in start_sections:
            section = upper
            continue
        if upper in end_sections:
            section = None
            continue
        if section is None:
            continue
        if "!" in line:
            issues.append(
                f"line {line_number}: inline comments are unsafe for the legacy parser"
            )
        tokens = line.split()
        if section == "CHARGE":
            if len(tokens) != 4:
                issues.append(f"line {line_number}: CHARGE record must have 4 fields")
            if tokens:
                check_id(tokens[0], line_number)
        elif section == "GEOMETRY":
            if len(tokens) not in {5, 6, 7}:
                issues.append(
                    f"line {line_number}: GEOMETRY record must have 5, 6, or 7 fields"
                )
            if tokens:
                check_id(tokens[0], line_number)
        elif section == "ENERGY":
            references = re.findall(r"[+-]\s+([^/\s]+)/", line)
            if not references:
                issues.append(f"line {line_number}: ENERGY record has no structure reference")
            for ident in references:
                check_id(ident, line_number)
    return issues


def validate_legacy_reaxff_inputs(geo_text: str, trainset_text: str) -> list[str]:
    """Validate GEO identifiers and all trainset cross-references together."""
    issues = validate_legacy_trainset_text(trainset_text)
    geo_ids: list[str] = []
    for line_number, raw in enumerate(geo_text.splitlines(), start=1):
        match = re.match(r"\s*DESCRP\s+(\S+)", raw, flags=re.IGNORECASE)
        if not match:
            continue
        ident = match.group(1)
        geo_ids.append(ident)
        if len(ident) > LEGACY_REAXFF_ID_MAX_LEN:
            issues.append(
                f"geo line {line_number}: identifier {ident!r} exceeds "
                f"{LEGACY_REAXFF_ID_MAX_LEN} characters"
            )
    duplicate_ids = sorted({ident for ident in geo_ids if geo_ids.count(ident) > 1})
    if duplicate_ids:
        issues.append(f"duplicate GEO DESCRP identifiers: {', '.join(duplicate_ids)}")
    known = set(geo_ids)
    section: str | None = None
    starts = {"CHARGE", "GEOMETRY", "ENERGY"}
    references: set[str] = set()
    for raw in trainset_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper in starts:
            section = upper
            continue
        if upper.startswith("END"):
            section = None
            continue
        if section in {"CHARGE", "GEOMETRY"}:
            references.add(line.split()[0])
        elif section == "ENERGY":
            references.update(re.findall(r"[+-]\s+([^/\s]+)/", line))
    missing = sorted(references - known)
    if missing:
        issues.append(f"trainset identifiers missing from GEO: {', '.join(missing)}")
    return issues


def repair_legacy_reaxff_inputs(
    geo_text: str,
    trainset_text: str,
) -> tuple[str, str, dict[str, str]]:
    """Repair an existing TSGen GEO/trainset pair for the bundled legacy ReaxFF."""
    original_ids: list[str] = []
    for raw in geo_text.splitlines():
        match = re.match(r"\s*DESCRP\s+(\S+)", raw, flags=re.IGNORECASE)
        if match:
            original_ids.append(match.group(1))
    mapping = {ident: sanitize_reaxff_id(ident) for ident in original_ids}
    reverse: dict[str, str] = {}
    for original, repaired in mapping.items():
        if repaired in reverse and reverse[repaired] != original:
            raise ValueError(
                f"Legacy identifier collision: {reverse[repaired]!r} and {original!r} "
                f"both map to {repaired!r}"
            )
        reverse[repaired] = original

    repaired_geo_lines: list[str] = []
    for raw in geo_text.splitlines():
        match = re.match(r"(\s*DESCRP\s+)(\S+)(.*)", raw, flags=re.IGNORECASE)
        if match:
            raw = f"{match.group(1)}{mapping[match.group(2)]}{match.group(3)}"
        repaired_geo_lines.append(raw)
    repaired_geo = "\n".join(repaired_geo_lines) + ("\n" if geo_text.endswith("\n") else "")

    repaired_trainset_lines: list[str] = []
    section: str | None = None
    starts = {"CHARGE", "GEOMETRY", "ENERGY"}
    for raw in trainset_text.splitlines():
        stripped = raw.strip()
        upper = stripped.upper()
        if upper in starts:
            section = upper
            repaired_trainset_lines.append(raw)
            continue
        if upper.startswith("END"):
            section = None
            repaired_trainset_lines.append(raw)
            continue
        if not stripped or stripped.startswith("#") or section is None:
            repaired_trainset_lines.append(raw)
            continue
        line = raw
        if section == "GEOMETRY":
            line = line.split("!", 1)[0].rstrip()
        if section in {"CHARGE", "GEOMETRY"}:
            match = re.match(r"(\s*)(\S+)(.*)", line)
            if match:
                ident = match.group(2)
                line = f"{match.group(1)}{mapping.get(ident, sanitize_reaxff_id(ident))}{match.group(3)}"
        elif section == "ENERGY":
            def replace_reference(match: re.Match[str]) -> str:
                ident = match.group(2)
                return f"{match.group(1)}{mapping.get(ident, sanitize_reaxff_id(ident))}{match.group(3)}"

            line = re.sub(r"([+-]\s+)([^/\s]+)(/)", replace_reference, line)
        repaired_trainset_lines.append(line)
    repaired_trainset = "\n".join(repaired_trainset_lines) + (
        "\n" if trainset_text.endswith("\n") else ""
    )
    issues = validate_legacy_reaxff_inputs(repaired_geo, repaired_trainset)
    if issues:
        raise ValueError("Repaired ReaxFF inputs are invalid:\n" + "\n".join(issues))
    return repaired_geo, repaired_trainset, mapping
