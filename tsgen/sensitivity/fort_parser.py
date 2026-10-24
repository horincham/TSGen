from __future__ import annotations

import math
from pathlib import Path
import re

from tsgen.sensitivity.models import EvaluationResult, ItemLoss, TrainingItem


FLOAT_TOKEN_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][+-]?\d+)?")
ENERGY_ID_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*/")
MOLECULAR_ITEM_MARKERS = (
    "charge atom:",
    "torsion angle:",
    "valence angle:",
    "bond distance:",
    "position atom:",
    "crms:",
    "rmsg force:",
    "heat of formation:",
    "frequency:",
    "error in mode:",
    "pressure ",
)
FORT99_COLUMN_WIDTH = 12
FORT99_VALUE_COLUMNS = 5


def _finite(token: str) -> float:
    value = float(token.replace("D", "E").replace("d", "e"))
    if not math.isfinite(value):
        raise ValueError(f"Non-finite ReaxFF value {token!r}")
    return value


def parse_fort13(path: str | Path) -> float:
    source = Path(path)
    if not source.exists():
        raise ValueError(f"Missing ReaxFF total objective file: {source}")
    tokens = FLOAT_TOKEN_RE.findall(source.read_text(encoding="utf-8", errors="ignore"))
    if not tokens:
        raise ValueError(f"{source}: no finite total objective found")
    return _finite(tokens[-1])


def _fixed_width_fort99_row(
    raw: str,
    line_number: int,
) -> tuple[str, list[float | None]] | None:
    values_width = FORT99_COLUMN_WIDTH * FORT99_VALUE_COLUMNS
    if len(raw) < values_width:
        return None
    fields = [
        raw[index : index + FORT99_COLUMN_WIDTH]
        for index in range(len(raw) - values_width, len(raw), FORT99_COLUMN_WIDTH)
    ]
    values: list[float | None] = []
    for column_index, field in enumerate(fields):
        if "*" in field:
            if column_index == FORT99_VALUE_COLUMNS - 1:
                values.append(None)
                continue
            column = (
                "item error"
                if column_index == FORT99_VALUE_COLUMNS - 2
                else f"item value column {column_index + 1}"
            )
            raise ValueError(
                f"fort.99 line {line_number}: ReaxFF {column} overflowed "
                f"as {field.strip()!r}"
            )
        try:
            values.append(_finite(field.strip()))
        except ValueError:
            return None
    return raw[:-values_width].strip(), values


def parse_fort99(path: str | Path, items: list[TrainingItem]) -> tuple[ItemLoss, ...]:
    source = Path(path)
    if not source.exists():
        raise ValueError(f"Missing ReaxFF item objective file: {source}")
    numeric_rows: list[tuple[str, list[float | None]]] = []
    for line_number, raw in enumerate(
        source.read_text(encoding="utf-8", errors="ignore").splitlines(),
        start=1,
    ):
        fixed_width = _fixed_width_fort99_row(raw, line_number)
        if fixed_width is not None:
            numeric_rows.append(fixed_width)
            continue
        tokens = FLOAT_TOKEN_RE.findall(raw)
        if len(tokens) < 5:
            continue
        try:
            values = [_finite(token) for token in tokens[-5:]]
        except ValueError:
            continue
        numeric_rows.append((raw[:60].strip(), values))
    losses = []
    current_structure = ""
    for row_index, (output_label, values) in enumerate(numeric_rows):
        predicted, reference, weight, squared_error, _cumulative = values
        if None in (predicted, reference, weight, squared_error):
            raise ValueError(f"fort.99 row {row_index + 1}: incomplete item values")
        lowered = output_label.lower()
        if lowered.startswith("energy "):
            section = "ENERGY"
            structure_ids = ENERGY_ID_RE.findall(output_label)
            structure_id = "+".join(structure_ids)
            item_label = output_label
        else:
            marker_positions = [
                (lowered.find(marker), marker)
                for marker in MOLECULAR_ITEM_MARKERS
                if lowered.find(marker) >= 0
            ]
            if marker_positions:
                marker_position, marker = min(marker_positions)
                prefix = output_label[:marker_position].strip()
                if prefix:
                    current_structure = prefix
                item_label = output_label[marker_position:].strip()
                section = "CHARGE" if marker == "charge atom:" else "GEOMETRY"
            else:
                item_label = output_label
                section = "GEOMETRY"
            structure_id = current_structure
        losses.append(
            ItemLoss(
                item_index=row_index,
                section=section,
                structure_id=structure_id,
                item_label=item_label,
                reaxff_value=float(predicted),
                reference_value=float(reference),
                weight=float(weight),
                squared_error=float(squared_error),
            )
        )
    return tuple(losses)


def build_evaluation_result(
    sample_id: str,
    fort13_path: str | Path,
    fort99_path: str | Path,
    items: list[TrainingItem],
    elapsed_seconds: float,
    return_code: int,
    ffield_sha256: str,
) -> EvaluationResult:
    item_losses = parse_fort99(fort99_path, items)
    item_total = math.fsum(item.squared_error for item in item_losses)
    try:
        total = parse_fort13(fort13_path)
    except ValueError:
        fort13_text = Path(fort13_path).read_text(encoding="utf-8", errors="ignore")
        if "*" not in fort13_text:
            raise
        total = item_total
    output_rounding_tolerance = max(5.0e-3, 5.0e-5 * (len(item_losses) + 1))
    if not math.isclose(
        total,
        item_total,
        rel_tol=1e-5,
        abs_tol=output_rounding_tolerance,
    ):
        raise ValueError(f"fort.13 total {total} does not match fort.99 item sum {item_total}")
    section_losses: dict[str, float] = {}
    for section in ("ENERGY", "GEOMETRY", "CHARGE"):
        values = [item.squared_error for item in item_losses if item.section == section]
        if values:
            section_losses[section] = math.fsum(values)
    return EvaluationResult(
        sample_id=sample_id,
        status="success",
        total_loss=total,
        section_losses=section_losses,
        item_losses=item_losses,
        elapsed_seconds=elapsed_seconds,
        return_code=return_code,
        ffield_sha256=ffield_sha256,
    )


def loss_summary_row(result: EvaluationResult) -> dict:
    counts = {
        section: sum(item.section == section for item in result.item_losses)
        for section in ("ENERGY", "GEOMETRY", "CHARGE")
    }
    return {
        "sample_id": result.sample_id,
        "total_loss": result.total_loss,
        "energy_loss": result.section_losses.get("ENERGY"),
        "geometry_loss": result.section_losses.get("GEOMETRY"),
        "charge_loss": result.section_losses.get("CHARGE"),
        "n_items_total": len(result.item_losses),
        "n_items_energy": counts["ENERGY"],
        "n_items_geometry": counts["GEOMETRY"],
        "n_items_charge": counts["CHARGE"],
        "run_status": result.status,
        "failure_reason": result.failure_reason,
        "elapsed_seconds": result.elapsed_seconds,
        "ffield_sha256": result.ffield_sha256,
    }


def item_loss_rows(result: EvaluationResult) -> list[dict]:
    return [
        {
            "sample_id": result.sample_id,
            "item_index": item.item_index,
            "section": item.section,
            "structure_id": item.structure_id,
            "item_label": item.item_label,
            "reaxff_value": item.reaxff_value,
            "reference_value": item.reference_value,
            "weight": item.weight,
            "squared_error": item.squared_error,
        }
        for item in result.item_losses
    ]
