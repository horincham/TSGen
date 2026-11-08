from __future__ import annotations

import math
from decimal import Decimal,ROUND_FLOOR,ROUND_CEILING,ROUND_HALF_UP
from pathlib import Path

import pandas as pd

from tsgen.sensitivity.config import SensitivityConfig
from tsgen.sensitivity.config import ParameterSpaceCfg
from tsgen.sensitivity.ffield import FFieldDocument
from tsgen.sensitivity.models import ParameterBound, ParameterCoordinate, TrainingItem
from tsgen.sensitivity.parameter_validity import exclusion_reason


DEFAULT_ITEM_ALLOWLIST = {
    3: [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    4: [1, 2, 3, 4, 5, 6],
    5: [1, 2, 3, 4, 5, 6, 7],
    6: [1, 2, 3, 4, 5, 6, 7],
    7: [1, 2, 3, 4],
}


def parse_params_ml(path: str | Path) -> list[ParameterBound]:
    source = Path(path)
    rows: list[ParameterBound] = []
    seen: set[ParameterCoordinate] = set()
    terminated = False
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.split("#", 1)[0].split('!',1)[0].strip()
        if not stripped:
            continue
        if stripped.lower() == "null":
            terminated = True
            continue
        if terminated:
            raise ValueError(f"{source}:{line_number}: data found after terminal null row")
        parts = stripped.split()
        if len(parts) != 6:
            raise ValueError(f"{source}:{line_number}: expected six columns, observed {len(parts)}")
        try:
            coordinate = ParameterCoordinate(*(int(value) for value in parts[:3]))
            delta, first, second = (float(value.replace("D", "E").replace("d", "e")) for value in parts[3:])
        except ValueError as exc:
            raise ValueError(f"{source}:{line_number}: invalid numeric value") from exc
        if not all(math.isfinite(value) for value in (delta, first, second)):
            raise ValueError(f"{source}:{line_number}: all values must be finite")
        if coordinate in seen:
            raise ValueError(f"{source}:{line_number}: duplicate coordinate {coordinate.parameter_id}")
        seen.add(coordinate)
        lower, upper = sorted((first, second))
        rows.append(ParameterBound(coordinate, delta, lower, upper, f"{source}:{line_number}"))
    if not rows:
        raise ValueError(f"{source}: no candidate parameter rows found")
    return rows


def validate_parameter_bounds(
    bounds: list[ParameterBound],
    document: FFieldDocument,
    require_initial_inside_bounds: bool = True,
) -> list[ParameterBound]:
    available = {parameter.coordinate for parameter in document.parameters}
    for bound in bounds:
        label = bound.coordinate.parameter_id
        if not math.isfinite(bound.delta) or bound.delta<=0:
            raise ValueError(f'Parameter {label} delta/step must be finite and positive')
        if not all(math.isfinite(v) for v in (bound.lower_bound,bound.upper_bound)):
            raise ValueError(f'Parameter {label} bounds must be finite')
        if bound.coordinate not in available:
            raise ValueError(f"Parameter coordinate {label} is absent from ffield")
        if bound.lower_bound >= bound.upper_bound:
            raise ValueError(f"Parameter {label} has a zero-width or reversed interval")
        initial = document.get(bound.coordinate)
        if require_initial_inside_bounds and not bound.lower_bound <= initial <= bound.upper_bound:
            raise ValueError(
                f"Initial value {initial} for {label} is outside [{bound.lower_bound}, {bound.upper_bound}]"
            )
    return bounds


def bound_decimal_places(bound: ParameterBound, initial: float | None = None) -> int:
    """Two places normally; retain small values, narrow ranges and the initial value."""
    width = bound.upper_bound - bound.lower_bound
    if not math.isfinite(width) or width <= 0:
        raise ValueError(f'Invalid optimizer interval: {bound.coordinate.parameter_id}')
    digits = max(2, 1 - math.floor(math.log10(width)))
    for value in (bound.lower_bound, bound.upper_bound):
        if 0 < abs(value) < .1:
            digits = max(digits, 2 - math.floor(math.log10(abs(value))))
    if initial is not None:
        if not bound.lower_bound <= initial <= bound.upper_bound:
            raise ValueError(f'Initial value outside bounds: {bound.coordinate.parameter_id}')
        # Preserve exact endpoints if rounding inward would exclude the initial.
        for value in (bound.lower_bound, bound.upper_bound):
            gap = abs(initial - value)
            if gap == 0:
                digits = max(digits, max(0, -Decimal(str(value)).as_tuple().exponent))
            else:
                digits = max(digits, math.ceil(-math.log10(gap)))
    return digits


def optimizer_bounds(bounds: list[ParameterBound], initial_values=None) -> list[ParameterBound]:
    rounded=[]
    for bound in sorted(bounds,key=lambda b:b.coordinate):
        if not all(math.isfinite(v) for v in (bound.delta,bound.lower_bound,bound.upper_bound)) or bound.delta<=0:
            raise ValueError(f'Invalid optimizer bounds/delta: {bound.coordinate.parameter_id}')
        initial = (initial_values or {}).get(bound.coordinate)
        digits = bound_decimal_places(bound, initial)
        quantum = Decimal(1).scaleb(-digits)
        # Never export a wider domain than the one actually sampled for HSIC.
        low=float(Decimal(str(bound.lower_bound)).quantize(quantum,rounding=ROUND_CEILING))
        high=float(Decimal(str(bound.upper_bound)).quantize(quantum,rounding=ROUND_FLOOR))
        step_digits = max(2, 1 - math.floor(math.log10(bound.delta)))
        step=float(Decimal(str(bound.delta)).quantize(Decimal(1).scaleb(-step_digits),rounding=ROUND_HALF_UP))
        if not all(math.isfinite(v) for v in (low,high,step)) or low>=high or bound.delta<=0:
            raise ValueError(f'Invalid optimizer interval/step: {bound.coordinate.parameter_id}')
        rounded.append(ParameterBound(bound.coordinate,step,low,high,bound.source))
    return rounded


def write_params_ml(path: str | Path, bounds: list[ParameterBound],style: str = 'internal', initial_values=None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if style not in {'internal','indeedopt'}:
        raise ValueError(f'Unknown parameter-file style: {style}')
    output_bounds=optimizer_bounds(bounds, initial_values) if style=='indeedopt' else sorted(bounds,key=lambda b:b.coordinate)
    originals = {b.coordinate: b for b in bounds}
    for bound in output_bounds:
        coordinate = bound.coordinate
        rows.append((
            f'{coordinate.section:4d}{coordinate.block:5d}{coordinate.item:5d}'
            f' {bound.delta:14.{max(2, 1-math.floor(math.log10(bound.delta)))}f}'
            f' {bound.lower_bound:14.{bound_decimal_places(originals[coordinate], (initial_values or {}).get(coordinate))}f}'
            f' {bound.upper_bound:14.{bound_decimal_places(originals[coordinate], (initial_values or {}).get(coordinate))}f}'
        ) if style=='indeedopt' else (
            f"{coordinate.section} {coordinate.block} {coordinate.item} "
            f"{bound.delta:.12g} {bound.lower_bound:.12g} {bound.upper_bound:.12g}"
        ))
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def candidate_audit_rows(bounds: list[ParameterBound], document: FFieldDocument) -> list[dict]:
    by_coordinate = {parameter.coordinate: parameter for parameter in document.parameters}
    rows = []
    for bound in bounds:
        parameter = by_coordinate[bound.coordinate]
        rows.append(
            {
                "parameter_id": bound.coordinate.parameter_id,
                "parameter_label": parameter.parameter_name,
                "section_name": parameter.section_name,
                "block_key": "-".join(parameter.block_key),
                "initial_value": parameter.initial_value,
                "delta": bound.delta,
                "lower_bound": bound.lower_bound,
                "upper_bound": bound.upper_bound,
                "source": bound.source,
                "status": "valid",
            }
        )
    return rows


def build_rule_candidates(
    document: FFieldDocument,
    cfg: ParameterSpaceCfg,
    geo_inventory: dict[str, set[str]],
    items: list[TrainingItem],
) -> tuple[list[ParameterBound], list[dict]]:
    all_symbols = set().union(*geo_inventory.values()) if geo_inventory else set()
    has_charge = any(item.section == "CHARGE" for item in items)
    allowlist = cfg.section_item_allowlist or DEFAULT_ITEM_ALLOWLIST
    force_exclude = {ParameterCoordinate.parse(value) for value in cfg.force_exclude}
    force_include = {ParameterCoordinate.parse(value) for value in cfg.force_include}
    rules = {(rule.section, rule.item): rule for rule in cfg.bound_rules}
    parameters = {parameter.coordinate: parameter for parameter in document.parameters}
    active = []
    audit = []
    for parameter in document.parameters:
        coordinate = parameter.coordinate
        section = coordinate.section
        included = section in cfg.allowed_sections and coordinate.item in allowlist.get(section, [])
        if section == 2:
            included = (
                cfg.include_charge_parameters
                and has_charge
                and coordinate.item in cfg.charge_atom_items
            )
        if section >= 3 and included:
            symbols = {
                document.atom_symbols.get(int(key), "")
                for key in parameter.block_key
                if key != "*" and key.isdigit()
            }
            included = bool(symbols) and symbols <= all_symbols
        if coordinate in force_include:
            included = True
        if included and coordinate not in force_exclude:
            reason=exclusion_reason(parameter,document,geo_inventory,items,cfg.exclude_inactive_parameters)
            if reason:
                audit.append(dict(parameter_id=coordinate.parameter_id,status=reason,initial_value=parameter.initial_value))
                continue
        if included and coordinate not in force_exclude:
            active.append(parameter)
    bounds = []
    for parameter in active:
        coordinate = parameter.coordinate
        override = cfg.bounds_overrides.get(coordinate.parameter_id)
        rule = rules.get((coordinate.section, coordinate.item))
        if override is not None:
            delta, lower, upper = override
            source = "bounds_override"
        elif rule is not None:
            delta, lower, upper = rule.delta, rule.lower_bound, rule.upper_bound
            source = "bound_rule"
        else:
            audit.append(
                {
                    "parameter_id": coordinate.parameter_id,
                    "parameter_label": parameter.parameter_name,
                    "status": "unresolved_bound",
                    "initial_value": parameter.initial_value,
                }
            )
            continue
        lower, upper = sorted((float(lower), float(upper)))
        bound = ParameterBound(coordinate, float(delta), lower, upper, source)
        bounds.append(bound)
        audit.extend(candidate_audit_rows([bound], document))
    return bounds, audit


def build_auto_candidates(
    document: FFieldDocument,
    cfg: ParameterSpaceCfg,
    geo_inventory: dict[str, set[str]],
    items: list[TrainingItem],
) -> tuple[list[ParameterBound], list[dict]]:
    """Build a conservative ReaxFF candidate interface directly from ffield."""
    all_symbols = set().union(*geo_inventory.values()) if geo_inventory else set()
    has_charge = any(item.section == "CHARGE" for item in items)
    allowlist = cfg.section_item_allowlist or DEFAULT_ITEM_ALLOWLIST
    force_exclude = {ParameterCoordinate.parse(value) for value in cfg.force_exclude}
    force_include = {ParameterCoordinate.parse(value) for value in cfg.force_include}
    rules = {(rule.section, rule.item): rule for rule in cfg.bound_rules}
    bounds: list[ParameterBound] = []
    audit: list[dict] = []

    for parameter in document.parameters:
        coordinate = parameter.coordinate
        section = coordinate.section
        included = section in cfg.allowed_sections and coordinate.item in allowlist.get(section, [])
        if section == 2:
            included = (
                cfg.include_charge_parameters
                and has_charge
                and coordinate.item in cfg.charge_atom_items
            )
        if section >= 3 and included:
            symbols = {
                document.atom_symbols.get(int(key), "")
                for key in parameter.block_key
                if key != "*" and key.isdigit()
            }
            included = bool(symbols) and symbols <= all_symbols
        if coordinate in force_include:
            included = True
        if not included or coordinate in force_exclude:
            continue
        reason=exclusion_reason(parameter,document,geo_inventory,items,cfg.exclude_inactive_parameters)
        if reason:
            audit.append(dict(parameter_id=coordinate.parameter_id,status=reason,initial_value=parameter.initial_value))
            continue

        initial = float(parameter.initial_value)
        override = cfg.bounds_overrides.get(coordinate.parameter_id)
        rule = rules.get((section, coordinate.item))
        if override is not None:
            delta, lower, upper = (float(value) for value in override)
            source = "bounds_override"
        elif rule is not None:
            delta = float(rule.delta)
            lower = float(rule.lower_bound)
            upper = float(rule.upper_bound)
            source = "bound_rule"
        elif cfg.require_explicit_bounds and (abs(initial) > cfg.zero_tolerance or
                                              cfg.include_zero_parameters or coordinate in force_include):
            raise ValueError(f'Parameter {coordinate.parameter_id} needs explicit bounds: '
                             'provide bounds_overrides, bound_rules or mode=from_params_ml')
        elif abs(initial) <= cfg.zero_tolerance:
            if not cfg.include_zero_parameters and coordinate not in force_include:
                audit.append(
                    {
                        "parameter_id": coordinate.parameter_id,
                        "parameter_label": parameter.parameter_name,
                        "section_name": parameter.section_name,
                        "block_key": "-".join(parameter.block_key),
                        "initial_value": initial,
                        "status": "excluded_zero",
                        "source": "auto_zero_policy",
                    }
                )
                continue
            delta = max(cfg.zero_absolute_range * cfg.delta_fraction, cfg.minimum_delta)
            lower = -cfg.zero_absolute_range
            upper = cfg.zero_absolute_range
            source = "auto_zero_absolute_range"
        else:
            span = abs(initial) * cfg.relative_range
            lower = initial - span
            upper = initial + span
            delta = max(abs(initial) * cfg.delta_fraction, cfg.minimum_delta)
            source = f"auto_relative_range:{cfg.relative_range:.6g}"

        lower, upper = sorted((float(lower), float(upper)))
        bound = ParameterBound(coordinate, float(delta), lower, upper, source)
        bounds.append(bound)
        audit.extend(candidate_audit_rows([bound], document))

    if not bounds:
        raise ValueError(
            "auto_from_ffield found no candidate parameters; inspect geo atom types, "
            "trainset sections, allowed_sections, and force_include"
        )
    return bounds, audit


def generate_candidate_file(
    cfg: SensitivityConfig,
    output: str | Path,
) -> list[ParameterBound]:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = FFieldDocument.from_path(cfg.inputs.ffield)
    if cfg.parameter_space.mode == "from_params_ml":
        bounds = validate_parameter_bounds(
            parse_params_ml(cfg.parameter_space.params_ml),
            document,
            cfg.parameter_space.require_initial_inside_bounds,
        )
        audit = candidate_audit_rows(bounds, document)
    else:
        if cfg.inputs.geo is None or cfg.inputs.trainset_in is None:
            raise ValueError(
                f"{cfg.parameter_space.mode} candidate generation requires "
                "inputs.geo and inputs.trainset_in"
            )
        from tsgen.sensitivity.trainset_map import (
            parse_geo_inventory,
            parse_trainset,
            validate_training_inputs,
        )

        inventory = parse_geo_inventory(cfg.inputs.geo)
        items = parse_trainset(cfg.inputs.trainset_in)
        validate_training_inputs(inventory, items)
        if cfg.parameter_space.mode == "auto_from_ffield":
            bounds, audit = build_auto_candidates(
                document,
                cfg.parameter_space,
                inventory,
                items,
            )
        else:
            bounds, audit = build_rule_candidates(
                document,
                cfg.parameter_space,
                inventory,
                items,
            )
        unresolved = [row for row in audit if row.get("status") == "unresolved_bound"]
        pd.DataFrame(audit).to_csv(destination.with_suffix(".audit.csv"), index=False)
        if unresolved:
            raise ValueError(
                f"{len(unresolved)} candidate parameters have unresolved bounds; "
                f"inspect {destination.with_suffix('.audit.csv')}"
            )
        validate_parameter_bounds(
            bounds,
            document,
            cfg.parameter_space.require_initial_inside_bounds,
        )
    write_params_ml(destination, bounds)
    pd.DataFrame(document.catalog_rows()).to_csv(
        destination.with_suffix(".catalog.csv"),
        index=False,
    )
    pd.DataFrame(audit).to_csv(destination.with_suffix(".audit.csv"), index=False)
    return bounds
