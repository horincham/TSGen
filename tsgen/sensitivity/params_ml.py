from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from tsgen.sensitivity.config import SensitivityConfig
from tsgen.sensitivity.config import ParameterSpaceCfg
from tsgen.sensitivity.ffield import FFieldDocument
from tsgen.sensitivity.models import ParameterBound, ParameterCoordinate, TrainingItem


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
        stripped = raw.split("#", 1)[0].strip()
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


def write_params_ml(path: str | Path, bounds: list[ParameterBound]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for bound in sorted(bounds, key=lambda item: item.coordinate):
        coordinate = bound.coordinate
        rows.append(
            f"{coordinate.section} {coordinate.block} {coordinate.item} "
            f"{bound.delta:.12g} {bound.lower_bound:.12g} {bound.upper_bound:.12g}"
        )
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
            active.append(parameter)
    audit = []
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
