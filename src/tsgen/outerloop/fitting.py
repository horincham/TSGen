"""Validate force fields and cumulative fitting inputs at the engine boundary."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import json
import re
import numpy as np
from tsgen.data.xyz import Frame
from tsgen.trainset.io import read_frames_from_reaxff_geo
from tsgen.sensitivity.ffield import FFieldDocument
from tsgen.sensitivity.params_ml import parse_params_ml
from tsgen.sensitivity.trainset_map import parse_geo_inventory, parse_trainset, validate_training_inputs
from tsgen.trainset.gaussian_scan_parser import parse_completed_opt_records
from tsgen.outerloop.io import read_json


def geometry_key(frame: Frame) -> str:
    """Conservative invariant exclusion key, not proof of molecular identity."""
    pairs = []
    for i in range(len(frame.symbols)):
        for j in range(i):
            pair = sorted([frame.symbols[i], frame.symbols[j]])
            pairs.append((*pair, round(float(np.linalg.norm(frame.coords[i]-frame.coords[j])), 4)))
    payload = [sorted(Counter(frame.symbols).items()), sorted(pairs)]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def validate_fit_pair(geo: Path, trainset: Path, protected: list[Frame], previous: tuple[Path, Path] | None = None):
    inventory = parse_geo_inventory(geo)
    items = parse_trainset(trainset)
    if not inventory or not items:
        raise ValueError('Empty ReaxFF fitting inputs')
    validate_training_inputs(inventory, items)
    geometries = read_frames_from_reaxff_geo(geo,preserve_ids=True)
    if not geometries or len({i for i, _ in geometries}) != len(geometries):
        raise ValueError('Invalid or duplicate GEO identifiers')
    if set(inventory) != {i for i, _ in geometries}:
        raise ValueError('GEO parser/inventory mismatch')
    forbidden = {geometry_key(f) for f in protected}
    if any(geometry_key(f) in forbidden for _, f in geometries):
        raise ValueError('Fitting GEO contains protected test geometry')
    if previous:
        old_geo, old_trainset = previous
        old = dict(read_frames_from_reaxff_geo(old_geo,preserve_ids=True))
        new = dict(geometries)
        if not set(old).issubset(new) or any(old[k].symbols != new[k].symbols or not np.allclose(old[k].coords, new[k].coords, atol=1e-5, rtol=0) for k in old):
            raise ValueError('Cumulative GEO must preserve previous geometries and identifiers')
        def signature(item):
            return item.section, item.structure_ids, item.item_label, item.reference_value, item.weight
        # Strict preservation of existing targets; weighting changes require a new run/protocol.
        if not (Counter(map(signature, parse_trainset(old_trainset))) <= Counter(map(signature, items))):
            raise ValueError('Cumulative trainset must preserve previous fitting targets and weights')
        if len(items) <= len(parse_trainset(old_trainset)):
            raise ValueError('Updated trainset must add actual fitting targets')
    return geometries


def validate_ffield(new: Path, old: Path, params: Path | None) -> list[dict]:
    before, after = FFieldDocument.from_path(old), FFieldDocument.from_path(new)
    old_values = {p.coordinate: p.initial_value for p in before.parameters}
    new_values = {p.coordinate: p.initial_value for p in after.parameters}
    if old_values.keys() != new_values.keys() or before.atom_symbols != after.atom_symbols:
        raise ValueError('Force-field schema changed')
    if params is None:
        if not new_values or not after.atom_symbols or not all(np.isfinite(list(old_values.values()) + list(new_values.values()))):
            raise ValueError('Existing force fields must have finite parameters and atom types')
        return []
    bounds = {b.coordinate: b for b in parse_params_ml(params)}
    if not bounds:
        raise ValueError('No selected optimization parameters')
    if not set(bounds).issubset(old_values):
        raise ValueError('Selected parameter absent from force field')
    rows = []
    for coordinate, x in old_values.items():
        y = new_values[coordinate]
        if not np.isfinite(y):
            raise ValueError('Nonfinite force-field parameter')
        if coordinate not in bounds and not np.isclose(x, y, atol=1e-10, rtol=0):
            raise ValueError(f'Unselected parameter changed: {coordinate.parameter_id}')
        if coordinate in bounds:
            b = bounds[coordinate]
            if not b.lower_bound <= y <= b.upper_bound:
                raise ValueError(f'Optimized parameter outside bounds: {coordinate.parameter_id}')
            rows.append(dict(parameter_id=coordinate.parameter_id, old=x, new=y,
                             absolute_change=abs(y-x),relative_change=None if x == 0 else abs((y-x)/x)))
    return rows


def _gaussian_input_geometry(path: Path):
    lines=path.read_text(encoding='utf8').splitlines()
    for i,line in enumerate(lines):
        if re.fullmatch(r'\s*-?\d+\s+\d+\s*',line):
            q,m=map(int,line.split());symbols=[];coords=[]
            for atom in lines[i+1:]:
                if not atom.strip():break
                parts=atom.split()
                if len(parts)!=4:raise ValueError('Derivation requires Cartesian Gaussian input')
                symbols.append(parts[0]);coords.append([float(v) for v in parts[1:]])
            if symbols:return Frame(symbols,np.asarray(coords),''),q,m
    raise ValueError('Missing derivation Gaussian input geometry')


def verify_derivation(row: dict, final_frame: Frame, source_label: dict, outputs: dict):
    """Direct labels must match; altered geometries need a verified Opt/scan chain.

    Lineage is a receipt artifact containing [{input_artifact,log_artifact,
    record_index}]. Every input geometry must originate at the preceding output;
    the first input must match the acquired label, and the last output the GEO.
    This supports relaxed scans and optimizations without accepting an arbitrary
    same-formula geometry as evidence of dataset updating.
    """
    current=Frame(source_label['symbols'],np.asarray(source_label['coords']),'')
    artifact=row.get('lineage_artifact')
    if artifact is not None and str(artifact) not in {'','nan'}:
        if artifact not in outputs:raise ValueError('Missing derivation lineage artifact')
        steps=read_json(Path(outputs[artifact]['path']))
        if not isinstance(steps,list) or not steps:raise ValueError('Empty geometry derivation chain')
        for step in steps:
            if step['input_artifact'] not in outputs or step['log_artifact'] not in outputs:
                raise ValueError('Missing derivation source artifacts')
            input_frame,q,m=_gaussian_input_geometry(Path(outputs[step['input_artifact']]['path']))
            if step.get('transform'):
                from tsgen.trainset.generate import _set_bond,_set_angle,_set_torsion
                transform=step['transform']
                function={'bond':_set_bond,'angle':_set_angle,'torsion':_set_torsion}.get(transform['scan_type'])
                if function is None:raise ValueError('Unknown derivation coordinate transform')
                current=function(current,tuple(transform['atoms_0based']),float(transform['target']))
                if current is None:raise ValueError('Invalid derivation coordinate transform')
            matching=geometry_key(input_frame)==geometry_key(current) or (input_frame.symbols==current.symbols and np.allclose(input_frame.coords,current.coords,atol=1e-5,rtol=0))
            if (q,m)!=(source_label['charge'],source_label['multiplicity']) or not matching:
                raise ValueError('Geometry derivation input does not match its source')
            text=Path(outputs[step['log_artifact']]['path']).read_text(encoding='utf8',errors='replace')
            if text.rfind('Normal termination')<0 or text.rfind('Error termination')>text.rfind('Normal termination'):
                raise ValueError('Geometry derivation Gaussian job failed')
            records=parse_completed_opt_records(text);index=int(step['record_index'])
            if not 0<=index<len(records):raise ValueError('Invalid derivation record index')
            _,current=records[index]
    if geometry_key(current)!=geometry_key(final_frame) and not (current.symbols==final_frame.symbols and np.allclose(current.coords,final_frame.coords,atol=1e-5,rtol=0)):
        raise ValueError('Fitting geometry does not match acquired label or verified derivation')
