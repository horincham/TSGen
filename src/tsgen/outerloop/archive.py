"""Canonical successful label records, independent of the current candidate pool."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tsgen.data.xyz import Frame, read_xyz_multiframe
from tsgen.utils.hashing import structure_hash
from tsgen.outerloop.io import atomic_json


def validate_record(row: dict) -> dict:
    r = dict(row)
    xyz = np.asarray(r['coords'], dtype=float)
    if not r['symbols'] or xyz.shape != (len(r['symbols']), 3) or not np.isfinite(xyz).all():
        raise ValueError('Invalid label geometry')
    if r.get('id') != structure_hash(r['symbols'], xyz):
        raise ValueError('Label geometry ID mismatch')
    if not np.isfinite(float(r['e_hartree'])):
        raise ValueError('Label energy must be finite')
    for key in ('charge', 'multiplicity'):
        if int(r[key]) != float(r[key]):
            raise ValueError(f'Invalid {key}')
        r[key] = int(r[key])
    if r['multiplicity'] < 1:
        raise ValueError('Invalid multiplicity')
    if not str(r.get('method', '')).strip() or not str(r.get('species_id', '')).strip():
        raise ValueError('Label method and species_id are required')
    r['coords'] = xyz.tolist()
    r['e_hartree'] = float(r['e_hartree'])
    return r


def load_archive(path: Path, project_root: Path | None = None) -> list[dict]:
    path = Path(path).resolve()
    if path.suffix.lower() == '.csv':
        rows = pd.read_csv(path, dtype={'id': str}).to_dict('records')
    elif path.suffix.lower() == '.jsonl':
        rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    else:
        rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list):
        raise ValueError('Archive must be a list of label records')
    out = []
    for raw in rows:
        if str(raw.get('status', 'ok')).lower() not in {'ok', 'success', 'labeled'}:
            continue
        r = dict(raw)
        if 'coords' not in r:
            xyz_path = r.get('xyz_path')
            is_job_path=not bool(xyz_path)
            if not xyz_path:
                job = r.get('job_dir')
                if not job:
                    raise ValueError('Label requires coords, xyz_path or job_dir')
                xyz_path = str(Path(job) / 'structure.xyz')
            base=Path(project_root) if is_job_path and project_root is not None else path.parent
            f = (base / str(xyz_path)).resolve()
            if not f.is_file():
                raise FileNotFoundError(f'Label geometry not found: {f}; legacy job_dir paths are relative to project_root, explicit xyz_path is relative to the label table')
            frames = read_xyz_multiframe(f)
            if len(frames) != 1:
                raise ValueError('Each label must map to exactly one geometry')
            if is_job_path and (f.parent/'input.gjf').is_file():
                # Legacy structure.xyz writes six decimals while Gaussian input
                # retains ten. Restore the actual SP geometry before checking
                # its four-decimal hash; do not weaken ID validation or rename it.
                from tsgen.outerloop.fitting import _gaussian_input_geometry
                precise,q,m=_gaussian_input_geometry(f.parent/'input.gjf')
                if precise.symbols!=frames[0].symbols or not np.allclose(precise.coords,frames[0].coords,rtol=0,atol=5.1e-7):
                    raise ValueError('Gaussian input and label structure.xyz disagree')
                if (q,m)!=(int(r['charge']),int(r['multiplicity'])):
                    raise ValueError('Gaussian input and label electronic state disagree')
                frames=[precise]
            r['symbols'], r['coords'] = frames[0].symbols, frames[0].coords.tolist()
        r['method'] = r.get('method') or r.get('route', '')
        out.append(validate_record(r))
    return merge_records([], out, set())


def merge_records(old: list[dict], new: list[dict], protected: set[str]) -> list[dict]:
    merged = {}
    for raw in old + new:
        r = validate_record(raw)
        if r['id'] in protected:
            raise ValueError(f'Label is protected test geometry: {r["id"]}')
        previous = merged.get(r['id'])
        if previous is not None:
            keys = ('symbols', 'coords', 'charge', 'multiplicity', 'method', 'species_id')
            if any(previous[k] != r[k] for k in keys) or not np.isclose(previous['e_hartree'], r['e_hartree'], rtol=0, atol=1e-9):
                raise ValueError(f'Label conflict: {r["id"]}')
        else:
            merged[r['id']] = r
    return list(merged.values())


def save_archive(path: Path, records: list[dict]) -> None:
    atomic_json(path, merge_records([], records, set()))


def species_reference_energies(records: list[dict], *, require_optimized: bool) -> dict[str, float]:
    references = {}
    sampled_minima = {}
    for record in records:
        sid = record['species_id']
        sampled_minima[sid] = min(sampled_minima.get(sid, np.inf), record['e_hartree'])
        energy = record.get('reference_hartree')
        if energy is None or not np.isfinite(float(energy)):
            if require_optimized:
                raise ValueError(f'Missing optimized reference_hartree for {sid}')
            continue
        energy = float(energy)
        if sid in references and not np.isclose(energy, references[sid], rtol=0, atol=1e-9):
            raise ValueError(f'Conflicting reference_hartree for {sid}')
        references.setdefault(sid, energy)
    return sampled_minima | references


def frames_from_records(records: list[dict]) -> list[Frame]:
    return [Frame(r['symbols'], np.asarray(r['coords'], dtype=float), comment=f'id={r["id"]}') for r in records]
