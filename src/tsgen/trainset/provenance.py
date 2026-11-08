"""Verify exported geometry/energy pairs against converged Gaussian records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from tsgen.data.xyz import Frame
from tsgen.trainset.gaussian_scan_parser import parse_completed_opt_records


def verify_optimized_record(row: dict, frame: Frame, cache: dict) -> dict:
    job = Path(row['job_dir'])
    key = str(job.resolve())
    if key not in cache:
        log = job / 'input.log'
        if not log.is_file():
            raise ValueError(f"Missing source log for converged record {row.get('id')}")
        text = log.read_text(encoding='utf-8', errors='replace')
        if 'Normal termination' not in text or text.rfind('Error termination') > text.rfind('Normal termination'):
            raise ValueError(f"Source log is not a successfully terminated converged job: {job}")
        code = job / 'gaussian.exitcode'
        if code.exists() and code.read_text().strip() != '0':
            raise ValueError(f"Source process did not finish successfully: {job}")
        meta_path = job / 'meta.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
        signature = meta.get('input_sha256', '')
        source = job / 'input.gjf'
        if signature and (not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != signature):
            raise ValueError(f"Source input signature mismatch: {job}")
        records = parse_completed_opt_records(text)
        if not records:
            raise ValueError(f"No converged optimization record: {job}")
        values = meta.get('scan_values')
        if row.get('scan_type') != 'equilibrium' and values and len(records) != len(values):
            raise ValueError(f"Incomplete converged scan record set: {job}")
        cache[key] = (records, signature)
    records, signature = cache[key]
    index = len(records)-1 if row['scan_type']=='equilibrium' else int(row['point_index'])
    if not 0 <= index < len(records):
        raise ValueError(f"Invalid converged record index: {row.get('id')}")
    energy, source_frame = records[index]
    provided = float(row['e_hartree'])
    if not np.isfinite(provided) or abs(provided-energy)>1e-9:
        raise ValueError(f"Energy mismatch with converged optimization record: {row.get('id')}")
    if (list(frame.symbols)!=list(source_frame.symbols) or frame.coords.shape!=source_frame.coords.shape
            or not np.isfinite(frame.coords).all() or not np.allclose(frame.coords,source_frame.coords,atol=1e-6,rtol=0)):
        raise ValueError(f"Geometry mismatch with converged optimization record: {row.get('id')}")
    return dict(id=row.get('reaxff_id') or row['id'],parent_id=row.get('parent_id',''),
        scan_type=row['scan_type'],source_log=str(job/'input.log'),converged_record_index=index,
        e_hartree=energy,input_sha256=signature,verified=True)
