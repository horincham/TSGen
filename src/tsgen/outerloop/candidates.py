"""Candidate extraction from explicitly unwrapped XYZ streams."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from tsgen.data.xyz import Frame, read_xyz_multiframe, write_xyz_multiframe
from tsgen.descriptors.features import compute_descriptor_matrix
from tsgen.explore.config import TrajXYZSourceCfg, load_explore_config
from tsgen.explore.gating import PhysicsGater
from tsgen.explore.oracle_gaussian import _resolve_charge_mult
from tsgen.explore.pool import CandidatePool
from tsgen.explore.species import species_signature, split_frame_components
from tsgen.filters.physics import PhysicsFilter
from tsgen.utils.hashing import structure_hash
from tsgen.outerloop.fitting import geometry_key
from tsgen.outerloop.archive import frames_from_records
from tsgen.outerloop.assessment import nearest_distances
from tsgen.outerloop.io import atomic_json


def _screen_pool(frames, cfg, directory):
    explore = load_explore_config(cfg.explore_config)
    pool_config = explore.candidate_pool
    source = directory/'fragments.xyz'
    write_xyz_multiframe(source, frames, precision=16)
    pool_config.sources = [TrajXYZSourceCfg(path=str(source))]
    pool_config.split_fragments = False
    explore.species.bond_tol = cfg.md.bond_tol
    explore.species.fingerprint_version = cfg.md.fingerprint_version
    pool = CandidatePool(pool_config, cfg.descriptor, explore.species)
    pool.initialize(directory/'preprocessing')
    _, screened = PhysicsGater(explore.gating).filter(pool.frames, pool.df)
    screened.to_csv(directory/'geometric_screening.csv', index=False)
    return screened.loc[screened['keep'], 'id'].astype(str).tolist()


def extract(streams: dict[str, Path], archive: list[dict], protected: list[Frame], cfg, directory: Path,
            *, protected_ids: set[str] | None = None) -> list[dict]:
    forbidden = {geometry_key(f) for f in protected}
    protected_ids = protected_ids or set()
    result, original_frames, audit = {}, {}, []
    legacy_filter = PhysicsFilter(min_distance_angstrom=cfg.md.min_distance,
                                  angle_limits_deg=(cfg.md.min_angle_deg, 180.))
    for stream, path in streams.items():
        for frame_idx, frame in enumerate(read_xyz_multiframe(path)):
            if not frame.symbols or not np.isfinite(frame.coords).all():
                raise ValueError('Invalid MD geometry')
            fragments = split_frame_components(frame, cfg.md.bond_tol, fingerprint_version=cfg.md.fingerprint_version) if cfg.md.split_fragments else [(frame, species_signature(frame,cfg.md.bond_tol, fingerprint_version=cfg.md.fingerprint_version))]
            for fragment_idx, (fr, sid) in enumerate(fragments):
                rid = structure_hash(fr.symbols, fr.coords)
                reason = ''
                if len(fr.symbols) < cfg.md.min_fragment_atoms:
                    reason = 'small_fragment'
                elif rid in protected_ids or (forbidden and geometry_key(fr) in forbidden):
                    reason = 'protected_test'
                elif not cfg.explore_config:
                    _, flags = legacy_filter.apply([fr])
                    if not flags[0]['keep']:
                        reason = str(flags[0]['reasons'])
                provenance = dict(stream=stream, frame=frame_idx, fragment=fragment_idx)
                audit.append(dict(id=rid, reason=reason, **provenance))
                if reason:
                    continue
                if rid not in result:
                    result[rid] = dict(id=rid, symbols=fr.symbols, coords=fr.coords.tolist(), species_id=sid, provenance=[])
                    original_frames[rid] = fr
                result[rid]['provenance'].append(provenance)
    ids, frames = list(result), list(original_frames.values())
    if cfg.explore_config and frames:
        ids = _screen_pool(frames, cfg, directory)
        selected = set(ids)
        for row in audit:
            if not row['reason'] and row['id'] not in selected:
                row['reason'] = 'excluded_by_preprocessing'
    pd.DataFrame(audit).to_csv(directory/'candidate_audit.csv',index=False)
    if not ids:
        raise ValueError('No valid candidates; empty exploration is not convergence')
    candidates = [result[rid] for rid in ids]
    frames = [Frame(r['symbols'], np.asarray(r['coords']), f'id={r["id"]}') for r in candidates]
    reference_frames = frames_from_records(archive)
    X = compute_descriptor_matrix(frames + reference_frames, cfg.descriptor)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if np.any(norms <= 1e-15):
        raise ValueError('Zero descriptor cannot define normalized novelty')
    X /= norms
    distances = nearest_distances(X[:len(frames)], X[len(frames):])
    known = {r['species_id'] for r in archive}
    for r, distance in zip(candidates, distances):
        q,m,_=_resolve_charge_mult(Frame(r['symbols'],np.asarray(r['coords']),''),cfg.fragment_charge,cfg.fragment_multiplicity)
        r['charge'],r['multiplicity']=q,m
        r['d_nn'] = float(distance)
        r['new_species'] = r['species_id'] not in known
    write_xyz_multiframe(directory/'candidates.xyz', frames, precision=16)
    atomic_json(directory/'candidates.json',candidates)
    np.save(directory/'candidate_features.npy',X[:len(frames)])
    return candidates
