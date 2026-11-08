"""Outer metrics use the pre-acquisition pool and a frozen historical archive."""
from __future__ import annotations
import numpy as np
import pandas as pd


def nearest_distances(X: np.ndarray, reference: np.ndarray, block: int = 128) -> np.ndarray:
    X, reference = np.asarray(X, dtype=float), np.asarray(reference, dtype=float)
    if X.ndim != 2 or reference.ndim != 2 or not len(X) or not len(reference):
        raise ValueError('Cannot assess empty candidate/reference matrix')
    if X.shape[1] != reference.shape[1] or not np.isfinite(X).all() or not np.isfinite(reference).all():
        raise ValueError('Invalid descriptor matrices')
    result = []
    for start in range(0, len(X), block):
        q = X[start:start + block]
        best = np.full(len(q), np.inf)
        for j in range(0, len(reference), block):
            r = reference[j:j + block]
            d2 = np.sum(q*q, axis=1)[:, None] + np.sum(r*r, axis=1)[None, :] - 2*q @ r.T
            best = np.minimum(best, np.maximum(d2, 0).min(axis=1))
        result.extend(np.sqrt(best))
    return np.asarray(result)


def energy_disagreement(table: pd.DataFrame, species: dict[str, str], references: dict[str, str]):
    required = {'id', 'species_id', 'reference_id', 'old_energy', 'new_energy', 'old_reference', 'new_reference'}
    if not required.issubset(table.columns):
        raise ValueError(f'Energy table missing columns: {required-set(table.columns)}')
    t = table.copy()
    for key in ['id','species_id','reference_id']:
        if t[key].isna().any() or not t[key].astype(str).str.strip().ne('').all():
            raise ValueError(f'Missing energy identity field: {key}')
    t['id'] = t['id'].astype(str)
    if t['id'].duplicated().any() or set(t['id']) != set(species) or not species:
        raise ValueError('Energy table IDs must exactly cover candidates')
    nums = ['old_energy', 'new_energy', 'old_reference', 'new_reference']
    if not np.isfinite(t[nums].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite energies')
    if any(species[i]!=str(s) for i,s in zip(t['id'],t['species_id'])):
        raise ValueError('Energy species mismatch')
    for sid, group in t.groupby('species_id'):
        ref = references.get(str(sid))
        if ref is None or set(group['reference_id'].astype(str)) != {ref}:
            raise ValueError('Inconsistent shared reference identity')
        if any(species[i] != str(sid) for i in group['id']):
            raise ValueError('Energy species mismatch')
        for col in ['old_reference', 'new_reference']:
            if not np.allclose(group[col], group[col].iloc[0], atol=1e-9, rtol=0):
                raise ValueError('Inconsistent reference energies')
        refrow = t[t['id'] == ref]
        if not refrow.empty:
            if abs(float(refrow['old_energy'].iloc[0])-float(group['old_reference'].iloc[0])) > 1e-8 or abs(float(refrow['new_energy'].iloc[0])-float(group['new_reference'].iloc[0])) > 1e-8:
                raise ValueError('Reference geometry energy mismatch')
    t['old_relative_kcalmol'] = t['old_energy'] - t['old_reference']
    t['new_relative_kcalmol'] = t['new_energy'] - t['new_reference']
    t['delta_ff_kcalmol'] = abs(t['new_relative_kcalmol'] - t['old_relative_kcalmol'])
    return t, float(t['delta_ff_kcalmol'].mean())
