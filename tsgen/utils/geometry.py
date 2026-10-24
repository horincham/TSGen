"""Element-aware invariant distance fingerprints (a redundancy heuristic)."""
from collections import Counter

import numpy as np


def composition_key(frame):
    return tuple(sorted(Counter(frame.symbols).items()))


def pair_distance_fingerprint(frame):
    syms, xyz = frame.symbols, np.asarray(frame.coords, dtype=float)
    if not np.isfinite(xyz).all():
        raise ValueError('Nonfinite geometry cannot be fingerprinted')
    pairs = []
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            pairs.append(('-'.join(sorted((syms[i], syms[j]))), float(np.linalg.norm(xyz[i] - xyz[j]))))
    return tuple(sorted(pairs))


def fingerprint_distance(fp1, fp2):
    if len(fp1) != len(fp2) or any(a[0] != b[0] for a, b in zip(fp1, fp2)):
        return float('inf')
    return max((abs(a[1] - b[1]) for a, b in zip(fp1, fp2)), default=0.0)
