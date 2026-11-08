"""Conformer-independent matching of final XYZ molecular graphs.

Bond perception is supplied by the caller (TSGen's existing distance rules).
Exact graph matching, rather than a graph hash, decides identity. Terminal H
atoms are implicit to avoid factorial hydrogen permutations. Tetrahedral carbon
and silicon, double-bond and cumulene configurations are checked under
the atom mapping. Search exhaustion conservatively keeps both structures.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib

import numpy as np

from tsgen.data.xyz import Frame


@dataclass
class MolecularGraph:
    frame: Frame
    adjacency: dict[int, dict[int, str]]
    hydrogens: dict[int, list[int]]
    colors: dict[int, str]
    cumulenes: list[list[int]]


def molecular_graph(frame: Frame, bonds: list[dict]) -> MolecularGraph:
    if not np.isfinite(frame.coords).all():
        raise ValueError('Nonfinite final geometry in molecular identity curation')
    full = {i: {} for i in range(len(frame.symbols))}
    for bond in bonds:
        i, j = int(bond['i']), int(bond['j'])
        full[i][j] = full[j][i] = str(bond['order'])
    implicit = {i for i, neighbors in full.items()
                if frame.symbols[i] == 'H' and len(neighbors) == 1
                and frame.symbols[next(iter(neighbors))] != 'H'}
    adjacency = {i: {j: order for j, order in neighbors.items() if j not in implicit}
                 for i, neighbors in full.items() if i not in implicit}
    hydrogens = {i: sorted(set(full[i]) & implicit) for i in adjacency}
    colors = {i: repr((frame.symbols[i], len(hydrogens[i]), len(neighbors)))
              for i, neighbors in adjacency.items()}
    # Isomorphic graphs receive the same refinement sequence. Hashes only prune
    # candidates; a full edge-preserving mapping is still required below.
    for _ in range(len(adjacency)):
        refined = {i: hashlib.sha256(repr((colors[i], sorted(
            (order, colors[j]) for j, order in neighbors.items()))).encode()).hexdigest()
            for i, neighbors in adjacency.items()}
        stable = len(set(refined.values())) == len(set(colors.values()))
        colors = refined
        if stable:
            break
    doubles = {i: [j for j, order in edges.items() if order == 'double']
               for i, edges in adjacency.items()}
    cumulenes = []
    for start, neighbors in doubles.items():
        if len(neighbors) != 1:
            continue
        path = [start, neighbors[0]]
        while len(doubles[path[-1]]) == 2:
            following = [j for j in doubles[path[-1]] if j != path[-2]][0]
            if following in path:
                break
            path.append(following)
        if len(path) >= 3 and len(doubles[path[-1]]) == 1 and start < path[-1]:
            cumulenes.append(path)
    return MolecularGraph(frame, adjacency, hydrogens, colors, cumulenes)


def _mapped_neighbor(a, b, center, neighbor, mapping):
    if neighbor in a.adjacency:
        return mapping.get(neighbor)
    # Only unique implicit H is used in stereo checks.
    other = mapping.get(center)
    if other is not None and len(b.hydrogens[other]) == 1:
        return b.hydrogens[other][0]
    return None


def _volume(coords, neighbors):
    vectors = coords[neighbors[:3]] - coords[neighbors[3]]
    denominator = float(np.prod(np.linalg.norm(vectors, axis=1)))
    return float(np.linalg.det(vectors) / denominator) if denominator > 1e-15 else 0.


def _side(coords, i, j, first, second):
    axis = coords[j]-coords[i]
    length = np.linalg.norm(axis)
    if length < 1e-12:
        return None
    axis = axis/length
    u, v = coords[first]-coords[i], coords[second]-coords[j]
    u, v = u-np.dot(u, axis)*axis, v-np.dot(v, axis)*axis
    denominator = np.linalg.norm(u)*np.linalg.norm(v)
    if denominator < 1e-12:
        return None
    return float(np.dot(u, v)/denominator)


def _axial_hand(coords, i, j, first, second):
    axis = coords[j]-coords[i]
    length = np.linalg.norm(axis)
    if length < 1e-12:
        return None
    axis = axis/length
    u, v = coords[first]-coords[i], coords[second]-coords[j]
    u, v = u-np.dot(u,axis)*axis, v-np.dot(v,axis)*axis
    denominator = np.linalg.norm(u)*np.linalg.norm(v)
    if denominator < 1e-12:
        return None
    return float(np.dot(axis,np.cross(u,v))/denominator)


def _stereo_compatible(a, b, mapping):
    # Cumulenes need the relationship between *terminal* substituents. Individual
    # C=C tests miss axial allene chirality because the central carbon is linear.
    for chain in a.cumulenes:
        i, j = chain[0], chain[-1]
        if i not in mapping or j not in mapping:
            continue
        left = sorted(set(a.adjacency[i])-{chain[1]}) + a.hydrogens[i]
        right = sorted(set(a.adjacency[j])-{chain[-2]}) + a.hydrogens[j]
        if len(left) != 2 or len(right) != 2 or len(a.hydrogens[i]) > 1 or len(a.hydrogens[j]) > 1:
            continue
        p = _mapped_neighbor(a,b,i,left[0],mapping)
        q = _mapped_neighbor(a,b,j,right[0],mapping)
        if p is None or q is None:
            continue
        measure = _axial_hand if (len(chain)-1) % 2 == 0 else _side
        first = measure(a.frame.coords,i,j,left[0],right[0])
        second = measure(b.frame.coords,mapping[i],mapping[j],p,q)
        if first is None or second is None or min(abs(first),abs(second)) < .1 or first*second < 0:
            return False
    for center, edges in a.adjacency.items():
        if center not in mapping:
            continue
        neighbors = sorted(edges) + a.hydrogens[center]
        if (a.frame.symbols[center] in {'C', 'Si'} and len(neighbors) == 4
                and len(a.hydrogens[center]) <= 1
                and all(order == 'single' for order in edges.values())):
            other_neighbors = [_mapped_neighbor(a, b, center, n, mapping) for n in neighbors]
            if all(n is not None for n in other_neighbors):
                first = _volume(a.frame.coords, neighbors)
                second = _volume(b.frame.coords, other_neighbors)
                # Unknown/planar versus a configured center is not a safe merge.
                if (abs(first) < 1e-4) != (abs(second) < 1e-4):
                    return False
                if first*second < 0 and min(abs(first), abs(second)) >= 1e-4:
                    return False
        for other, order in edges.items():
            if order != 'double' or center > other or other not in mapping:
                continue
            left = sorted(set(edges)-{other}) + a.hydrogens[center]
            right = sorted(set(a.adjacency[other])-{center}) + a.hydrogens[other]
            # Terminal =CH2 has no E/Z configuration. Each endpoint must have
            # two substituents; graph automorphisms account for equal branches.
            if len(left) != 2 or len(right) != 2 or len(a.hydrogens[center]) > 1 or len(a.hydrogens[other]) > 1:
                continue
            p = _mapped_neighbor(a, b, center, left[0], mapping)
            q = _mapped_neighbor(a, b, other, right[0], mapping)
            if p is None or q is None:
                continue
            first = _side(a.frame.coords, center, other, left[0], right[0])
            second = _side(b.frame.coords, mapping[center], mapping[other], p, q)
            # Near-orthogonal or undefined substituents do not justify merging.
            if first is None or second is None or min(abs(first), abs(second)) < .1:
                return False
            if first*second < 0:
                return False
    return True


def same_molecular_identity(a: MolecularGraph | None, b: MolecularGraph | None, max_states: int = 20000) -> bool:
    """Match connectivity, perceived bond order and supported stereochemistry.

    Charge/multiplicity are record metadata and must be compared by the caller.
    This is not a replacement for electronic-structure bond-order analysis.
    """
    if a is None or b is None or not a.adjacency or Counter(a.frame.symbols) != Counter(b.frame.symbols):
        return False
    if Counter(a.colors.values()) != Counter(b.colors.values()):
        return False
    candidates = {i: [j for j in b.adjacency if a.colors[i] == b.colors[j]] for i in a.adjacency}
    mapping, used = {}, set()
    states = 0

    def search():
        nonlocal states
        states += 1
        if states > max_states:
            return False
        if not _stereo_compatible(a, b, mapping):
            return False
        if len(mapping) == len(a.adjacency):
            return True
        choices = []
        for i, possible in candidates.items():
            if i in mapping:
                continue
            available = [j for j in possible if j not in used and all(
                a.adjacency[i].get(k) == b.adjacency[j].get(mapped)
                for k, mapped in mapping.items())]
            if not available:
                return False
            choices.append((len(available), -sum(k in mapping for k in a.adjacency[i]), i, available))
        _, _, i, available = min(choices)
        for j in available:
            if states > max_states:
                break
            mapping[i] = j
            used.add(j)
            if search():
                return True
            del mapping[i]
            used.remove(j)
        return False

    return search()
