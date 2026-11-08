import numpy as np
import pandas as pd
import pytest

from tsgen.data.xyz import Frame
from tsgen.explore.config import ExploreCfg, load_explore_config
from tsgen.explore.species import _component_hash, _graph_hash, species_signature
from tsgen.explore.trainset_export import CandidateRecord, _select_representatives
from tsgen.utils.geometry import pair_distance_fingerprint


def methylhexane_graph(branch):
    symbols = ["C"] * 7
    graph = [[] for _ in symbols]
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (branch, 6)]:
        graph[a].append(b)
        graph[b].append(a)
    for carbon in range(7):
        for _ in range(4 - len(graph[carbon])):
            hydrogen = len(symbols)
            symbols.append("H")
            graph.append([carbon])
            graph[carbon].append(hydrogen)
    return symbols, graph


def test_graph_refinement_distinguishes_methylhexane_isomers():
    a_symbols, a_graph = methylhexane_graph(1)
    b_symbols, b_graph = methylhexane_graph(2)
    indices = list(range(len(a_symbols)))
    assert _component_hash(a_symbols, a_graph, indices) == _component_hash(b_symbols, b_graph, indices)
    assert _graph_hash(a_symbols, a_graph, indices, "wl_v2") != _graph_hash(b_symbols, b_graph, indices, "wl_v2")


def test_graph_refinement_is_atom_order_invariant():
    symbols, graph = methylhexane_graph(2)
    permutation = np.random.default_rng(32).permutation(len(symbols))
    inverse = {int(old): new for new, old in enumerate(permutation)}
    reordered_symbols = [symbols[int(old)] for old in permutation]
    reordered_graph = [[inverse[neighbor] for neighbor in graph[int(old)]] for old in permutation]
    indices = list(range(len(symbols)))
    assert _graph_hash(symbols, graph, indices, "wl_v2") == _graph_hash(reordered_symbols, reordered_graph, indices, "wl_v2")
    water = Frame(["O", "H", "H"], np.array([[0., 0., 0.], [.96, 0., 0.], [-.24, .93, 0.]]), "")
    swapped = Frame([water.symbols[i] for i in [2, 0, 1]], water.coords[[2, 0, 1]], "")
    assert species_signature(water) == species_signature(swapped)
    assert "_v2" in species_signature(water)


def test_old_run_config_keeps_legacy_species_ids(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("species: {bond_tol: 1.2}\n", encoding="utf-8")
    assert load_explore_config(config).species.fingerprint_version == "wl_v2"
    (tmp_path / "pool").mkdir()
    pd.DataFrame({"id": []}).to_csv(tmp_path / "pool" / "pool_index.csv", index=False)
    assert load_explore_config(config).species.fingerprint_version == "legacy"
    config.write_text("species: {fingerprint_version: wl_v2}\n", encoding="utf-8")
    assert load_explore_config(config).species.fingerprint_version == "wl_v2"


def test_global_representative_cap_takes_one_per_species_before_extra_conformers(tmp_path):
    records = []
    for species, distances in [("H2_dense", [.68, .70, .74]), ("H2_sparse_b", [.71]), ("H2_sparse_c", [.72])]:
        for position, distance in enumerate(distances):
            frame = Frame(["H", "H"], np.array([[0., 0., 0.], [distance, 0., 0.]]), "")
            records.append(CandidateRecord(
                id=f"{species}_{position}", species_id=species, iter=0, is_seed=False,
                e_hartree=-1.0, e_kcalmol_abs=-627.0, e_kcalmol_rel=float(position),
                charge=0, multiplicity=1, job_dir=tmp_path, source_xyz=tmp_path / "source.xyz",
                frame=frame, fingerprint=pair_distance_fingerprint(frame),
            ))
    cfg = ExploreCfg.model_validate({"export": {"selection": {
        "max_total": 3, "max_per_species": 3, "fingerprint_tol": 0.0,
    }}})
    selected, decisions = _select_representatives(records, cfg)
    assert {record.species_id for record in selected} == {"H2_dense", "H2_sparse_b", "H2_sparse_c"}
    assert decisions["keep_for_trainset"].sum() == 3
