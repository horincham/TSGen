import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tsgen.data.xyz import Frame, write_xyz_multiframe
from tsgen.explore.config import ExploreCfg, load_explore_config
from tsgen.explore.oracle_gaussian import GaussianJobSpec
from tsgen.explore.species import species_signature
from tsgen.outerloop.archive import species_reference_energies
from tsgen.outerloop.bridge import initialize_explore, reconcile_jobs
from tsgen.outerloop.candidates import extract
from tsgen.outerloop.config import OuterConfig, StopConfig
from tsgen.outerloop.runner import OuterRunner
from tsgen.utils.hashing import structure_hash


def test_outerloop_allocates_runs_and_retains_explicit_paths(tmp_path, monkeypatch):
    from tsgen.outerloop.config import load_config

    inputs = {key: key for key in ("ffield", "geo", "trainset", "control", "params")}
    for path in inputs.values():
        (tmp_path / path).touch()
    inputs["labels"] = "labels.json"
    symbols = ["H", "H"]
    coords = [[0., 0., 0.], [.74, 0., 0.]]
    (tmp_path / "labels.json").write_text(json.dumps([dict(
        id=structure_hash(symbols, np.asarray(coords)),
        species_id="H2", symbols=symbols, coords=coords,
        charge=0, multiplicity=1, e_hartree=-1.2,
        method="# M062X/6-311G(d,p) SP",
    )]))
    monkeypatch.setattr("tsgen.outerloop.runner.validate_fit_pair", lambda *args: None)
    monkeypatch.setattr("tsgen.outerloop.runner.validate_ffield", lambda *args: None)
    config_path = tmp_path / "outerloop.yaml"
    data = dict(initial=inputs, md=dict(coordinates="unwrapped"))
    config_path.write_text(json.dumps(data))
    cfg = load_config(config_path)
    first = OuterRunner.create(cfg)
    second = OuterRunner.create(cfg)
    assert first.root == tmp_path / "output/outerloop_run_0001"
    assert second.root == tmp_path / "output/outerloop_run_0002"
    assert OuterRunner(first.root).cfg.run_dir == first.root
    assert cfg.run_dir is None
    config_path.write_text(json.dumps(data | {"run_dir": "existing_convention"}))
    explicit = load_config(config_path)
    assert OuterRunner.create(explicit).root == tmp_path / "existing_convention"
    with pytest.raises(ValueError, match="use resume"):
        OuterRunner.create(explicit)


def test_optimized_reference_is_retained_over_sampled_energy():
    archive = [
        {"species_id": "H2", "e_hartree": -1.0, "reference_hartree": -1.2},
        {"species_id": "H2", "e_hartree": -1.1, "reference_hartree": -1.2},
    ]
    assert species_reference_energies(archive, require_optimized=True) == {"H2": -1.2}
    with pytest.raises(ValueError, match="Missing optimized"):
        species_reference_energies(
            [{"species_id": "H2", "e_hartree": -1.0}], require_optimized=True
        )
    archive[1]["reference_hartree"] = -1.25
    with pytest.raises(ValueError, match="Conflicting"):
        species_reference_energies(archive, require_optimized=True)


def test_outer_fixed_pool_does_not_require_quickstart_support_frames(tmp_path):
    config = ExploreCfg.model_validate({
        "descriptor": {"backend": "coulomb_eigs", "max_atoms": 2},
        "candidate_pool": {"balance": {"enabled": True, "method": "representative"}},
        "acquisition": {"representative": {}},
    })
    reference = Frame(["H", "H"], np.array([[0., 0., 0.], [.74, 0., 0.]]), "")
    candidate = Frame(["H", "H"], np.array([[0., 0., 0.], [.79, 0., 0.]]), "")
    archive = [dict(
        id=structure_hash(reference.symbols, reference.coords), species_id=species_signature(reference),
        symbols=reference.symbols, coords=reference.coords.tolist(), e_hartree=-1.2,
        reference_hartree=-1.2, method="# M062X/6-311G(d,p) SP", charge=0, multiplicity=1,
    )]
    proposals = [dict(id=structure_hash(candidate.symbols, candidate.coords),
                      symbols=candidate.symbols, coords=candidate.coords.tolist())]

    runner = initialize_explore(tmp_path, config, archive, proposals, [], 1)

    assert load_explore_config(tmp_path / "config.yaml").acquisition.representative is None
    assert config.acquisition.representative is not None
    assert runner.pool.df.iloc[runner._select_next_batch(1)]["id"].tolist() == [proposals[0]["id"]]


def test_outer_acquisition_reports_terminal_species_reference_failure(tmp_path):
    config = ExploreCfg.model_validate({
        "descriptor": {"backend": "coulomb_eigs", "max_atoms": 2},
        "labeling": {"gaussian": {"mode": "manual", "reference_mode": "optimized"}},
    })
    candidate = Frame(["H", "H"], np.array([[0., 0., 0.], [.74, 0., 0.]]), "")
    candidate_id = structure_hash(candidate.symbols, candidate.coords)
    species_id = species_signature(candidate)
    proposals = [dict(id=candidate_id, symbols=candidate.symbols, coords=candidate.coords.tolist())]
    runner = initialize_explore(tmp_path, config, [], proposals, [], 1)
    spec = GaussianJobSpec(candidate_id, species_id, candidate)
    single_point = runner.paths.jobs_dir(0) / f"job_{candidate_id}"
    runner.oracle.write_job(single_point, spec)
    (single_point / "input.log").write_text(" SCF Done: E(RHF) = -1.1 A.U.\n Normal termination\n")
    runner.reference_oracle.prepare(spec)
    reference_log = runner.reference_oracle.job_dir(species_id) / "input.log"
    reference_log.write_text(" Error termination\n")
    runner._collect_oracle_labels()

    with pytest.raises(RuntimeError, match=f"Species reference optimization failed for {species_id}") as error:
        reconcile_jobs(runner)
    assert str(reference_log) in str(error.value)


@pytest.mark.parametrize("paired", [False, True])
def test_md_candidates_only_use_updated_force_field(tmp_path, paired):
    from tsgen.outerloop.io import atomic_json, sha256

    system = tmp_path / "system.xyz"
    system.write_text("initial conditions")
    old_xyz = tmp_path / "old.xyz"
    new_xyz = tmp_path / "new.xyz"
    conditions = tmp_path / "conditions.json"
    atomic_json(conditions, dict(
        input_hashes={"system": sha256(system)},
        coordinates="unwrapped",
        new={"temperature": 3500},
        **({"old": {"temperature": 3500}} if paired else {}),
    ))
    runner = OuterRunner.__new__(OuterRunner)
    runner.directory = tmp_path
    runner.cfg = SimpleNamespace(md=SimpleNamespace(inputs={"system": system}))
    runner.state = {"new_ffield": str(tmp_path / "ffield")}
    runner._task = lambda *args, **kwargs: {"outputs": {
        "conditions": {"path": str(conditions)},
        "new_xyz": {"path": str(new_xyz)},
        **({"old_xyz": {"path": str(old_xyz)}} if paired else {}),
    }}
    streams = {}
    runner._extract_md_candidates = streams.update
    runner._md()
    assert streams == {"new_xyz": new_xyz}


def test_outer_candidates_use_shared_geometry_and_chemistry_screening(tmp_path):
    def dimer(element, distance):
        coords = np.array([[0., 0., 0.], [distance, 0., 0.]])
        return Frame([element, element], coords, "reference_id=dimer")

    accepted, stretched, wrong_element = (
        dimer("H", .74), dimer("H", .94), dimer("He", .8)
    )
    md = tmp_path / "new.xyz"
    write_xyz_multiframe(md, [accepted, stretched, wrong_element])
    references = tmp_path / "references.json"
    references.write_text(json.dumps({"dimer": {"symbols": ["H", "H"], "bonds": [[0, 1]]}}))
    explore = tmp_path / "explore.yaml"
    explore.write_text(json.dumps({
        "candidate_pool": {
            "split_fragments": False,
            "chemistry_filter": {"enabled": True, "allowed_elements": ["H"]},
        },
        "gating": {"max_bond_ratio": 1.5, "reference_bonds_file": "references.json"},
    }))
    cfg = OuterConfig.model_validate(dict(
        run_dir=tmp_path / "run",
        initial=dict(ffield="ffield", labels="labels.json", geo="geo", trainset="trainset.in",
                     control="control", params="params-ml"),
        md=dict(coordinates="unwrapped", split_fragments=False, bond_tol=1.6),
        descriptor=dict(backend="coulomb_eigs", max_atoms=2, normalization="l2"),
        explore_config=explore,
    ))
    archive_frame = dimer("H", .7)
    archive = [dict(id=structure_hash(archive_frame.symbols, archive_frame.coords),
                    symbols=archive_frame.symbols, coords=archive_frame.coords.tolist(),
                    species_id="fixture")]
    candidates = extract({"new": md}, archive, [], cfg, tmp_path)
    candidate_id = structure_hash(accepted.symbols, accepted.coords)
    assert [r["id"] for r in candidates] == [candidate_id]
    protected_dir = tmp_path / "protected"
    protected_dir.mkdir()
    with pytest.raises(ValueError, match="No valid candidates"):
        extract({"new": md}, archive, [], cfg, protected_dir, protected_ids={candidate_id})


@pytest.mark.parametrize("acquire_on_final_cycle", [False, True])
def test_final_assessment_can_request_one_more_acquisition(tmp_path, acquire_on_final_cycle):
    candidates = [{"id": "a", "species_id": "H2", "d_nn": .2, "new_species": False}]
    (tmp_path / "candidates.json").write_text(json.dumps(candidates))
    (tmp_path / "reference_geometries.json").write_text(json.dumps([{"id": "ref", "species_id": "H2"}]))
    energies = tmp_path / "engine.csv"
    pd.DataFrame([dict(id="a", species_id="H2", reference_id="ref",
                       old_energy=1., new_energy=2., old_reference=0., new_reference=0.)]).to_csv(energies, index=False)
    validation = tmp_path / "validation.json"
    validation.write_text('{"metrics": {}, "reason": "fixture"}')

    runner = OuterRunner.__new__(OuterRunner)
    runner.root = runner.directory = tmp_path
    runner.cfg = SimpleNamespace(existing_md=None, validation_limits={},
                                 stop=StopConfig(max_cycles=1, acquire_on_final_cycle=acquire_on_final_cycle))
    runner.state = dict(cycle=1, history=[], dft_used=0, archive=str(energies),
                        ffield=str(energies), new_ffield=str(energies))
    runner._task = lambda *args, **kwargs: {"outputs": {
        "energies": {"path": str(energies)}, "validation": {"path": str(validation)},
    }}
    runner._track = lambda *args: None
    runner._finish = lambda reason: runner.state.update(status=reason)
    runner._evaluate()
    if acquire_on_final_cycle:
        assert runner.state["stage"] == "acquire"
    else:
        assert runner.state["status"] == "MAX_CYCLES"
