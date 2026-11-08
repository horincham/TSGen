import csv
import json

import numpy as np
import pytest

from tsgen.data.xyz import Frame, read_xyz_multiframe
from tsgen.explore.label_store import HARTREE_TO_KCALMOL, LabelRecord, LabelStore
from tsgen.utils.hashing import structure_hash


def reference_record(tmp_path, **overrides):
    directory = tmp_path / "reference"
    directory.mkdir(exist_ok=True)
    log = directory / "input.log"
    log.write_text("Optimization completed.\nNormal termination of Gaussian 16\n", encoding="utf-8")
    fields = dict(
        id="H2_graph", species_id="H2_graph", e_hartree=-1.2,
        e_kcalmol_abs=-1.2 * HARTREE_TO_KCALMOL, e_kcalmol_rel=0.0,
        iter=-1, job_dir=str(directory), log_path=str(log),
        route="# M062X/6-311G(d,p) Opt", charge=0, multiplicity=1,
    )
    return LabelRecord(**(fields | overrides))


def sampled_record(identifier, energy, **overrides):
    fields = dict(
        id=identifier, species_id="H2_graph", e_hartree=energy,
        e_kcalmol_abs=energy * HARTREE_TO_KCALMOL, e_kcalmol_rel=0.0,
        iter=0, job_dir="", log_path="", route="# M062X/6-311G(d,p) SP",
        charge=0, multiplicity=1,
    )
    return LabelRecord(**(fields | overrides))


def store_with_reference(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl", reference_path=tmp_path / "energy_references.json", require_optimized=True)
    frame = Frame(["H", "H"], np.array([[0., 0., 0.], [0.74, 0., 0.]]), "")
    store.record_reference(reference_record(tmp_path), frame)
    return store, frame


def test_optimized_species_reference_keeps_energy_and_geometry_fixed(tmp_path):
    store, frame = store_with_reference(tmp_path)
    store.append([sampled_record("first", -1.0)])
    store.append([sampled_record("second", -1.1)])
    labels = store.load().set_index("id")
    assert labels.loc["first", "e_kcalmol_rel"] == pytest.approx(0.2 * HARTREE_TO_KCALMOL)
    assert labels.loc["second", "e_kcalmol_rel"] == pytest.approx(0.1 * HARTREE_TO_KCALMOL)
    metadata = store.reference_metadata()["H2_graph"]
    assert metadata["geometry_id"] == structure_hash(frame.symbols, frame.coords)
    optimized = read_xyz_multiframe(tmp_path / "reference" / "optimized_reference.xyz")[0]
    assert np.array_equal(optimized.coords, frame.coords)
    assert labels["reference_geometry_path"].nunique() == 1
    assert labels["reference_hartree"].tolist() == pytest.approx([-1.2, -1.2])
    store.record_reference(reference_record(tmp_path), frame)
    with pytest.raises(ValueError, match="fixed optimized reference already exists"):
        store.record_reference(reference_record(tmp_path, e_hartree=-1.3, e_kcalmol_abs=-1.3 * HARTREE_TO_KCALMOL), frame)


@pytest.mark.parametrize("change,message", [
    ({"multiplicity": 3}, "electronic states"),
    ({"charge": 2}, "electronic states"),
    ({"route": "# M062X/6-31G(d) SP"}, "DFT methods"),
])
def test_reference_rejects_mismatched_state_or_method_before_append(tmp_path, change, message):
    store, _ = store_with_reference(tmp_path)
    with pytest.raises(ValueError, match=message):
        store.append([sampled_record("invalid", -1.0, **change)])
    assert not store.labels_jsonl.exists()


def test_changed_reference_provenance_is_detected_on_resume(tmp_path):
    store, _ = store_with_reference(tmp_path)
    store.append([sampled_record("first", -1.0)])
    references = json.loads(store.reference_path.read_text())
    references["H2_graph"] = -1.3 * HARTREE_TO_KCALMOL
    store.reference_path.write_text(json.dumps(references))
    with pytest.raises(ValueError, match="reference energy changed"):
        store.load()


def test_export_references_from_archived_provenance_without_restarting_explore(tmp_path):
    from typer.testing import CliRunner
    from tsgen.explore.cli import app

    store, optimized = store_with_reference(tmp_path)
    provenance = store.reference_metadata_path.read_bytes()
    (tmp_path / "reference" / "input.log").unlink()
    (tmp_path / "reference" / "optimized_reference.xyz").unlink()
    args = ["export-references", "--run-dir", str(tmp_path)]
    output = tmp_path / "exports" / "optimized_references"
    for _ in range(2):
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "Exported 1" in result.output
    with (output / "index.csv").open(newline="") as handle:
        row, = csv.DictReader(handle)
    assert row["species_id"] == "H2_graph"
    assert row["route"].endswith(" Opt")
    assert row["energy_hartree"] == "-1.2"
    xyz, = read_xyz_multiframe(output / row["xyz_file"])
    np.testing.assert_array_equal(xyz.coords, optimized.coords)
    assert store.reference_metadata_path.read_bytes() == provenance
    assert not (tmp_path / "pool").exists()


def test_imported_numeric_reference_rejects_mixed_electronic_states(tmp_path):
    path = tmp_path / "energy_references.json"
    path.write_text(json.dumps({"H2_graph": -1.2 * HARTREE_TO_KCALMOL}))
    store = LabelStore(tmp_path / "labels.jsonl", reference_path=path, require_optimized=True)
    store.append([sampled_record("singlet", -1.0)])
    before = store.labels_jsonl.read_text()
    with pytest.raises(ValueError, match="different charge or multiplicity"):
        store.append([sampled_record("triplet", -0.9, multiplicity=3)])
    assert store.labels_jsonl.read_text() == before


def test_outer_acquisition_keeps_equilibrium_geometry_provenance(tmp_path):
    from tsgen.explore.config import ExploreCfg
    from tsgen.explore.species import species_signature
    from tsgen.outerloop.bridge import initialize_explore

    origin = tmp_path / "origin"
    origin.mkdir()
    optimized = Frame(["H", "H"], np.array([[0., 0., 0.], [0.74, 0., 0.]]), "")
    sampled = Frame(["H", "H"], np.array([[0., 0., 0.], [0.72, 0., 0.]]), "")
    sid = species_signature(sampled)
    store = LabelStore(origin / "labels.jsonl", reference_path=origin / "energy_references.json", require_optimized=True)
    store.record_reference(reference_record(origin, id=sid, species_id=sid), optimized)
    iid = structure_hash(sampled.symbols, sampled.coords)
    store.append([sampled_record(iid, -1.0, species_id=sid)])
    row = store.load().iloc[0].to_dict()
    row.update(symbols=sampled.symbols, coords=sampled.coords.tolist(), method=row["route"])
    cfg = ExploreCfg.model_validate({
        "descriptor": {"backend": "coulomb_eigs", "max_atoms": 2, "normalization": "l2"},
        "labeling": {"gaussian": {"reference_mode": "optimized"}},
    })
    runner = initialize_explore(tmp_path / "next_cycle", cfg, [row], [], [], 40)
    inherited = runner.labels.reference_metadata()[sid]
    assert inherited["geometry_id"] == structure_hash(optimized.symbols, optimized.coords)
    assert inherited["coords"] == optimized.coords.tolist()
    assert runner.labels.load().iloc[0]["e_kcalmol_rel"] == pytest.approx(0.2 * HARTREE_TO_KCALMOL)
