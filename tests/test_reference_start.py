import json

import numpy as np
import pytest

from tsgen.data.xyz import Frame, read_xyz_multiframe
from tsgen.explore.config import GaussianCfg
from tsgen.explore.label_store import LabelStore
from tsgen.explore.oracle_gaussian import GaussianJobSpec, GaussianOracle
from tsgen.explore.references import OptimizedReferences
from tsgen.explore.species import species_signature


def _reference_workflow(tmp_path, cfg=None, fingerprint_version="wl_v2"):
    cfg = cfg or GaussianCfg()
    labels = LabelStore(
        tmp_path / "labels.jsonl",
        reference_path=tmp_path / "energy_references.json",
        require_optimized=True,
    )
    return OptimizedReferences(
        cfg, 1.2, tmp_path / "references", labels, fingerprint_version=fingerprint_version
    )


def _hydrogen_pair(distance):
    return Frame(["H", "H"], np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]), "")


def test_reference_can_start_at_a_different_geometry_without_changing_sp(tmp_path):
    cfg = GaussianCfg(charge=1, multiplicity=2)
    refs = _reference_workflow(tmp_path, cfg)
    sample = _hydrogen_pair(0.74)
    representative = _hydrogen_pair(0.71)
    candidate = GaussianJobSpec("sp", species_signature(sample), sample)

    refs.prepare(candidate, reference_frame=representative)
    sp_dir = tmp_path / "sp"
    GaussianOracle(cfg).write_job(sp_dir, candidate)

    opt_dir = refs.job_dir(candidate.species_id)
    opt_frame, = read_xyz_multiframe(opt_dir / "structure.xyz")
    sp_frame, = read_xyz_multiframe(sp_dir / "structure.xyz")
    np.testing.assert_array_equal(opt_frame.coords, representative.coords)
    np.testing.assert_array_equal(sp_frame.coords, sample.coords)
    for job in (opt_dir, sp_dir):
        meta = json.loads((job / "meta.json").read_text(encoding="utf-8"))
        assert (meta["charge"], meta["multiplicity"]) == (1, 2)

    before = {name: (opt_dir / name).read_bytes() for name in ("structure.xyz", "input.gjf", "meta.json")}
    refs.prepare(candidate, reference_frame=_hydrogen_pair(0.72))
    assert before == {name: (opt_dir / name).read_bytes() for name in before}


def test_reference_rejects_changed_fingerprint_before_preparing_job(tmp_path):
    refs = _reference_workflow(tmp_path)
    sample = _hydrogen_pair(0.74)
    candidate = GaussianJobSpec("sp", species_signature(sample), sample)

    with pytest.raises(ValueError, match="fingerprint"):
        refs.prepare(candidate, reference_frame=_hydrogen_pair(1.0))

    assert not refs.job_dir(candidate.species_id).exists()


@pytest.mark.parametrize(
    ("extra_hydrogens", "message"),
    [(1, "Electronic state differs"), (2, "Composition differs")],
)
def test_reference_rejects_mismatches_hidden_by_legacy_fingerprint(tmp_path, extra_hydrogens, message):
    refs = _reference_workflow(tmp_path, fingerprint_version="legacy")
    coords = np.array(
        [[x + offset, 0.0, 0.0] for x in (0.0, 4.0, 8.0, 12.0) for offset in (0.0, 0.74)]
    )
    sample = Frame(["H"] * 8, coords, "")
    extra_coords = [[20.0 + 4.0 * i, 0.0, 0.0] for i in range(extra_hydrogens)]
    extra_h = Frame(["H"] * (8 + extra_hydrogens), np.vstack((coords, extra_coords)), "")
    sid = species_signature(sample, fingerprint_version="legacy")
    assert species_signature(extra_h, fingerprint_version="legacy") == sid

    with pytest.raises(ValueError, match=message):
        refs.prepare(GaussianJobSpec("sp", sid, sample), reference_frame=extra_h)

    assert not refs.job_dir(sid).exists()
