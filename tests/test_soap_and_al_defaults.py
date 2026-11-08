from __future__ import annotations

import pytest

import sys
import unittest
from pathlib import Path

import numpy as np

from tsgen.data.xyz import Frame
from tsgen.explore.config import ExploreCfg
from tsgen.explore.surrogate.features import compute_features


class SoapDescriptorTests(unittest.TestCase):
    def test_explore_defaults_use_mainstream_soap_gp_uncertainty_diversity(self):
        cfg = ExploreCfg()

        self.assertEqual(cfg.descriptor.backend, "soap")
        self.assertEqual(cfg.surrogate.kind, "gp")
        self.assertEqual(cfg.acquisition.method, "sigma")
        self.assertEqual(cfg.acquisition.diversity.method, "farthest_in_descriptor")

    def test_coulomb_descriptor_remains_available_as_fallback(self):
        cfg = ExploreCfg()
        cfg.descriptor.backend = "coulomb_eigs"
        cfg.descriptor.max_atoms = 8
        frame = Frame(
            symbols=["C", "H", "H", "H", "H"],
            coords=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.09],
                    [1.03, 0.0, -0.36],
                    [-0.52, 0.89, -0.36],
                    [-0.52, -0.89, -0.36],
                ],
                dtype=float,
            ),
            comment="methane",
        )

        X = compute_features([frame], cfg.descriptor)

        self.assertEqual(X.shape, (1, 8))
        self.assertTrue(np.isfinite(X).all())

    def test_soap_descriptor_backend_is_recognized(self):
        cfg = ExploreCfg()
        cfg.descriptor.backend = "soap"
        frame = Frame(
            symbols=["H", "H"],
            coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float),
            comment="hydrogen",
        )

        try:
            X = compute_features([frame], cfg.descriptor)
        except ImportError as exc:
            self.assertIn("DScribe", str(exc))
            self.assertIn("ASE", str(exc))
        else:
            self.assertEqual(X.shape[0], 1)
            self.assertGreater(X.shape[1], 0)
            self.assertTrue(np.isfinite(X).all())


if __name__ == "__main__":
    unittest.main()
def test_descriptor_resume_keeps_verified_cache_with_roundoff(tmp_path):
    from tsgen.descriptors.preprocessing import preprocess_features
    from tsgen.explore.config import DescriptorCfg

    cfg = DescriptorCfg(normalization="l2")
    raw = np.array([[0.2, 1.0], [3.0, 0.4]])
    before = preprocess_features(raw, cfg, tmp_path)
    recomputed = raw + 1e-10
    after = preprocess_features(recomputed, cfg, tmp_path)
    np.testing.assert_array_equal(after, before)
    with pytest.raises(ValueError, match="descriptors/preprocessing changed"):
        preprocess_features(raw + 1e-4, cfg, tmp_path)
    np.save(tmp_path / "features_raw.npy", recomputed)
    with pytest.raises(ValueError, match="descriptors/preprocessing changed"):
        preprocess_features(recomputed, cfg, tmp_path)
