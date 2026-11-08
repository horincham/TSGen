"""Manuscript HSIC aggregation on synthetic responses."""

import numpy as np

from tsgen.sensitivity.analysis import _summary_from_repeats
from tsgen.sensitivity.hsic import (
    conjunctive_gaussian_kernel,
    gaussian_kernel,
    repeated_hsic,
    transform_loss,
    unbiased_hsic,
)


def test_normalization_and_selection_happen_within_each_subsample():
    rng = np.random.default_rng(8)
    parameters = rng.random((32, 3))
    loss = np.exp(4 * parameters[:, 0] + 2 * parameters[:, 1])
    loss[-1] = 1e8
    repeats, size, seed = 4, 14, 17

    result = repeated_hsic(
        parameters, loss, ["a", "b", "c"], repeats, size, seed,
        selection_threshold=0.8,
    )
    scaled_loss = transform_loss(loss)
    sampler = np.random.default_rng(seed)
    expected_raw = []
    expected_normalized = []
    expected_selected = []
    for _ in range(repeats):
        rows = sampler.choice(len(loss), size=size, replace=False)
        loss_kernel = conjunctive_gaussian_kernel(scaled_loss[rows], 0.3)
        raw = np.array([
            unbiased_hsic(gaussian_kernel(column, 0.3), loss_kernel)
            for column in parameters[rows].T
        ])
        positive = raw.clip(min=0)
        normalized = positive / positive.sum()
        order = np.argsort(-normalized, kind="stable")
        cutoff = np.searchsorted(np.cumsum(normalized[order]), 0.8) + 1
        expected_raw.extend(raw)
        expected_normalized.extend(normalized)
        expected_selected.extend(np.isin(np.arange(3), order[:cutoff]))

    np.testing.assert_allclose(result["raw_hsic"], expected_raw)
    np.testing.assert_allclose(result["normalized_hsic"], expected_normalized)
    np.testing.assert_array_equal(result["selected"], expected_selected)

    summary = _summary_from_repeats(result, "TOTAL").set_index("parameter_id").loc[["a", "b", "c"]]
    np.testing.assert_allclose(
        summary["hsic_mean"],
        np.asarray(expected_normalized).reshape(repeats, 3).mean(axis=0),
    )
    np.testing.assert_allclose(
        summary["selection_frequency"],
        np.asarray(expected_selected).reshape(repeats, 3).mean(axis=0),
    )
