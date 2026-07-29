from __future__ import annotations

import numpy as np

from fem_inhouse.workflows.dic_uncertainty_propagation import (
    _ranking_probabilities,
    _summaries,
    contiguous_residual_on_support,
    periodic_residual_on_support,
)


def test_periodic_residual_samples_nodal_support_and_sign() -> None:
    flow = np.arange(4 * 5 * 2, dtype=float).reshape(4, 5, 2)
    sampled = periodic_residual_on_support(
        flow,
        solve_bounds=(3, 5, 4, 6),
        shift_x=2,
        shift_y=3,
        sign=-1,
    )
    expected_x = (np.arange(3, 6) + 2) % 4
    expected_y = (np.arange(4, 7) + 3) % 5
    assert sampled.shape == (3, 3, 2)
    np.testing.assert_array_equal(sampled, -flow[np.ix_(expected_x, expected_y)])


def test_contiguous_residual_never_introduces_a_wrap_join() -> None:
    flow = np.arange(7 * 8 * 2, dtype=float).reshape(7, 8, 2)
    sampled = contiguous_residual_on_support(
        flow,
        support_shape=(3, 4),
        origin_x=2,
        origin_y=3,
        sign=1,
    )
    np.testing.assert_array_equal(sampled, flow[2:5, 3:7])


def test_surrogate_summaries_and_rankings_are_deterministic() -> None:
    rows = []
    for sample in range(4):
        for label, alpha, offset in (("local", 0.0, 1.0), ("a1", 1.0, 0.0)):
            rows.append(
                {
                    "sample": sample,
                    "label": label,
                    "alpha": alpha,
                    "rmse": sample + offset,
                    "relative_l2": sample + offset,
                    "pearson": 1.0 - offset,
                    "top10_iou": 1.0 - offset,
                    "absolute_q90_iou": 1.0 - offset,
                    "absolute_q90_active_fraction_error": offset,
                }
            )
    summaries = _summaries(rows)
    rankings = _ranking_probabilities(rows)
    assert [entry["label"] for entry in summaries] == ["a1", "local"]
    assert summaries[0]["metrics"]["rmse"]["median"] == 1.5
    for metric in rankings:
        assert rankings[metric]["a1"] == 1.0
        assert rankings[metric]["local"] == 0.0
