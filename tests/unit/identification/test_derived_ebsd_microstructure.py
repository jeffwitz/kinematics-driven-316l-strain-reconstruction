"""Unit checks for the global EBSD-product preflight contract."""

import numpy as np

from scripts.build_derived_ebsd_microstructure import _orientation_labels, _orientation_preflight


def test_exact_orientation_plateaus_are_labelled_without_angular_tolerance() -> None:
    angles = np.asarray(
        [
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
        ]
    )
    labels, unique = _orientation_labels(angles)
    assert labels.tolist() == [[0, 0], [1, 1]]
    np.testing.assert_allclose(unique, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_preflight_rejects_a_non_separated_positive_difference_distribution() -> None:
    row = np.linspace(0.0, 1.0, 12)
    angles = np.stack((row[:, None], np.zeros((12, 1)), np.zeros((12, 1))), axis=-1)
    result = _orientation_preflight(angles)
    assert result["clear_numeric_gap_detected"] is False
    assert result["decision"] == "diagnostic only; no segmentation gate"
