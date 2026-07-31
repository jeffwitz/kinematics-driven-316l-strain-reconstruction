from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.validation.residual_structure import (
    OVER_AMPLITUDE,
    TOO_NARROW,
    TOO_WIDE,
    UNDER_AMPLITUDE,
    UNSTRUCTURED,
    classify_residual,
    directional_variogram,
    energy_partition,
    radial_power_spectrum,
    residual_associations,
    signed_residual,
)

SHAPE = (80, 120)


def _band(centre: float = 40.0, width: float = 5.0, amplitude: float = 1.0) -> np.ndarray:
    rows = np.arange(SHAPE[0], dtype=np.float64)[:, None]
    return amplitude * np.exp(-0.5 * ((rows - centre) / width) ** 2) * np.ones((1, SHAPE[1]))


def _corridor_and_flanks(centre=40, half=6, flank=14):
    corridor = np.zeros(SHAPE, dtype=bool)
    corridor[centre - half : centre + half, :] = True
    wide = np.zeros(SHAPE, dtype=bool)
    wide[centre - flank : centre + flank, :] = True
    return corridor, wide & ~corridor


def test_the_sign_convention_makes_a_positive_residual_missing_strain() -> None:
    reference = _band(amplitude=1.0)
    candidate = _band(amplitude=0.5)

    residual = signed_residual(reference, candidate)

    assert float(np.max(residual)) > 0.0


def test_excluded_pixels_become_nan() -> None:
    mask = np.ones(SHAPE, dtype=bool)
    mask[0, 0] = False

    residual = signed_residual(_band(), _band(), valid_mask=mask)

    assert np.isnan(residual[0, 0])
    assert np.isfinite(residual[10, 10])


def test_energy_partition_separates_corridor_from_background() -> None:
    corridor, _ = _corridor_and_flanks()
    residual = np.zeros(SHAPE)
    residual[corridor] = 1.0

    partition = energy_partition(residual, corridor=corridor)

    assert partition.corridor_fraction == pytest.approx(1.0)
    assert partition.background == pytest.approx(0.0)


def test_energy_partition_reports_a_background_dominated_residual() -> None:
    corridor, _ = _corridor_and_flanks()
    residual = np.ones(SHAPE)
    residual[corridor] = 0.0

    partition = energy_partition(residual, corridor=corridor)

    assert partition.corridor_fraction == pytest.approx(0.0)


def test_a_narrow_candidate_shows_a_positive_centre_and_negative_flanks() -> None:
    corridor, flanks = _corridor_and_flanks()
    # Candidate band is too narrow: it under-predicts at the centre and
    # over-predicts nowhere, so the residual is positive inside, negative out.
    residual = np.zeros(SHAPE)
    residual[corridor] = 0.5
    residual[flanks] = -0.5

    result = classify_residual(residual, corridor=corridor, flanks=flanks)

    assert result["label"] == TOO_NARROW


def test_a_wide_candidate_shows_the_mirror_pattern() -> None:
    corridor, flanks = _corridor_and_flanks()
    residual = np.zeros(SHAPE)
    residual[corridor] = -0.5
    residual[flanks] = 0.5

    assert classify_residual(residual, corridor=corridor, flanks=flanks)["label"] == TOO_WIDE


def test_a_uniformly_low_candidate_reads_as_an_amplitude_defect() -> None:
    corridor, flanks = _corridor_and_flanks()
    residual = np.full(SHAPE, 0.4)

    result = classify_residual(residual, corridor=corridor, flanks=flanks)

    assert result["label"] == UNDER_AMPLITUDE


def test_a_uniformly_high_candidate_reads_as_the_opposite() -> None:
    corridor, flanks = _corridor_and_flanks()

    result = classify_residual(np.full(SHAPE, -0.4), corridor=corridor, flanks=flanks)

    assert result["label"] == OVER_AMPLITUDE


def test_a_negligible_residual_is_not_given_a_story() -> None:
    corridor, flanks = _corridor_and_flanks()
    generator = np.random.default_rng(5)

    result = classify_residual(
        generator.normal(0.0, 1e-6, SHAPE), corridor=corridor, flanks=flanks
    )

    assert result["label"] == UNSTRUCTURED


def test_the_classification_declares_itself_a_heuristic() -> None:
    corridor, flanks = _corridor_and_flanks()

    result = classify_residual(np.full(SHAPE, 0.4), corridor=corridor, flanks=flanks)

    assert "heuristic" in str(result["interpretation"])
    assert np.isfinite(float(result["centre"]))


def test_an_amplitude_error_correlates_with_the_reference() -> None:
    reference = _band()
    residual = 0.2 * reference  # exactly proportional

    result = residual_associations(residual, reference)

    assert result["with_reference"] == pytest.approx(1.0, abs=1e-9)


def test_a_placement_error_correlates_with_the_signed_derivative() -> None:
    reference = _band(centre=40.0)
    shifted = _band(centre=42.0)
    residual = reference - shifted

    result = residual_associations(residual, reference)

    # Shifting by delta gives a residual close to delta * df/dx.
    assert abs(result["with_reference_derivative_x"]) > 0.9
    assert abs(result["with_reference_derivative_x"]) > abs(result["with_reference"])


def test_the_gradient_magnitude_is_blind_to_a_displacement() -> None:
    reference = _band(centre=40.0)
    residual = reference - _band(centre=42.0)

    result = residual_associations(residual, reference)

    # A shift residual is antisymmetric across the band and the magnitude is
    # symmetric, so their correlation cancels. Reporting only the magnitude
    # would hide every placement error.
    assert abs(result["with_reference_gradient_magnitude"]) < 1e-6


def test_the_variogram_grows_with_lag_for_a_smooth_residual() -> None:
    lags, gamma = directional_variogram(_band(width=12.0), axis=0, maximum_lag_pixels=20)

    assert lags[0] == 1.0
    assert gamma[-1] > gamma[0]


def test_the_variogram_separates_the_two_directions() -> None:
    # Structured across rows, constant along columns.
    field = _band(width=6.0)

    _, across = directional_variogram(field, axis=0, maximum_lag_pixels=20)
    _, along = directional_variogram(field, axis=1, maximum_lag_pixels=20)

    assert float(np.nanmax(across)) > 0.0
    assert float(np.nanmax(along)) == pytest.approx(0.0, abs=1e-20)


def test_the_variogram_rejects_a_bad_axis() -> None:
    with pytest.raises(ValueError, match="axis must be"):
        directional_variogram(np.zeros(SHAPE), axis=2)


def test_a_coarse_residual_has_its_power_at_low_frequency() -> None:
    frequencies, spectrum = radial_power_spectrum(_band(width=15.0))
    finite = np.isfinite(spectrum)

    weighted = float(
        np.sum(frequencies[finite] * spectrum[finite]) / np.sum(spectrum[finite])
    )

    assert weighted < 0.1  # cycles per pixel


def test_a_fine_residual_has_more_high_frequency_power() -> None:
    generator = np.random.default_rng(2)
    coarse = _band(width=15.0)
    fine = generator.normal(0.0, 1.0, SHAPE)

    def _centroid(field: np.ndarray) -> float:
        frequencies, spectrum = radial_power_spectrum(field)
        finite = np.isfinite(spectrum)
        return float(
            np.sum(frequencies[finite] * spectrum[finite]) / np.sum(spectrum[finite])
        )

    assert _centroid(fine) > _centroid(coarse)
