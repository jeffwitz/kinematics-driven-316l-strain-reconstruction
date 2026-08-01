from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.validation.gradient_fluctuation import (
    SCALES_PIXELS,
    antisymmetric_part,
    displacement_gradient,
    frobenius_norm,
    gradient_criteria,
    highpass,
    map_comparison,
    multiscale_fluctuation,
    relative_frobenius_distance,
    remove_mean,
    symmetric_part,
)

SHAPE = (81, 97)
SPACING = 0.00184


def _coordinates() -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(SHAPE[0], dtype=np.float64)[:, None] * SPACING
    y = np.arange(SHAPE[1], dtype=np.float64)[None, :] * SPACING
    return np.broadcast_to(x, SHAPE), np.broadcast_to(y, SHAPE)


def _affine(a: np.ndarray) -> np.ndarray:
    """Displacement of the affine map ``u = A x``."""

    x, y = _coordinates()
    return np.ascontiguousarray(
        np.stack((a[0, 0] * x + a[0, 1] * y, a[1, 0] * x + a[1, 1] * y), axis=-1)
    )


def _band(amplitude=1e-3, centre=40.0, width=6.0) -> np.ndarray:
    rows = np.arange(SHAPE[0], dtype=np.float64)[:, None]
    profile = amplitude * np.exp(-0.5 * ((rows - centre) / width) ** 2)
    return np.ascontiguousarray(
        np.stack((np.broadcast_to(profile, SHAPE), np.zeros(SHAPE)), axis=-1)
    )


def test_an_affine_displacement_gives_its_exact_gradient() -> None:
    a = np.array([[2.0e-3, -5.0e-4], [7.0e-4, 1.0e-3]])

    gradient = displacement_gradient(_affine(a), spacing_x_mm=SPACING, spacing_y_mm=SPACING)

    assert gradient.shape == (SHAPE[0] - 1, SHAPE[1] - 1, 2, 2)
    assert np.allclose(gradient, a, atol=1e-12)


def test_a_rigid_rotation_moves_the_gradient_but_not_the_strain() -> None:
    # The registered separation of section 6: rotation must contribute to
    # J_gradient and leave J_strain at zero.
    angle = 1.0e-3
    rotation = np.array([[0.0, -angle], [angle, 0.0]])
    reference = displacement_gradient(
        _affine(np.zeros((2, 2))), spacing_x_mm=SPACING, spacing_y_mm=SPACING
    )
    rotated = displacement_gradient(_affine(rotation), spacing_x_mm=SPACING, spacing_y_mm=SPACING)

    assert np.allclose(symmetric_part(rotated), 0.0, atol=1e-15)
    assert np.abs(antisymmetric_part(rotated)).max() > 0.9 * angle
    # Against a zero reference the relative distances are undefined, so the
    # rotation is checked on the tensors themselves.
    assert relative_frobenius_distance(rotated, reference + 1.0) > 0.0


def test_rotation_is_invisible_to_the_strain_criterion_on_a_real_pattern() -> None:
    angle = 5.0e-4
    rotation = np.array([[0.0, -angle], [angle, 0.0]])
    base = _band()
    reference = displacement_gradient(base, spacing_x_mm=SPACING, spacing_y_mm=SPACING)
    with_rotation = displacement_gradient(
        base + _affine(rotation), spacing_x_mm=SPACING, spacing_y_mm=SPACING
    )

    criteria = gradient_criteria(with_rotation, reference)

    assert criteria["J_gradient"] > 1.0e-3
    assert criteria["J_strain"] == pytest.approx(0.0, abs=1e-12)


def test_the_symmetric_and_antisymmetric_parts_recompose_the_gradient() -> None:
    gradient = displacement_gradient(_band(), spacing_x_mm=SPACING, spacing_y_mm=SPACING)

    assert np.allclose(
        symmetric_part(gradient) + antisymmetric_part(gradient), gradient, atol=1e-18
    )


def test_the_frobenius_norm_matches_an_explicit_sum_of_squares() -> None:
    gradient = displacement_gradient(_band(), spacing_x_mm=SPACING, spacing_y_mm=SPACING)

    expected = np.sqrt(
        gradient[..., 0, 0] ** 2
        + gradient[..., 0, 1] ** 2
        + gradient[..., 1, 0] ** 2
        + gradient[..., 1, 1] ** 2
    )

    assert np.allclose(frobenius_norm(gradient), expected, atol=1e-18)


def test_a_uniform_translation_changes_no_criterion() -> None:
    # Registered invariance: adding a constant displacement is not a defect.
    base = _band()
    shifted = base + np.array([3.7e-3, -1.2e-3])
    reference = displacement_gradient(base, spacing_x_mm=SPACING, spacing_y_mm=SPACING)
    candidate = displacement_gradient(shifted, spacing_x_mm=SPACING, spacing_y_mm=SPACING)

    criteria = gradient_criteria(candidate, reference)

    for name in ("J_gradient", "J_strain", "J_norm_map", "J_fluctuation"):
        assert criteria[name] == pytest.approx(0.0, abs=1e-12)


def test_removing_the_mean_kills_a_uniform_affine_offset() -> None:
    a = np.array([[1.0e-3, 0.0], [0.0, -3.0e-4]])
    base = displacement_gradient(_band(), spacing_x_mm=SPACING, spacing_y_mm=SPACING)
    offset = displacement_gradient(_band() + _affine(a), spacing_x_mm=SPACING, spacing_y_mm=SPACING)

    # The mean differs, the fluctuation does not.
    assert relative_frobenius_distance(offset, base) > 1.0e-3
    assert relative_frobenius_distance(remove_mean(offset), remove_mean(base)) == pytest.approx(
        0.0, abs=1e-12
    )


def test_the_highpass_removes_a_constant_and_keeps_fine_structure() -> None:
    gradient = displacement_gradient(_band(width=3.0), spacing_x_mm=SPACING, spacing_y_mm=SPACING)
    constant = np.full_like(gradient, 2.5e-4)

    assert np.allclose(highpass(constant, scale_pixels=16), 0.0, atol=1e-15)
    fine = np.abs(highpass(gradient, scale_pixels=8)).max()
    coarse = np.abs(highpass(gradient, scale_pixels=96)).max()
    # A wider filter removes more, so it leaves at least as much residual.
    assert coarse >= fine


def test_multiscale_returns_every_registered_scale() -> None:
    reference = displacement_gradient(_band(), spacing_x_mm=SPACING, spacing_y_mm=SPACING)
    candidate = displacement_gradient(
        _band(centre=44.0), spacing_x_mm=SPACING, spacing_y_mm=SPACING
    )

    curve = multiscale_fluctuation(candidate, reference)

    assert tuple(curve) == SCALES_PIXELS
    assert all(np.isfinite(value) for value in curve.values())


def test_a_translated_band_is_penalised_more_than_an_untouched_one() -> None:
    reference = displacement_gradient(_band(), spacing_x_mm=SPACING, spacing_y_mm=SPACING)
    shifted = displacement_gradient(_band(centre=46.0), spacing_x_mm=SPACING, spacing_y_mm=SPACING)

    assert gradient_criteria(reference, reference)["J_strain"] == pytest.approx(0.0)
    assert gradient_criteria(shifted, reference)["J_strain"] > 0.5


def test_the_axis_convention_is_the_one_of_the_strain_operator() -> None:
    # Array axis 0 is canonical x: a displacement varying along axis 0 must
    # land in du_x/dx, not in du_x/dy.
    x, _ = _coordinates()
    field = np.ascontiguousarray(np.stack((x, np.zeros(SHAPE)), axis=-1))

    gradient = displacement_gradient(field, spacing_x_mm=SPACING, spacing_y_mm=SPACING)

    assert gradient[..., 0, 0] == pytest.approx(1.0)
    assert np.allclose(gradient[..., 0, 1], 0.0, atol=1e-12)


def test_the_computation_is_deterministic() -> None:
    field = _band()
    first = displacement_gradient(field, spacing_x_mm=SPACING, spacing_y_mm=SPACING)
    second = displacement_gradient(field, spacing_x_mm=SPACING, spacing_y_mm=SPACING)

    assert np.array_equal(first, second)
    reference = displacement_gradient(
        _band(centre=44.0), spacing_x_mm=SPACING, spacing_y_mm=SPACING
    )
    assert gradient_criteria(first, reference) == gradient_criteria(second, reference)


def test_non_finite_displacement_is_rejected_rather_than_silently_propagated() -> None:
    field = _band()
    field[10, 10, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        displacement_gradient(field, spacing_x_mm=SPACING, spacing_y_mm=SPACING)


def test_non_finite_tensor_points_are_excluded_from_the_norms() -> None:
    reference = displacement_gradient(_band(), spacing_x_mm=SPACING, spacing_y_mm=SPACING)
    candidate = reference.copy()
    clean = relative_frobenius_distance(candidate, reference)
    candidate[5, 5] = np.nan

    assert clean == pytest.approx(0.0)
    # The bad point drops out instead of poisoning the whole scalar.
    assert relative_frobenius_distance(candidate, reference) == pytest.approx(0.0)


def test_an_empty_mask_is_an_error_rather_than_a_nan() -> None:
    gradient = displacement_gradient(_band(), spacing_x_mm=SPACING, spacing_y_mm=SPACING)
    mask = np.zeros(gradient.shape[:2], dtype=bool)

    with pytest.raises(ValueError, match="no valid point"):
        relative_frobenius_distance(gradient, gradient, valid_mask=mask)


def test_map_comparison_reports_both_correlations_and_the_quantile_ratios() -> None:
    reference = frobenius_norm(
        displacement_gradient(_band(), spacing_x_mm=SPACING, spacing_y_mm=SPACING)
    )

    identical = map_comparison(reference, reference)

    assert identical["pearson"] == pytest.approx(1.0)
    assert identical["spearman"] == pytest.approx(1.0)
    assert identical["mean_bias"] == pytest.approx(0.0)
    assert identical["quantile_ratio_q90"] == pytest.approx(1.0)
    assert identical["quantile_ratio_q95"] == pytest.approx(1.0)


def test_spearman_sees_a_monotone_change_that_pearson_underrates() -> None:
    reference = np.abs(
        frobenius_norm(displacement_gradient(_band(), spacing_x_mm=SPACING, spacing_y_mm=SPACING))
    )
    squashed = np.sqrt(reference)

    outcome = map_comparison(squashed, reference)

    # Not exactly 1: sqrt merges neighbouring floats, creating ties the
    # reference does not have. That is rounding, not a rank inversion.
    assert outcome["spearman"] == pytest.approx(1.0, abs=1e-3)
    assert outcome["pearson"] < 0.99
    assert outcome["pearson"] < outcome["spearman"]
