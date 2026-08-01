from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.validation.fractions_skill_score import fractions_skill_score
from fem_inhouse.validation.gradient_fluctuation import displacement_gradient
from fem_inhouse.validation.selection_indicators import (
    DEFECT_NAMES,
    PRINCIPAL_SCALE_PIXELS,
    amplitude_defect,
    energy_ratio,
    evaluate,
    fluctuation_magnitude,
    localisation_defect,
    minimax,
    normalise,
    presence_defect,
    shape_defect,
)

SHAPE = (160, 180)
SPACING = 0.00184
RNG = np.random.default_rng(20260801)


def _textured(amplitude: float = 1.0, shift: int = 0, seed: int = 1) -> np.ndarray:
    """A displacement whose strain has structure at the principal scale."""

    generator = np.random.default_rng(seed)
    field = generator.normal(0.0, 1.0, SHAPE)
    from scipy import ndimage

    field = ndimage.gaussian_filter(field, sigma=8.0)
    field = amplitude * 1.0e-3 * field / float(np.std(field))
    field = np.roll(field, shift, axis=0)
    return np.ascontiguousarray(np.stack((field, np.zeros(SHAPE)), axis=-1))


def _gradient(displacement: np.ndarray) -> np.ndarray:
    return displacement_gradient(displacement, spacing_x_mm=SPACING, spacing_y_mm=SPACING)


def test_the_reference_against_itself_is_optimal_on_every_defect() -> None:
    gradient = _gradient(_textured())

    defects = evaluate(gradient, gradient, label="self")

    assert defects.shape == pytest.approx(0.0, abs=1e-12)
    assert defects.amplitude == pytest.approx(0.0, abs=1e-12)
    assert defects.localisation == pytest.approx(0.0, abs=1e-12)
    assert defects.presence == pytest.approx(0.0, abs=1e-12)


def test_a_smooth_field_fails_on_presence_where_a_distance_would_not() -> None:
    """The defect the earlier Frobenius criteria could not catch."""

    reference = _gradient(_textured())
    # A field carrying almost no fluctuation at the principal scale.
    smooth = _gradient(_textured(amplitude=0.05))

    defects = evaluate(smooth, reference, label="smooth")

    # Suppressing the fluctuations by a factor 20 costs about 2*log(20) in
    # energy, so presence is large and unmistakable.
    assert defects.presence > 3.0
    assert defects.amplitude > 1.0


def test_presence_penalises_too_much_energy_as_well_as_too_little() -> None:
    reference = fluctuation_magnitude(_gradient(_textured()), scale_pixels=PRINCIPAL_SCALE_PIXELS)
    weak = fluctuation_magnitude(
        _gradient(_textured(amplitude=0.5)), scale_pixels=PRINCIPAL_SCALE_PIXELS
    )
    strong = fluctuation_magnitude(
        _gradient(_textured(amplitude=2.0)), scale_pixels=PRINCIPAL_SCALE_PIXELS
    )

    # Symmetric in log, which is the whole point of using a logarithm.
    assert presence_defect(weak, reference) == pytest.approx(
        presence_defect(strong, reference), rel=1e-9
    )
    assert energy_ratio(weak, reference) < 1.0 < energy_ratio(strong, reference)


def test_shape_ignores_amplitude_and_amplitude_ignores_position() -> None:
    reference = fluctuation_magnitude(_gradient(_textured()), scale_pixels=PRINCIPAL_SCALE_PIXELS)
    scaled = fluctuation_magnitude(
        _gradient(_textured(amplitude=1.6)), scale_pixels=PRINCIPAL_SCALE_PIXELS
    )
    displaced = fluctuation_magnitude(
        _gradient(_textured(shift=24)), scale_pixels=PRINCIPAL_SCALE_PIXELS
    )

    # A pure rescaling leaves the correlation exactly untouched.
    assert shape_defect(scaled, reference) == pytest.approx(0.0, abs=1e-9)
    # A pure displacement barely moves the upper quantile. Not exactly zero:
    # the roll wraps, and the seam adds a little strain of its own.
    assert amplitude_defect(displaced, reference) < 0.1
    # And each defect is caught by the other indicator, an order of magnitude
    # above the leak.
    assert amplitude_defect(scaled, reference) > 0.4
    assert shape_defect(displaced, reference) > 0.5


def test_a_displaced_field_is_caught_by_localisation() -> None:
    reference = _gradient(_textured())
    displaced = _gradient(_textured(shift=24))

    defects = evaluate(displaced, reference, label="displaced")
    identical = evaluate(reference, reference, label="self")

    # FSS at a 49 px neighbourhood is forgiving of a 24 px shift, so the
    # response is real but modest. What matters is that it is not zero.
    assert defects.localisation > 0.1
    assert defects.localisation > 100.0 * max(identical.localisation, 1e-12)


def test_normalisation_puts_the_floor_at_zero_and_the_control_at_one() -> None:
    floor = dict.fromkeys(DEFECT_NAMES, 0.1)
    null = dict.fromkeys(DEFECT_NAMES, 1.1)

    assert normalise(floor, self_defects=floor, null_defects=null) == pytest.approx(
        dict.fromkeys(DEFECT_NAMES, 0.0)
    )
    assert normalise(null, self_defects=floor, null_defects=null) == pytest.approx(
        dict.fromkeys(DEFECT_NAMES, 1.0)
    )


def test_a_degenerate_span_gives_nan_rather_than_a_huge_number() -> None:
    floor = dict.fromkeys(DEFECT_NAMES, 0.5)

    normalised = normalise(dict.fromkeys(DEFECT_NAMES, 0.6), self_defects=floor, null_defects=floor)

    assert all(np.isnan(value) for value in normalised.values())


def test_minimax_reports_the_worst_defect_and_refuses_to_average() -> None:
    normalised = {"D_shape": 0.1, "D_amplitude": 0.2, "D_localisation": 0.9, "D_presence": 0.3}

    # Not the mean, 0.375, and not a weighted sum: the worst one.
    assert minimax(normalised) == pytest.approx(0.9)


def test_minimax_is_undefined_when_any_defect_is() -> None:
    normalised = {
        "D_shape": 0.1,
        "D_amplitude": float("nan"),
        "D_localisation": 0.2,
        "D_presence": 0.3,
    }

    assert np.isnan(minimax(normalised))


def test_an_excellent_amplitude_cannot_compensate_a_bad_localisation() -> None:
    # The registered reason for choosing minimax over a sum.
    balanced = {"D_shape": 0.4, "D_amplitude": 0.4, "D_localisation": 0.4, "D_presence": 0.4}
    lopsided = {"D_shape": 0.0, "D_amplitude": 0.0, "D_localisation": 0.9, "D_presence": 0.0}

    assert sum(lopsided.values()) < sum(balanced.values())
    assert minimax(lopsided) > minimax(balanced)


def test_the_localisation_threshold_comes_from_the_reference_only() -> None:
    """No candidate may move the boundary it is judged against.

    Checked against the threshold computed by hand from the reference, rather
    than through an invariance the indicator does not have: an absolute
    threshold applied to a rescaled candidate does change its active set, and
    that is correct.
    """

    reference_gradient = _gradient(_textured())
    candidate_gradient = _gradient(_textured(amplitude=3.0))
    reference = fluctuation_magnitude(reference_gradient, scale_pixels=PRINCIPAL_SCALE_PIXELS)
    candidate = fluctuation_magnitude(candidate_gradient, scale_pixels=PRINCIPAL_SCALE_PIXELS)
    threshold = float(np.quantile(reference, 0.90))

    expected = 1.0 - fractions_skill_score(
        reference >= threshold,
        candidate >= threshold,
        scale_pixels=PRINCIPAL_SCALE_PIXELS,
    )

    assert localisation_defect(candidate, reference) == pytest.approx(expected)
    # Recomputing the threshold on the candidate would give a different answer,
    # which is exactly what freezing it prevents.
    candidate_threshold = float(np.quantile(candidate, 0.90))
    assert candidate_threshold > 2.0 * threshold


def test_every_registered_defect_is_reported() -> None:
    gradient = _gradient(_textured())

    assert set(evaluate(gradient, gradient, label="x").as_dict()) == set(DEFECT_NAMES)
