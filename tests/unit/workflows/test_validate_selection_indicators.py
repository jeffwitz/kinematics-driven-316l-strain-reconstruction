from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.postprocessing.kinematics import plane_stress_equivalent_strain
from fem_inhouse.validation.gradient_fluctuation import symmetric_part
from fem_inhouse.workflows.compare_gradient_fluctuation_criteria import (
    PIXEL_SIZE_MM,
    gradient_on_core,
)
from fem_inhouse.workflows.validate_selection_indicators import (
    REPETITION_COLUMN_SIGMA_PIXELS,
    REPETITION_CORRELATION_PIXELS,
    REPETITION_EVM_RMS,
    REPETITION_ROW_SIGMA_PIXELS,
    correlated_repetition_residual,
)

SOLVE_SHAPE = (661, 611)


def _evm_rms(displacement: np.ndarray) -> float:
    strain = symmetric_part(gradient_on_core(displacement))
    evm = plane_stress_equivalent_strain(
        strain[..., 0, 0],
        strain[..., 1, 1],
        2.0 * strain[..., 0, 1],
        poisson_ratio=0.3,
        shear_convention="engineering",
    )
    return float(np.sqrt(np.mean(np.asarray(evm) ** 2)))


def test_the_floor_is_calibrated_on_the_measured_strain() -> None:
    """The registered anchor is the spurious EVM RMS, not the displacement."""

    residual = correlated_repetition_residual(
        SOLVE_SHAPE, generator=np.random.default_rng(20260801)
    )

    assert _evm_rms(residual) == pytest.approx(REPETITION_EVM_RMS, rel=1e-9)


def test_anchoring_on_displacement_would_inflate_the_floor_twelvefold() -> None:
    """Why the anchor is the strain, locked so the reasoning cannot be lost.

    A Gaussian field reproducing the measured displacement deviations and the
    measured coherence produces far more strain than the same campaign
    measured. The real residual is smoother than a Gaussian field of that
    nominal coherence, consistent with slow optical drift.
    """

    generator = np.random.default_rng(20260801)
    smoothing = 0.5 * REPETITION_CORRELATION_PIXELS
    from scipy import ndimage

    components = []
    for sigma_pixels in (REPETITION_ROW_SIGMA_PIXELS, REPETITION_COLUMN_SIGMA_PIXELS):
        white = generator.normal(0.0, 1.0, SOLVE_SHAPE)
        field = ndimage.gaussian_filter(white, sigma=smoothing, mode="nearest")
        components.append(field * (sigma_pixels / float(np.std(field))) * PIXEL_SIZE_MM)
    displacement_anchored = np.ascontiguousarray(np.stack(components, axis=-1))

    inflation = _evm_rms(displacement_anchored) / REPETITION_EVM_RMS

    assert inflation > 8.0


def test_the_residual_keeps_the_measured_component_ratio() -> None:
    residual = correlated_repetition_residual(SOLVE_SHAPE, generator=np.random.default_rng(7))

    ratio = float(np.std(residual[..., 1]) / np.std(residual[..., 0]))
    expected = REPETITION_COLUMN_SIGMA_PIXELS / REPETITION_ROW_SIGMA_PIXELS

    assert ratio == pytest.approx(expected, rel=1e-9)


def test_the_residual_is_coherent_rather_than_white() -> None:
    """A white perturbation would be filtered straight out by the high pass."""

    residual = correlated_repetition_residual(SOLVE_SHAPE, generator=np.random.default_rng(11))
    row = residual[:, 300, 0]
    centred = row - row.mean()
    lag = 10
    correlation = float(
        np.sum(centred[:-lag] * centred[lag:])
        / np.sqrt(np.sum(centred[:-lag] ** 2) * np.sum(centred[lag:] ** 2))
    )

    # At a tenth of the coherence length the field must still be correlated.
    assert correlation > 0.5


def test_different_realisations_differ_but_share_the_calibration() -> None:
    generator = np.random.default_rng(3)
    first = correlated_repetition_residual(SOLVE_SHAPE, generator=generator)
    second = correlated_repetition_residual(SOLVE_SHAPE, generator=generator)

    assert not np.allclose(first, second)
    assert _evm_rms(first) == pytest.approx(_evm_rms(second), rel=1e-9)
