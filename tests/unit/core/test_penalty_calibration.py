from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fem_inhouse.core.penalty_calibration import (
    MEASURED_BOUNDARY_SIGMA_MM,
    calibrate_boundary_penalty_stiffness,
    elastic_misfit_for_stiffness,
)


def _chain(count: int = 200, stiffness: float = 2.0e5) -> csr_matrix:
    """A one-dimensional spring chain with the two ends prescribed."""

    diagonal = np.full(count, 2.0 * stiffness)
    diagonal[0] = diagonal[-1] = stiffness
    off = np.full(count - 1, -stiffness)
    return csr_matrix(np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1))


BOUNDARY = np.array([0, 199])
VALUES = np.array([0.0, 0.05])


def test_misfit_decreases_as_the_spring_stiffens() -> None:
    matrix = _chain()
    misfits = [
        elastic_misfit_for_stiffness(
            matrix, boundary_dofs=BOUNDARY, boundary_values=VALUES, stiffness=k
        )
        for k in (1.0e4, 1.0e6, 1.0e8)
    ]

    assert misfits[0] > misfits[1] > misfits[2]


def test_misfit_rejects_a_nonpositive_stiffness() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        elastic_misfit_for_stiffness(
            _chain(), boundary_dofs=BOUNDARY, boundary_values=VALUES, stiffness=0.0
        )


def test_calibration_hits_the_measured_noise_floor() -> None:
    result = calibrate_boundary_penalty_stiffness(
        _chain(), boundary_dofs=BOUNDARY, boundary_values=VALUES
    )

    assert result.converged
    assert result.target_rms_misfit_mm == MEASURED_BOUNDARY_SIGMA_MM
    assert result.achieved_rms_misfit_mm == pytest.approx(
        MEASURED_BOUNDARY_SIGMA_MM, rel=0.05
    )
    # A finite, well-conditioned spring, not a hard-Dirichlet impersonation.
    assert 0.1 < result.stiffness / result.reference_diagonal < 100.0


def test_calibration_honours_a_tighter_target() -> None:
    loose = calibrate_boundary_penalty_stiffness(
        _chain(), boundary_dofs=BOUNDARY, boundary_values=VALUES
    )
    tight = calibrate_boundary_penalty_stiffness(
        _chain(),
        boundary_dofs=BOUNDARY,
        boundary_values=VALUES,
        target_rms_misfit_mm=MEASURED_BOUNDARY_SIGMA_MM / 10.0,
    )

    # Ten times less tolerated disagreement needs a stiffer spring.
    assert tight.stiffness > loose.stiffness
    assert tight.achieved_rms_misfit_mm < loose.achieved_rms_misfit_mm


def test_calibration_rejects_an_impossible_target() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        calibrate_boundary_penalty_stiffness(
            _chain(),
            boundary_dofs=BOUNDARY,
            boundary_values=VALUES,
            target_rms_misfit_mm=0.0,
        )


def test_calibration_rejects_mismatched_boundary_inputs() -> None:
    with pytest.raises(ValueError, match="one shape"):
        calibrate_boundary_penalty_stiffness(
            _chain(),
            boundary_dofs=np.array([0, 199]),
            boundary_values=np.array([0.0]),
        )
