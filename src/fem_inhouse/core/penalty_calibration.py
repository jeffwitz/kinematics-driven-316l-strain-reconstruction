"""Fix the boundary penalty stiffness by the discrepancy principle.

The misfit returned by a penalty solve is proportional to one over the spring
stiffness, so any amount of disagreement can be manufactured by choosing that
stiffness. This module removes the choice: the spring is set so that the solver
disagrees with the measurement by exactly as much as the measurement is
uncertain.

Registered in `validation/boundary_penalty_calibration_preregistration.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from fem_inhouse.core.linear_solver import create_linear_solver

FloatArray = NDArray[np.float64]

#: Robust per-state boundary noise measured on P43, in millimetres.
MEASURED_BOUNDARY_SIGMA_MM = 9.40e-5

#: Registered tolerance on hitting the target misfit, as a fraction.
CALIBRATION_TOLERANCE = 0.05

#: Registered iteration budget for the secant search on log k.
MAXIMUM_CALIBRATION_ITERATIONS = 40


@dataclass(frozen=True, slots=True)
class PenaltyCalibration:
    """A spring stiffness and the misfit it produces on the elastic operator."""

    stiffness: float
    achieved_rms_misfit_mm: float
    target_rms_misfit_mm: float
    iterations: int
    converged: bool
    reference_diagonal: float


def _diagonal_positions(matrix: csr_matrix, rows: NDArray[np.int64]) -> NDArray[np.int64]:
    indptr, indices = matrix.indptr, matrix.indices
    positions = np.empty(rows.size, dtype=np.int64)
    for order, row in enumerate(rows):
        start, stop = int(indptr[row]), int(indptr[row + 1])
        offset = int(np.searchsorted(indices[start:stop], row))
        if start + offset >= stop or indices[start + offset] != row:
            raise ValueError(f"row {int(row)} has no stored diagonal entry")
        positions[order] = start + offset
    return positions


def elastic_misfit_for_stiffness(
    stiffness_matrix: csr_matrix,
    *,
    boundary_dofs: NDArray[np.int64],
    boundary_values: NDArray[np.float64],
    stiffness: float,
) -> float:
    """Return the RMS boundary misfit of one elastic penalty solve."""

    if stiffness <= 0.0 or not np.isfinite(stiffness):
        raise ValueError("stiffness must be finite and positive")
    matrix = stiffness_matrix.copy()
    positions = _diagonal_positions(matrix, boundary_dofs)
    matrix.data[positions] += stiffness
    right_hand_side = np.zeros(matrix.shape[0], dtype=np.float64)
    right_hand_side[boundary_dofs] = stiffness * boundary_values
    solver = create_linear_solver("nonsymmetric")
    try:
        solution = solver.factorize_and_solve(matrix, right_hand_side)
    finally:
        solver.close()
    misfit = boundary_values - np.asarray(solution, dtype=np.float64)[boundary_dofs]
    return float(np.sqrt(np.mean(np.square(misfit))))


def calibrate_boundary_penalty_stiffness(
    stiffness_matrix: csr_matrix,
    *,
    boundary_dofs: NDArray[np.int64],
    boundary_values: NDArray[np.float64],
    target_rms_misfit_mm: float = MEASURED_BOUNDARY_SIGMA_MM,
    tolerance: float = CALIBRATION_TOLERANCE,
    maximum_iterations: int = MAXIMUM_CALIBRATION_ITERATIONS,
) -> PenaltyCalibration:
    """Find the spring stiffness whose elastic misfit matches the noise floor.

    The misfit decreases monotonically with the stiffness, so the search is a
    bisection on ``log k`` after bracketing. The elastic operator is used
    deliberately: plasticity will move the achieved misfit, and the full run
    reports what it actually reaches rather than assuming this value holds.
    """

    if target_rms_misfit_mm <= 0.0 or not np.isfinite(target_rms_misfit_mm):
        raise ValueError("target_rms_misfit_mm must be finite and positive")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must lie strictly between zero and one")
    values = np.asarray(boundary_values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("boundary_values must be finite and non-empty")
    dofs = np.asarray(boundary_dofs, dtype=np.int64)
    if dofs.shape != values.shape:
        raise ValueError("boundary_dofs and boundary_values must have one shape")

    diagonal = np.asarray(stiffness_matrix.diagonal(), dtype=np.float64)[dofs]
    reference = float(np.median(np.abs(diagonal)))
    if reference <= 0.0:
        raise ValueError("the boundary tangent diagonal is not positive")

    def misfit(stiffness: float) -> float:
        return elastic_misfit_for_stiffness(
            stiffness_matrix,
            boundary_dofs=dofs,
            boundary_values=values,
            stiffness=stiffness,
        )

    # Bracket: a soft spring overshoots the target, a stiff one undershoots.
    low, high = reference * 1.0e-4, reference * 1.0e4
    iterations = 0
    if misfit(low) < target_rms_misfit_mm:
        return PenaltyCalibration(
            stiffness=low,
            achieved_rms_misfit_mm=misfit(low),
            target_rms_misfit_mm=target_rms_misfit_mm,
            iterations=0,
            converged=False,
            reference_diagonal=reference,
        )
    if misfit(high) > target_rms_misfit_mm:
        return PenaltyCalibration(
            stiffness=high,
            achieved_rms_misfit_mm=misfit(high),
            target_rms_misfit_mm=target_rms_misfit_mm,
            iterations=0,
            converged=False,
            reference_diagonal=reference,
        )

    achieved = float("nan")
    stiffness = high
    iterations = 0
    for step in range(1, maximum_iterations + 1):
        iterations = step
        stiffness = float(np.sqrt(low * high))
        achieved = misfit(stiffness)
        if abs(achieved - target_rms_misfit_mm) <= tolerance * target_rms_misfit_mm:
            break
        if achieved > target_rms_misfit_mm:
            low = stiffness
        else:
            high = stiffness

    return PenaltyCalibration(
        stiffness=stiffness,
        achieved_rms_misfit_mm=achieved,
        target_rms_misfit_mm=target_rms_misfit_mm,
        iterations=iterations,
        converged=bool(
            abs(achieved - target_rms_misfit_mm) <= tolerance * target_rms_misfit_mm
        ),
        reference_diagonal=reference,
    )
