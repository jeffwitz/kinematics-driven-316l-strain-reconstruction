"""Sequence-level tests of the causal TANN-FCC trajectory.

The two structural guarantees of the masked-state holdout: the interior
DIC of a holdout state never influences the forward prediction (only its
recorded loss), and the constitutive state is never reinitialised or
recalibrated between states.
"""

from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.constitutive.tann_fcc import TannFCCBatch, TannFCCConfig
from fem_inhouse.core.fcc_interaction_matrix import SLIP_SYSTEMS
from fem_inhouse.identification.tann_fcc_sequence import TannFCCSequence
from fem_inhouse.spectral2d import EBISpectralSolverConfig, StructuredGrid2D
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig

STATES = [21, 22]
HOLDOUT = {22}


def _identity_systems(points: int) -> np.ndarray:
    tensors = np.empty((12, 3, 3))
    for index, (burgers, normal) in enumerate(SLIP_SYSTEMS):
        s = np.asarray(burgers, dtype=np.float64)
        m = np.asarray(normal, dtype=np.float64)
        s /= np.linalg.norm(s)
        m /= np.linalg.norm(m)
        tensors[index] = 0.5 * (np.outer(s, m) + np.outer(m, s))
    return np.tile(tensors, (points, 1, 1, 1))


def _solver_config() -> EBISpectralSolverConfig:
    return EBISpectralSolverConfig(
        relative_equilibrium_tolerance=1.0e-10,
        transform=SpectralTransformConfig(backend="scipy"),
    )


def _histories(grid: StructuredGrid2D) -> tuple[np.ndarray, np.ndarray]:
    x, y = grid.coordinates
    boundary = np.zeros((3, *grid.node_shape, 2))
    for state in (1, 2):
        boundary[state, ..., 0] = state * 0.01 * x[:, None]
        boundary[state, ..., 1] = state * 0.005 * y[None, :]
    # measured interior aligned to the increments (no zero-reference entry):
    # an independent smooth field, not derivable from the boundary (it
    # exists to be modified and to be scored)
    measured = boundary[1:].copy()
    measured[:, 1:-1, 1:-1, 0] += 0.003 * x[1:-1, None] * y[None, 1:-1]
    measured[:, 1:-1, 1:-1, 1] -= 0.002 * x[1:-1, None]
    return boundary, measured


def _sequence(grid: StructuredGrid2D, measured: np.ndarray) -> TannFCCSequence:
    material = TannFCCBatch(
        TannFCCConfig(),
        point_count=2 * grid.nx * grid.ny,
        systems_global=_identity_systems(2 * grid.nx * grid.ny),
    )
    boundary, _ = _histories(grid)
    return TannFCCSequence(
        grid=grid,
        material=material,
        boundary_history=boundary,
        measured_interior=measured,
        state_indices=STATES,
        holdout=HOLDOUT,
        solver_config=_solver_config(),
    )


def test_tann_sequence_masked_state_no_dic_leak() -> None:
    pytest.importorskip("pyfftw")
    grid = StructuredGrid2D(4, 4, 2.0, 2.0)
    boundary, measured = _histories(grid)
    result = _sequence(grid, measured).rollout()

    # Arbitrarily corrupt the interior DIC of the holdout state, boundary
    # untouched: the forward predictions must be strictly identical.
    corrupted = measured.copy()
    holdout_index = STATES.index(next(iter(HOLDOUT)))
    corrupted[holdout_index, 1:-1, 1:-1, :] += 17.3 * measured[holdout_index, 1:-1, 1:-1, :]
    corrupted_result = _sequence(grid, corrupted).rollout()

    for original, modified in zip(result.records, corrupted_result.records, strict=True):
        np.testing.assert_array_equal(original.displacement, modified.displacement)
        np.testing.assert_array_equal(original.stress_in_plane_mpa, modified.stress_in_plane_mpa)
        np.testing.assert_array_equal(original.committed_state, modified.committed_state)
        if original.holdout:
            # only the recorded loss of the corrupted state may change
            assert original.loss_raw != modified.loss_raw
        else:
            assert original.loss_raw == modified.loss_raw


def test_tann_sequence_state_not_reset_at_holdout() -> None:
    pytest.importorskip("pyfftw")
    grid = StructuredGrid2D(4, 4, 2.0, 2.0)
    boundary, measured = _histories(grid)
    sequence = _sequence(grid, measured)
    result = sequence.rollout()
    # Played once, no reinitialisation: the material ends at q_N, and the
    # record of the last increment carries exactly that state.
    assert np.array_equal(
        sequence.material.committed_state, result.records[-1].committed_state
    )
    # The trajectory advanced materially from the zero initial state, and
    # the committed strain of the last increment is the converged one.
    assert np.max(np.abs(result.records[-1].committed_state)) > 0.0
    assert np.max(np.abs(result.records[-1].strain_in_plane_mpa)) > 0.0
    # Every record differs from the previous one's state: no increment was
    # silently skipped or reset (holds across the holdout boundary too).
    for previous, current in zip(result.records, result.records[1:]):
        assert not np.array_equal(previous.committed_state, current.committed_state)
