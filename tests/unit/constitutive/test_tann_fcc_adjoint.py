"""The small gate that precedes any long training: the trajectory adjoint
against central finite differences, and the mechanical dot-product test.

Frozen bars (preregistration, section "Differentiation"): gradient vs
central FD relative error <= 1e-4 (target 1e-5); the transpose tangent
action matches the primal action to ~1e-8 in float64.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fem_inhouse.constitutive.tann_fcc import TannFCCBatch, TannFCCConfig
from fem_inhouse.core.fcc_interaction_matrix import SLIP_SYSTEMS
from fem_inhouse.identification.tann_fcc_adjoint import TannFCCTrajectoryAdjoint
from fem_inhouse.identification.tann_fcc_sequence import TannFCCSequence
from fem_inhouse.spectral2d import EBISpectralSolverConfig, StructuredGrid2D
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig

STATES = [21, 22]
HOLDOUT = {22}
SEED_LOCAL = 20260817


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
    # Deliberately NON-affine: an affine boundary gives a uniform strain,
    # a uniform plastic response and a trivially satisfied equilibrium,
    # which would make the displacement -- and the loss -- independent of
    # the constitutive law and the FD gate vacuous.
    x, y = grid.coordinates
    span_x = x[-1] - x[0] if x.size else 1.0
    boundary = np.zeros((3, *grid.node_shape, 2))
    for state in (1, 2):
        boundary[state, ..., 0] = state * (
            0.005 * x[:, None] + 0.004 * x[:, None] ** 2 / span_x
        )
        boundary[state, ..., 1] = state * (
            0.002 * y[None, :] + 0.003 * x[:, None] * y[None, :] / span_x
        )
    measured = boundary[1:].copy()  # aligned to the increments
    measured[:, 1:-1, 1:-1, 0] += 0.001 * x[1:-1, None] * y[None, 1:-1] / span_x
    return boundary, measured


def _rollout_loss(
    grid: StructuredGrid2D, boundary: np.ndarray, measured: np.ndarray
) -> tuple[float, TannFCCBatch, TannFCCSequence, tuple]:
    material = TannFCCBatch(
        TannFCCConfig(),
        point_count=2 * grid.nx * grid.ny,
        systems_global=_identity_systems(2 * grid.nx * grid.ny),
    )
    sequence = TannFCCSequence(
        grid=grid,
        material=material,
        boundary_history=boundary,
        measured_interior=measured,
        state_indices=STATES,
        holdout=HOLDOUT,
        solver_config=_solver_config(),
    )
    result = sequence.rollout()
    return result.total_loss_raw, material, sequence, result.records


def test_tann_sequence_adjoint_fd() -> None:
    pytest.importorskip("pyfftw")
    grid = StructuredGrid2D(8, 8, 1.0, 1.0)
    boundary, measured = _histories(grid)
    point_count = 2 * grid.nx * grid.ny

    # At the tiny registered initialisation the law is near-elastic and the
    # loss is flat in theta (FD and adjoint both exactly zero -- consistent,
    # but vacuous). The gate therefore runs at random weight radii, as the
    # preregistration prescribes ("plusieurs parametres/rayons aleatoires").
    base_material = TannFCCBatch(
        TannFCCConfig(),
        point_count=point_count,
        systems_global=_identity_systems(point_count),
    )
    template = [
        parameter.detach().numpy().copy()
        for parameter in base_material._network.parameters()
    ]

    def material_at_radius(radius: float) -> TannFCCBatch:
        trial = TannFCCBatch(
            TannFCCConfig(),
            point_count=point_count,
            systems_global=_identity_systems(point_count),
        )
        with torch.no_grad():
            for parameter, base in zip(
                trial._network.parameters(), template, strict=True
            ):
                parameter.copy_(torch.from_numpy(base * radius))
        return trial

    def loss_of(material: TannFCCBatch) -> tuple[float, tuple]:
        sequence = TannFCCSequence(
            grid=grid,
            material=material,
            boundary_history=boundary,
            measured_interior=measured,
            state_indices=STATES,
            holdout=HOLDOUT,
            solver_config=_solver_config(),
        )
        result = sequence.rollout()
        return result.total_loss_raw, result.records

    # The gate compares DIRECTIONAL derivatives along random directions in
    # theta space, per the preregistration ("plusieurs parametres/rayons
    # aleatoires de theta"): per-parameter comparisons degenerate at the
    # near-elastic radii, where individual entries of the gradient are
    # below the solver-tolerance noise floor of the loss.
    rng = np.random.default_rng(SEED_LOCAL)
    worst = 0.0
    for radius in (5.0, 20.0, 50.0):
        material = material_at_radius(radius)
        loss, records = loss_of(material)
        assert np.isfinite(loss)
        adjoint = TannFCCTrajectoryAdjoint(
            grid=grid, material=material, records=records
        )
        dtheta, _ = adjoint.sweep()
        scaled_template = [base * radius for base in template]
        for _ in range(2):
            direction = [
                rng.normal(size=parameter.shape)
                for parameter in scaled_template
            ]
            direction_norm = float(
                np.sqrt(sum(float(np.sum(d**2)) for d in direction))
            )
            direction = [d / direction_norm for d in direction]
            step = 1e-3
            values = []
            for sign in (+1.0, -1.0):
                trial = TannFCCBatch(
                    TannFCCConfig(),
                    point_count=point_count,
                    systems_global=_identity_systems(point_count),
                )
                with torch.no_grad():
                    for parameter, base, d in zip(
                        trial._network.parameters(), scaled_template, direction, strict=True
                    ):
                        parameter.copy_(torch.from_numpy(base + sign * step * d))
                trial_loss, _ = loss_of(trial)
                values.append(trial_loss)
            fd = (values[0] - values[1]) / (2 * step)
            adjoint_value = float(
                sum(np.sum(g * d) for g, d in zip(dtheta, direction, strict=True))
            )
            scale = max(abs(fd), abs(adjoint_value), 1e-30)
            worst = max(worst, abs(fd - adjoint_value) / scale)
    assert worst <= 1e-4


def test_tann_global_jacobian_transpose() -> None:
    pytest.importorskip("pyfftw")
    grid = StructuredGrid2D(8, 8, 1.0, 1.0)
    boundary, measured = _histories(grid)
    _, material, sequence, records = _rollout_loss(grid, boundary, measured)
    adjoint = TannFCCTrajectoryAdjoint(
        grid=grid, material=material, records=records
    )
    rng = np.random.default_rng(20260817)
    for index, record in enumerate(records):
        if record.committed_tangent_mpa is None:
            continue
        discrepancy = adjoint.dot_product_test(index, rng)
        assert discrepancy < 1e-8
