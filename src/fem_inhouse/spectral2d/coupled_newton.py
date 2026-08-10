"""Experimental Newton--Krylov driver for a coupled ``(u, chi)`` problem.

This module is deliberately not called by the production partitioned solver.
It defines the nonlinear seam required to compare a monolithic solve with the
existing Picard/Aitken implementation.  An adapter owns constitutive state,
transactions and the four current Jacobian-block actions; this driver owns
only Newton updates and the block Krylov solve.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.spectral2d.coupled_blocks import CoupledBlockActions
from fem_inhouse.spectral2d.krylov import solve_nonsymmetric_krylov

FloatArray = NDArray[np.float64]
CoupledState = tuple[FloatArray, FloatArray]
CoupledResidual = Callable[[CoupledState], tuple[FloatArray, FloatArray]]


@dataclass(frozen=True, slots=True)
class CoupledLinearisation:
    """Residual and block actions at one trial state."""

    mechanical_residual: FloatArray
    nonlocal_residual: FloatArray
    actions: CoupledBlockActions


@dataclass(frozen=True, slots=True)
class CoupledNewtonConfig:
    """Conservative controls for the experimental coupled driver."""

    maximum_iterations: int = 20
    relative_tolerance: float = 1.0e-8
    absolute_tolerance: float = 1.0e-10
    krylov_relative_tolerance: float = 1.0e-10
    krylov_maximum_iterations: int = 500
    krylov_restart: int = 100
    evaluate_initial_residual: bool = True


@dataclass(frozen=True, slots=True)
class CoupledNewtonResult:
    """Result and diagnostics of one experimental coupled solve."""

    mechanical: FloatArray
    nonlocal_field: FloatArray
    converged: bool
    iterations: int
    initial_residual_norm: float
    final_residual_norm: float
    krylov_iterations: tuple[int, ...]


def solve_coupled_newton(
    initial_mechanical: ArrayLike,
    initial_nonlocal: ArrayLike,
    evaluate: Callable[[CoupledState], CoupledLinearisation],
    *,
    config: CoupledNewtonConfig | None = None,
    evaluate_residual: CoupledResidual | None = None,
) -> CoupledNewtonResult:
    """Solve a coupled residual with matrix-free Newton--Krylov steps.

    ``evaluate`` must return residuals and a linearisation at the supplied
    state.  ``evaluate_residual`` may provide a cheaper residual-only path;
    when supplied, the full linearisation is built only for a state that still
    needs a Newton correction.  This avoids constructing a Jacobian after
    convergence.  Both callbacks are responsible for starting from a valid
    committed material state.  This first driver has no line search or cutback
    policy by design; those policies remain outside the experimental
    comparison until the block formulation is qualified.
    """

    controls = CoupledNewtonConfig() if config is None else config
    if controls.maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    mechanical = np.array(initial_mechanical, dtype=np.float64, copy=True).reshape(-1)
    nonlocal_field = np.array(initial_nonlocal, dtype=np.float64, copy=True).reshape(-1)
    if not np.isfinite(mechanical).all() or not np.isfinite(nonlocal_field).all():
        raise ValueError("initial state must be finite")

    def residual_norm(residual: tuple[FloatArray, FloatArray]) -> float:
        return float(
            max(
                np.linalg.norm(residual[0]),
                np.linalg.norm(residual[1]),
            )
        )

    def norm(linearisation: CoupledLinearisation) -> float:
        return residual_norm(
            (linearisation.mechanical_residual, linearisation.nonlocal_residual)
        )

    state = (mechanical, nonlocal_field)
    initial_residual = (
        evaluate_residual(state)
        if evaluate_residual is not None and controls.evaluate_initial_residual
        else None
    )
    initial = evaluate(state) if initial_residual is None else None
    initial_norm = (
        norm(initial) if initial_residual is None else residual_norm(initial_residual)
    )
    residual_scale = max(initial_norm, 1.0)
    krylov_iterations: list[int] = []
    if initial_norm <= controls.absolute_tolerance:
        return CoupledNewtonResult(
            mechanical=mechanical,
            nonlocal_field=nonlocal_field,
            converged=True,
            iterations=0,
            initial_residual_norm=initial_norm,
            final_residual_norm=initial_norm,
            krylov_iterations=(),
        )

    current = initial
    if current is None:
        current = evaluate(state)

    for iteration in range(1, controls.maximum_iterations + 1):
        rhs = -np.concatenate(
            (current.mechanical_residual, current.nonlocal_residual)
        )
        correction, info, calls = solve_nonsymmetric_krylov(
            current.actions.operator(),
            rhs,
            preconditioner=current.actions.preconditioner(),
            method="gmres",
            rtol=controls.krylov_relative_tolerance,
            maximum_iterations=controls.krylov_maximum_iterations,
            restart=controls.krylov_restart,
            callback=None,
        )
        krylov_iterations.append(calls)
        if info != 0:
            raise RuntimeError(f"coupled GMRES failed with info={info}")
        split = current.actions.mechanical_size
        mechanical += correction[:split]
        nonlocal_field += correction[split:]
        state = (mechanical, nonlocal_field)
        current_residual = (
            evaluate_residual(state)
            if evaluate_residual is not None
            else None
        )
        current = (
            evaluate(state) if current_residual is None else current
        )
        current_norm = (
            norm(current)
            if current_residual is None
            else residual_norm(current_residual)
        )
        if current_norm <= max(
            controls.absolute_tolerance,
            controls.relative_tolerance * residual_scale,
        ):
            return CoupledNewtonResult(
                mechanical=mechanical,
                nonlocal_field=nonlocal_field,
                converged=True,
                iterations=iteration,
                initial_residual_norm=initial_norm,
                final_residual_norm=current_norm,
                krylov_iterations=tuple(krylov_iterations),
            )
        if current_residual is not None:
            current = evaluate(state)

    return CoupledNewtonResult(
        mechanical=mechanical,
        nonlocal_field=nonlocal_field,
        converged=False,
        iterations=controls.maximum_iterations,
        initial_residual_norm=initial_norm,
        final_residual_norm=norm(current),
        krylov_iterations=tuple(krylov_iterations),
    )
