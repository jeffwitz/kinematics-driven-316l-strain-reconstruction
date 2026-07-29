#!/usr/bin/env python3
"""
Nonlinear finite-element solver for the kinematics-driven 316L case study.
=========================================================================
Element  : CPS4, plane stress, 2x2 Gauss
Material : von Mises + Ludwik hardening  sy(ep) = sy0 + K*ep^n
BCs      : All 4 edges prescribed from DIC data
Solver   : Incremental loading, Newton-Raphson with CONSISTENT tangent

Outputs (GP-averaged to element level, same grid as input maps):
  U    (nx_nodes, ny_nodes, 2)
  S    (nx_elems, ny_elems, 3)  [s11, s22, s12]
  E    (nx_elems, ny_elems, 3)  [e11, e22, g12]
  PE   (nx_elems, ny_elems, 3)
  PEEQ (nx_elems, ny_elems)
  RF   (nx_nodes, ny_nodes, 2)

Usage:
    from fem_pixel import run_fem
    result = run_fem(disp_x, disp_y, yield_map, K_map, n_exp=0.245,
                     x_size=0.1, y_size=0.1, element_size=0.001,
                     scale_factor=1.84, E_mod=205000., nu=0.3,
                     N_inc=20, verbose=True)
"""

import logging
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import cast

import numpy as np

from fem_inhouse.core.assembly import (
    FixedCSRAssembler,
    assemble_stiffness,
    element_tangent_stiffness,
    internal_force,
)
from fem_inhouse.core.constitutive import von_mises
from fem_inhouse.core.element import (
    GAUSS_POINT_COUNT,
    GAUSS_POINTS,
    GAUSS_WEIGHTS,
    plane_stress_elasticity,
    precompute_element,
)
from fem_inhouse.core.linear_solver import (
    LinearSystemMatrixType,
    create_linear_solver,
)
from fem_inhouse.core.mesh import StructuredMesh
from fem_inhouse.core.nonlocal_plasticity import (
    NonlocalCouplingConvergenceError,
    NonlocalCouplingEvaluation,
    NonlocalFixedPointWorkspace,
    NonlocalPlaneStressMaterialBatch,
    evaluate_nonlocal_fixed_point,
)
from fem_inhouse.core.plane_stress_material import (
    SYMMETRIC_TANGENT_RELATIVE_TOLERANCE,
    ConstitutiveIntegrationError,
    ConstitutiveTrial,
    InPlaneConstitutiveTrial,
    create_plane_stress_material_batch,
    relative_tangent_asymmetry,
)

LOGGER = logging.getLogger(__name__)

GP_XI = GAUSS_POINTS
GP_W = GAUSS_WEIGHTS
N_GP = GAUSS_POINT_COUNT


class NonlinearConvergenceError(RuntimeError):
    """Global convergence failure carrying machine-readable diagnostics."""

    def __init__(self, message: str, *, diagnostics: dict[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


# ── Main solver ──────────────────────────────────────────────────────────────
def run_fem(
    disp_x,
    disp_y,
    yield_map,
    K_map,
    n_exp,
    x_size,
    y_size,
    element_size,
    scale_factor,
    E_mod=205000.0,
    nu=0.3,
    N_inc=20,
    max_nr=15,
    nr_tol=1e-6,
    hardening="ludwik",
    ep_table_max=0.2,
    n_table=1000,
    first_positive_plastic_strain=1e-6,
    minimum_step_divisor=1024,
    constitutive_backend="mfront",
    mfront_library="build/mfront/src/libBehaviour.so",
    mfront_threads=1,
    local_plane_stress_tolerance_mpa=1e-8,
    local_plane_stress_relative_tolerance=1e-10,
    maximum_local_plane_stress_iterations=15,
    maximum_cbb_condition_number=1e12,
    newton_line_search=False,
    line_search_reduction=0.5,
    line_search_armijo_coefficient=1e-4,
    line_search_minimum_factor=2.0**-12,
    line_search_maximum_trials=12,
    boundary_history_predictor="elastic",
    nonlocal_plasticity_enabled=False,
    nonlocal_length_scale_mm=0.05888,
    nonlocal_coupling_modulus_mpa=0.0,
    nonlocal_relaxation=0.5,
    nonlocal_relaxation_strategy="fixed",
    nonlocal_minimum_relaxation=0.05,
    nonlocal_maximum_relaxation=0.8,
    nonlocal_aitken_residual_growth_factor=1.25,
    nonlocal_relative_tolerance=1e-6,
    nonlocal_maximum_iterations=15,
    nonlocal_maximum_helmholtz_residual=1e-10,
    nonlocal_record_iteration_history=False,
    snapshot_fractions=None,
    boundary_displacement_history=None,
    verbose=True,
):
    """
    hardening : 'ludwik'  - analytic sy = sy0 + K*ep^n
                'tabular' - piecewise-linear table clamped at ep_table_max
                            (matches the *Plastic table Abaqus interpolates)
    snapshot_fractions : optional list of load fractions in (0,1]; the S/E/PEEQ
                fields at those pseudo-times are recorded during ONE incremental
                solve and returned in result['frames'] = {fraction: {...}}.
                (Replaces re-solving the whole problem per load level.)
    boundary_displacement_history : optional nodal array with shape
                (N_inc + 1, nx + 1, ny + 1, 2). The first state must be zero
                and the last state must equal (disp_x, disp_y). Measured knots
                are reached exactly; cutbacks interpolate only between them.
    boundary_history_predictor : ``elastic`` uses the elastic response to the
                current boundary increment. ``secant-corrected-elastic`` adds
                the time-scaled nonlinear free-displacement correction from
                the previous converged increment. Constitutive internal
                variables are never extrapolated.
    constitutive_backend : 'python' uses the in-house return mapping; 'mfront'
                delegates stress, state update and consistent tangent to MGIS.
    """
    if nonlocal_plasticity_enabled and constitutive_backend == "python":
        raise ValueError("nonlocal plasticity currently requires an MFront backend")
    if newton_line_search and nonlocal_plasticity_enabled:
        raise ValueError("Newton line search is currently restricted to local plasticity")
    if boundary_history_predictor not in {
        "elastic",
        "secant-corrected-elastic",
    }:
        raise ValueError("unsupported boundary_history_predictor")
    if boundary_history_predictor != "elastic" and boundary_displacement_history is None:
        raise ValueError(
            "secant-corrected-elastic predictor requires boundary_displacement_history"
        )
    started_at = time.perf_counter()
    mesh = StructuredMesh(x_size, y_size, element_size, scale_factor)
    nx, ny = mesh.nx, mesh.ny
    nxn, nyn = nx + 1, ny + 1
    n_e = mesh.n_elems

    # element->grid output helpers (also used for snapshots)
    def gm(a):
        return a.mean(axis=1)

    def tg(a):
        if a.ndim == 1:
            return a.reshape(nx, ny, order="F")
        return a.reshape(nx, ny, *a.shape[1:], order="F")

    # Material per GP
    sy0_gp = np.repeat(yield_map.ravel(order="F"), N_GP)
    K_gp = np.repeat(K_map.ravel(order="F"), N_GP)
    material_batch = create_plane_stress_material_batch(
        constitutive_backend,
        sy0_gp,
        K_gp,
        n_exp,
        young_modulus_mpa=E_mod,
        poisson_ratio=nu,
        hardening_mode=hardening,
        plastic_strain_max=ep_table_max,
        plastic_table_points=n_table,
        first_positive_plastic_strain=first_positive_plastic_strain,
        mfront_library=mfront_library,
        mfront_threads=mfront_threads,
        local_plane_stress_options={
            "local_tolerance_mpa": local_plane_stress_tolerance_mpa,
            "local_relative_tolerance": local_plane_stress_relative_tolerance,
            "maximum_local_iterations": maximum_local_plane_stress_iterations,
            "maximum_cbb_condition_number": maximum_cbb_condition_number,
        },
        nonlocal_coupling_modulus_mpa=(
            nonlocal_coupling_modulus_mpa if nonlocal_plasticity_enabled else None
        ),
    )
    nonlocal_material_batch: NonlocalPlaneStressMaterialBatch | None = None
    if nonlocal_plasticity_enabled:
        if not hasattr(material_batch, "set_nonlocal_equivalent_plastic_strain"):
            raise TypeError("selected constitutive backend does not expose the nonlocal field")
        nonlocal_material_batch = cast(NonlocalPlaneStressMaterialBatch, material_batch)
    linear_system_matrix_type = cast(
        LinearSystemMatrixType,
        getattr(
            material_batch,
            "linear_system_matrix_type",
            "nonsymmetric",
        ),
    )
    linear_solver = create_linear_solver(linear_system_matrix_type)

    LOGGER.info(
        "nonlinear solve started",
        extra={
            "event": "nonlinear_solve_started",
            "elements": n_e,
            "free_dofs": len(mesh.dofs_free),
            "backend": linear_solver.backend_name,
            "linear_system_matrix_type": linear_system_matrix_type,
            "matrix_storage": linear_solver.matrix_storage,
        },
    )

    C_ps = plane_stress_elasticity(E_mod, nu)
    element_matrix_started_at = time.perf_counter()
    operators = precompute_element(mesh, C_ps)
    Ke = operators.elastic_stiffness
    Bs = operators.strain_displacement
    dJs = operators.jacobian_determinants
    element_matrix_seconds = time.perf_counter() - element_matrix_started_at
    plastic_point_indices = np.arange(n_e * N_GP)

    # DOF maps
    nd_all = mesh.node_ids
    ld = mesh.location_matrix()
    dof_I = mesh.dofs_free
    dof_B = mesh.dofs_bc
    initialization_seconds = time.perf_counter() - started_at

    # The reduced stiffness graph and contribution mapping remain fixed.
    assembly_started_at = time.perf_counter()
    fixed_free_assembler = FixedCSRAssembler.from_location_matrix(
        ld,
        dof_I,
        storage=linear_solver.matrix_storage,
    )

    # Assemble KIB once. KII is the fixed CSR object later updated in place.
    sparse_assembly_started_at = time.perf_counter()
    K_el = assemble_stiffness(mesh, Ke, ld)
    KII_el = fixed_free_assembler.assemble(Ke)
    sparse_assembly_seconds = time.perf_counter() - sparse_assembly_started_at
    extraction_started_at = time.perf_counter()
    KIB_el = K_el[dof_I][:, dof_B].tocsr()
    del K_el
    free_system_extraction_seconds = time.perf_counter() - extraction_started_at
    elastic_assembly_seconds = time.perf_counter() - assembly_started_at
    constitutive_seconds = 0.0
    tangent_assembly_seconds = 0.0
    internal_force_seconds = 0.0
    maximum_relative_constitutive_tangent_asymmetry = 0.0
    line_search_evaluations = 0
    line_search_reductions = 0
    line_search_failures = 0
    minimum_accepted_line_search_factor = 1.0
    secant_predictor_uses = 0
    secant_predictor_fallbacks = 0

    def timed_solve(matrix, right_hand_side):
        return linear_solver.factorize_and_solve(matrix, right_hand_side)

    def solve_el(b):
        return timed_solve(KII_el, b)

    # Prescribed displacements
    u_bc = np.zeros(mesh.n_dof)
    u_bc[2 * nd_all.ravel(order="F")] = disp_x.ravel(order="F")
    u_bc[2 * nd_all.ravel(order="F") + 1] = disp_y.ravel(order="F")
    history = None
    boundary_target: Callable[[float], np.ndarray] | None = None
    if boundary_displacement_history is not None:
        history = np.asarray(boundary_displacement_history, dtype=np.float64)
        expected_shape = (N_inc + 1, nxn, nyn, 2)
        if history.shape != expected_shape:
            raise ValueError(
                "boundary_displacement_history has shape "
                f"{history.shape}, expected {expected_shape}"
            )
        if not np.isfinite(history).all():
            raise ValueError("boundary_displacement_history must contain finite values")
        if not np.allclose(history[0], 0.0, rtol=0.0, atol=1.0e-14):
            raise ValueError("boundary_displacement_history must start from zero")
        final_displacement = np.stack((disp_x, disp_y), axis=-1)
        if not np.allclose(history[-1], final_displacement, rtol=0.0, atol=1.0e-14):
            raise ValueError(
                "boundary_displacement_history final state must match disp_x and disp_y"
            )
        history_dofs = np.zeros((N_inc + 1, mesh.n_dof), dtype=np.float64)
        history_dofs[:, 2 * nd_all.ravel(order="F")] = np.stack(
            [frame.ravel(order="F") for frame in history[..., 0]]
        )
        history_dofs[:, 2 * nd_all.ravel(order="F") + 1] = np.stack(
            [frame.ravel(order="F") for frame in history[..., 1]]
        )

        def boundary_target(pseudo_time: float) -> np.ndarray:
            scaled = min(max(float(pseudo_time), 0.0), 1.0) * N_inc
            lower = min(int(np.floor(scaled)), N_inc - 1)
            local = scaled - lower
            return (1.0 - local) * history_dofs[lower, dof_B] + local * history_dofs[
                lower + 1, dof_B
            ]

        elastic_predictor_direction = None
    else:
        history_dofs = None
        boundary_target = None
        elastic_predictor_direction = solve_el(-KIB_el @ u_bc[dof_B])

    # State
    u = np.zeros(mesh.n_dof)
    eps_p = np.zeros((n_e, N_GP, 3))
    ep_bar = np.zeros((n_e, N_GP))
    sig = np.zeros((n_e, N_GP, 3))
    eps_tot = np.zeros((n_e, N_GP, 3))
    accepted_constitutive_trial: ConstitutiveTrial | None = None
    chi_committed = np.zeros((nx, ny), dtype=np.float64)
    chi_trial_guess = chi_committed.copy()
    nonlocal_workspace = (
        NonlocalFixedPointWorkspace.create((nx, ny), N_GP) if nonlocal_plasticity_enabled else None
    )
    accepted_nonlocal_evaluation: NonlocalCouplingEvaluation | None = None
    previous_accepted_boundary_increment: np.ndarray | None = None
    previous_accepted_free_increment: np.ndarray | None = None
    previous_accepted_elastic_free_increment: np.ndarray | None = None
    previous_accepted_step_size: float | None = None

    # Incremental loading with automatic cutback (Abaqus-style):
    # pseudo-time t: 0 -> 1, initial/maximum step 1/N_inc, halved on failure.
    t = 0.0
    dt = 1.0 / N_inc
    dt_max = 1.0 / N_inc
    dt_min = dt_max / minimum_step_divisor
    inc = 0
    converged_increments = 0
    cutbacks = 0
    total_newton_iterations = 0
    maximum_newton_iterations = 0
    final_residual_norm = float("nan")
    final_relative_residual = float("nan")
    final_convergence_criterion = "none"
    nonlocal_iterations_per_newton: list[int] = []
    nonlocal_iterations_per_increment: list[int] = []
    nonlocal_total_iterations = 0
    nonlocal_maximum_iterations_observed = 0
    nonlocal_coupling_failures = 0
    nonlocal_final_relative_residual = 0.0
    nonlocal_maximum_helmholtz_residual_observed = 0.0
    nonlocal_maximum_absolute_mean_drift = 0.0
    nonlocal_mfront_seconds = 0.0
    nonlocal_mfront_without_tangent_seconds = 0.0
    nonlocal_mfront_with_tangent_seconds = 0.0
    helmholtz_seconds = 0.0
    nonlocal_fixed_point_history: list[dict[str, object]] = []
    last_failed_fixed_point_history: list[dict[str, object]] = []
    first_cutback: dict[str, object] | None = None
    first_constitutive_failure: dict[str, object] | None = None
    last_constitutive_failure: dict[str, object] | None = None
    snaps = {}
    pending = sorted(snapshot_fractions) if snapshot_fractions else []
    while t < 1.0 - 1e-12:
        dt = min(dt, 1.0 - t)
        if history is not None:
            next_history_knot = (np.floor(t * N_inc + 1.0e-10) + 1.0) / N_inc
            dt = min(dt, max(next_history_knot - t, 1.0e-12))
        if pending:  # land exactly on snapshot fractions
            dt = min(dt, max(pending[0] - t, 1e-12))
        inc += 1
        if history is None:
            du_B = u_bc[dof_B] * dt
        else:
            assert boundary_target is not None
            du_B = boundary_target(t + dt) - boundary_target(t)
        u_save = u.copy()

        u[dof_B] += du_B
        # Elastic predictor (reused factorization)
        if history is None:
            assert elastic_predictor_direction is not None
            elastic_free_increment = elastic_predictor_direction * dt
        else:
            elastic_free_increment = solve_el(-KIB_el @ du_B)
        predictor_free_increment = elastic_free_increment
        if (
            boundary_history_predictor == "secant-corrected-elastic"
            and previous_accepted_boundary_increment is not None
            and previous_accepted_free_increment is not None
            and previous_accepted_elastic_free_increment is not None
            and previous_accepted_step_size is not None
        ):
            alignment = float(np.dot(du_B, previous_accepted_boundary_increment))
            if alignment > 0.0:
                previous_nonlinear_correction = (
                    previous_accepted_free_increment
                    - previous_accepted_elastic_free_increment
                )
                predictor_free_increment = (
                    elastic_free_increment
                    + (dt / previous_accepted_step_size) * previous_nonlinear_correction
                )
                secant_predictor_uses += 1
            else:
                secant_predictor_fallbacks += 1
        u[dof_I] += predictor_free_increment

        KII = KII_el  # start with elastic tangent, replaced after 1st iter

        # Saved converged state (updated each NR iter before possible break)
        sf_acc = sig.copy()
        eps_p_acc = eps_p.copy()
        ep_new = ep_bar.copy()
        constitutive_trial_acc: InPlaneConstitutiveTrial | None = None
        nonlocal_evaluation_acc: NonlocalCouplingEvaluation | None = None
        increment_nonlocal_iterations = 0
        converged = False
        increment_failure_reason = "newton_iterations_exhausted"
        failed_newton_iteration = 0

        for nrit in range(max_nr):
            failed_newton_iteration = nrit + 1
            total_newton_iterations += 1
            maximum_newton_iterations = max(maximum_newton_iterations, nrit + 1)
            u_e = u[ld]
            eps_tot = np.einsum("gak,ek->ega", Bs, u_e)
            if not np.isfinite(eps_tot).all():
                break
            # Constitutive trial from the last converged material state.
            constitutive_started_at = time.perf_counter()
            try:
                if nonlocal_plasticity_enabled:
                    if nonlocal_material_batch is None:
                        raise RuntimeError("nonlocal material adapter was not initialised")
                    nonlocal_evaluation = evaluate_nonlocal_fixed_point(
                        nonlocal_material_batch,
                        eps_tot.reshape(-1, 3),
                        time_increment=dt,
                        element_shape=(nx, ny),
                        gauss_points_per_element=N_GP,
                        initial_nonlocal_peeq=chi_trial_guess,
                        length_scale_mm=nonlocal_length_scale_mm,
                        spacing_x_mm=mesh.element_size,
                        spacing_y_mm=mesh.element_size,
                        coupling_modulus_mpa=nonlocal_coupling_modulus_mpa,
                        relaxation=nonlocal_relaxation,
                        relaxation_strategy=nonlocal_relaxation_strategy,
                        minimum_relaxation=nonlocal_minimum_relaxation,
                        maximum_relaxation=nonlocal_maximum_relaxation,
                        aitken_residual_growth_factor=(nonlocal_aitken_residual_growth_factor),
                        relative_tolerance=nonlocal_relative_tolerance,
                        maximum_iterations=nonlocal_maximum_iterations,
                        maximum_helmholtz_residual=nonlocal_maximum_helmholtz_residual,
                        workspace=nonlocal_workspace,
                    )
                    trial = nonlocal_evaluation.constitutive_trial
                    np.copyto(chi_trial_guess, nonlocal_evaluation.nonlocal_peeq)
                    nonlocal_evaluation_acc = nonlocal_evaluation
                    trace_start = len(nonlocal_fixed_point_history)
                    if nonlocal_record_iteration_history:
                        nonlocal_fixed_point_history.extend(
                            {
                                "increment": inc,
                                "pseudo_time": t + dt,
                                "step_size": dt,
                                "newton_iteration": nrit + 1,
                                "mechanical_residual_norm": None,
                                "mechanical_relative_residual": None,
                                **asdict(item),
                            }
                            for item in nonlocal_evaluation.iteration_history
                        )
                    nonlocal_iterations_per_newton.append(nonlocal_evaluation.iterations)
                    increment_nonlocal_iterations += nonlocal_evaluation.iterations
                    nonlocal_total_iterations += nonlocal_evaluation.iterations
                    nonlocal_maximum_iterations_observed = max(
                        nonlocal_maximum_iterations_observed,
                        nonlocal_evaluation.iterations,
                    )
                    nonlocal_final_relative_residual = nonlocal_evaluation.relative_residual
                    nonlocal_maximum_helmholtz_residual_observed = max(
                        nonlocal_maximum_helmholtz_residual_observed,
                        nonlocal_evaluation.helmholtz_residual_relative,
                    )
                    nonlocal_maximum_absolute_mean_drift = max(
                        nonlocal_maximum_absolute_mean_drift,
                        abs(nonlocal_evaluation.mean_drift),
                    )
                    nonlocal_mfront_seconds += nonlocal_evaluation.mfront_seconds
                    nonlocal_mfront_without_tangent_seconds += (
                        nonlocal_evaluation.mfront_without_tangent_seconds
                    )
                    nonlocal_mfront_with_tangent_seconds += (
                        nonlocal_evaluation.mfront_with_tangent_seconds
                    )
                    helmholtz_seconds += nonlocal_evaluation.helmholtz_seconds
                else:
                    evaluate_in_plane = getattr(
                        material_batch,
                        "evaluate_in_plane",
                        material_batch.evaluate,
                    )
                    trial = evaluate_in_plane(
                        eps_tot.reshape(-1, 3),
                        time_increment=dt,
                        consistent_tangent=True,
                    )
            except ConstitutiveIntegrationError as error:
                if nonlocal_plasticity_enabled:
                    nonlocal_coupling_failures += 1
                increment_failure_reason = str(error)
                absolute_strain = np.abs(eps_tot)
                flat_index = int(np.argmax(absolute_strain))
                element_index, gauss_point_index, component_index = np.unravel_index(
                    flat_index,
                    eps_tot.shape,
                )
                last_constitutive_failure = {
                    "pseudo_time": float(t + dt),
                    "step_size": float(dt),
                    "newton_iteration": nrit + 1,
                    "maximum_absolute_engineering_strain": float(absolute_strain.flat[flat_index]),
                    "component_minima": [float(value) for value in np.min(eps_tot, axis=(0, 1))],
                    "component_maxima": [float(value) for value in np.max(eps_tot, axis=(0, 1))],
                    "element_index": int(element_index),
                    "gauss_point_index": int(gauss_point_index),
                    "component_index": int(component_index),
                }
                if first_constitutive_failure is None:
                    first_constitutive_failure = dict(last_constitutive_failure)
                if isinstance(error, NonlocalCouplingConvergenceError):
                    last_failed_fixed_point_history = [
                        {
                            "increment": inc,
                            "pseudo_time": t + dt,
                            "step_size": dt,
                            "newton_iteration": nrit + 1,
                            "mechanical_residual_norm": None,
                            "mechanical_relative_residual": None,
                            "failure_reason": error.reason,
                            **asdict(item),
                        }
                        for item in error.iteration_history
                    ]
                    if nonlocal_record_iteration_history:
                        nonlocal_fixed_point_history.extend(last_failed_fixed_point_history)
                LOGGER.warning(
                    "constitutive trial failed: %s",
                    error,
                    extra={
                        "event": "constitutive_trial_failed",
                        "increment": inc,
                        "iteration": nrit + 1,
                        "reason": str(error),
                    },
                )
                constitutive_seconds += time.perf_counter() - constitutive_started_at
                break
            sf = trial.stress_in_plane_mpa.reshape(n_e, N_GP, 3)
            eps_p_trial = trial.observables["plastic_strain_2d"].reshape(n_e, N_GP, 3)
            ep_new = trial.observables["equivalent_plastic_strain"].reshape(n_e, N_GP)
            material_tangents = trial.tangent_in_plane_mpa
            if material_tangents is None:
                raise RuntimeError("constitutive backend did not return a consistent tangent")
            if linear_system_matrix_type == "symmetric_positive_definite":
                tangent_asymmetry = relative_tangent_asymmetry(material_tangents)
                maximum_relative_constitutive_tangent_asymmetry = max(
                    maximum_relative_constitutive_tangent_asymmetry,
                    tangent_asymmetry,
                )
                if tangent_asymmetry > SYMMETRIC_TANGENT_RELATIVE_TOLERANCE:
                    raise RuntimeError(
                        "constitutive backend declared a symmetric tangent but "
                        f"its relative asymmetry is {tangent_asymmetry:.3e}"
                    )
            constitutive_seconds += time.perf_counter() - constitutive_started_at

            # Save state from this iteration (used if we break here)
            sf_acc = sf
            eps_p_acc = eps_p_trial
            constitutive_trial_acc = trial

            # Internal forces and residual
            internal_force_started_at = time.perf_counter()
            R = internal_force(mesh, sf, Bs, dJs, ld)
            internal_force_seconds += time.perf_counter() - internal_force_started_at
            R_I = R[dof_I]
            res = float(np.linalg.norm(R_I))
            if not np.isfinite(res):
                break  # diverged -> cutback
            if nrit == 0:
                res0 = max(res, 1e-30)
            rel = res / res0
            if nonlocal_plasticity_enabled and nonlocal_record_iteration_history:
                for record in nonlocal_fixed_point_history[trace_start:]:
                    record["mechanical_residual_norm"] = float(res)
                    record["mechanical_relative_residual"] = float(rel)
            if verbose:
                LOGGER.info(
                    "Newton iteration",
                    extra={
                        "event": "newton_iteration",
                        "increment": inc,
                        "pseudo_time": t + dt,
                        "iteration": nrit + 1,
                        "residual_norm": float(res),
                        "relative_residual": float(rel),
                    },
                )

            # Converge on absolute OR relative residual
            if res < 1e-10 or rel < nr_tol:
                converged = True
                final_residual_norm = float(res)
                final_relative_residual = float(rel)
                final_convergence_criterion = (
                    "absolute_residual" if res < 1e-10 else "relative_residual"
                )
                break

            # Build consistent tangent stiffness
            tangent_started_at = time.perf_counter()
            plastic_tangents = material_tangents
            element_matrix_started_at = time.perf_counter()
            Ke_ep = element_tangent_stiffness(
                Ke,
                C_ps,
                plastic_tangents,
                plastic_point_indices,
                Bs,
                dJs,
                element_count=n_e,
            )
            element_matrix_seconds += time.perf_counter() - element_matrix_started_at
            sparse_assembly_started_at = time.perf_counter()
            K_tang = fixed_free_assembler.assemble(Ke_ep)
            sparse_assembly_seconds += time.perf_counter() - sparse_assembly_started_at
            KII = K_tang
            tangent_assembly_seconds += time.perf_counter() - tangent_started_at

            du = timed_solve(KII, -R_I)
            if not np.isfinite(du).all():
                break  # singular tangent -> cutback
            if not newton_line_search:
                u[dof_I] += du
                continue

            base_u_I = u[dof_I].copy()
            factor = 1.0
            accepted_line_search = False
            for _line_search_trial in range(line_search_maximum_trials):
                if factor < line_search_minimum_factor:
                    break
                line_search_evaluations += 1
                u[dof_I] = base_u_I + factor * du
                candidate_eps = np.einsum("gak,ek->ega", Bs, u[ld])
                try:
                    candidate_started_at = time.perf_counter()
                    candidate_trial = evaluate_in_plane(
                        candidate_eps.reshape(-1, 3),
                        time_increment=dt,
                        consistent_tangent=False,
                    )
                    constitutive_seconds += time.perf_counter() - candidate_started_at
                    candidate_stress = candidate_trial.stress_in_plane_mpa.reshape(
                        n_e,
                        N_GP,
                        3,
                    )
                    candidate_force_started_at = time.perf_counter()
                    candidate_residual = internal_force(
                        mesh,
                        candidate_stress,
                        Bs,
                        dJs,
                        ld,
                    )[dof_I]
                    internal_force_seconds += time.perf_counter() - candidate_force_started_at
                    candidate_norm = float(np.linalg.norm(candidate_residual))
                except ConstitutiveIntegrationError:
                    candidate_norm = float("inf")
                target_norm = (1.0 - line_search_armijo_coefficient * factor) * res
                if np.isfinite(candidate_norm) and candidate_norm <= target_norm:
                    accepted_line_search = True
                    minimum_accepted_line_search_factor = min(
                        minimum_accepted_line_search_factor,
                        factor,
                    )
                    break
                factor *= line_search_reduction
                line_search_reductions += 1
            if not accepted_line_search:
                u[dof_I] = base_u_I
                line_search_failures += 1
                increment_failure_reason = "Newton line search found no residual-decreasing trial"
                break

        if not converged:
            material_batch.revert()
            np.copyto(chi_trial_guess, chi_committed)
            u = u_save
            dt *= 0.5
            cutbacks += 1
            if first_cutback is None:
                first_cutback = {
                    "increment": inc,
                    "newton_iteration": failed_newton_iteration,
                    "pseudo_time": t + 2.0 * dt,
                    "failed_step_size": 2.0 * dt,
                    "next_step_size": dt,
                    "reason": increment_failure_reason,
                }
            LOGGER.warning(
                "increment cutback",
                extra={
                    "event": "increment_cutback",
                    "increment": inc,
                    "next_step": dt,
                },
            )
            if dt < dt_min:
                raise NonlinearConvergenceError(
                    f"run_fem: increment cutback below minimum ({dt:.2e}) "
                    "- solution not converging",
                    diagnostics={
                        "first_cutback": first_cutback,
                        "last_cutback": {
                            "increment": inc,
                            "newton_iteration": failed_newton_iteration,
                            "pseudo_time": t + 2.0 * dt,
                            "failed_step_size": 2.0 * dt,
                            "next_step_size": dt,
                            "reason": increment_failure_reason,
                        },
                        "attempted_increments": inc,
                        "converged_increments": converged_increments,
                        "cutbacks": cutbacks,
                        "total_newton_iterations": total_newton_iterations,
                        "relaxation_strategy": nonlocal_relaxation_strategy,
                        "fixed_point_history": nonlocal_fixed_point_history,
                        "last_failed_fixed_point_history": (last_failed_fixed_point_history),
                        "first_constitutive_failure": first_constitutive_failure,
                        "last_constitutive_failure": last_constitutive_failure,
                        "line_search_evaluations": line_search_evaluations,
                        "line_search_reductions": line_search_reductions,
                        "line_search_failures": line_search_failures,
                        "minimum_accepted_line_search_factor": (
                            minimum_accepted_line_search_factor
                        ),
                        "boundary_history_predictor": boundary_history_predictor,
                        "secant_predictor_uses": secant_predictor_uses,
                        "secant_predictor_fallbacks": secant_predictor_fallbacks,
                    },
                )
            continue

        if constitutive_trial_acc is None:
            raise RuntimeError("global Newton converged without a constitutive trial")
        if t + dt >= 1.0 - 1e-12:
            complete_trial = getattr(material_batch, "complete_trial", None)
            if complete_trial is None:
                if not isinstance(constitutive_trial_acc, ConstitutiveTrial):
                    raise RuntimeError(
                        "constitutive backend cannot complete the final tensor state"
                    )
                accepted_constitutive_trial = constitutive_trial_acc
            else:
                accepted_constitutive_trial = complete_trial(constitutive_trial_acc)
        material_batch.commit()
        if nonlocal_plasticity_enabled:
            if nonlocal_evaluation_acc is None:
                raise RuntimeError("coupled Newton converged without a nonlocal evaluation")
            accepted_nonlocal_evaluation = nonlocal_evaluation_acc
            np.copyto(chi_committed, nonlocal_evaluation_acc.nonlocal_peeq)
            np.copyto(chi_trial_guess, chi_committed)
            nonlocal_iterations_per_increment.append(increment_nonlocal_iterations)
        eps_p = eps_p_acc
        ep_bar = ep_new
        sig = sf_acc
        previous_accepted_boundary_increment = du_B.copy()
        previous_accepted_free_increment = u[dof_I] - u_save[dof_I]
        previous_accepted_elastic_free_increment = elastic_free_increment.copy()
        previous_accepted_step_size = dt
        t += dt
        converged_increments += 1
        dt = min(dt * 1.5, dt_max)  # grow back after success

        # record snapshot fields at requested load fractions (element fields
        # + nodal U, so a stress/strain/u2 - "time" curve can be built later
        # without re-solving; same nodal-reshape as the final-state U below)
        while pending and t >= pending[0] - 1e-9:
            fsnap = pending.pop(0)
            Usnap = np.zeros((nxn, nyn, 2))
            Usnap[..., 0] = u[2 * nd_all].reshape(nxn, nyn)
            Usnap[..., 1] = u[2 * nd_all + 1].reshape(nxn, nyn)
            snaps[fsnap] = dict(S=tg(gm(sig)), E=tg(gm(eps_tot)), PEEQ=tg(gm(ep_bar)), U=Usnap)
            LOGGER.info(
                "snapshot recorded",
                extra={
                    "event": "snapshot_recorded",
                    "load_fraction": fsnap,
                },
            )

    # Output
    output_started_at = time.perf_counter()
    internal_force_started_at = time.perf_counter()
    F_all = internal_force(mesh, sig, Bs, dJs, ld)
    internal_force_seconds += time.perf_counter() - internal_force_started_at
    bc_m = np.zeros(mesh.n_dof, dtype=bool)
    bc_m[dof_B] = True

    U = np.zeros((nxn, nyn, 2))
    U[..., 0] = u[2 * nd_all].reshape(nxn, nyn)
    U[..., 1] = u[2 * nd_all + 1].reshape(nxn, nyn)
    RF = np.zeros((nxn, nyn, 2))
    RF[..., 0] = np.where(bc_m[2 * nd_all], F_all[2 * nd_all], 0.0).reshape(nxn, nyn)
    RF[..., 1] = np.where(bc_m[2 * nd_all + 1], F_all[2 * nd_all + 1], 0.0).reshape(nxn, nyn)

    S = tg(gm(sig))
    E = tg(gm(eps_tot))
    PE = tg(gm(eps_p))
    if accepted_constitutive_trial is None:
        raise RuntimeError("converged solve has no accepted constitutive trial")
    stress_3d = tg(gm(accepted_constitutive_trial.full_stress_tensor_mpa.reshape(n_e, N_GP, 3, 3)))
    strain_3d = tg(gm(accepted_constitutive_trial.full_strain_tensor.reshape(n_e, N_GP, 3, 3)))
    elastic_strain_3d = tg(
        gm(accepted_constitutive_trial.elastic_strain_tensor.reshape(n_e, N_GP, 3, 3))
    )
    plastic_strain_3d = tg(
        gm(accepted_constitutive_trial.plastic_strain_tensor.reshape(n_e, N_GP, 3, 3))
    )
    residual_vector = tg(
        gm(accepted_constitutive_trial.plane_stress_residual_mpa.reshape(n_e, N_GP, 3))
    )
    residual_s33 = residual_vector[..., 0]
    local_statistics = material_batch.statistics
    material_timing = getattr(material_batch, "timing_statistics", None)
    mfront_integration_without_tangent_seconds = float(
        getattr(material_timing, "integration_without_tangent_seconds", 0.0)
    )
    mfront_integration_with_tangent_seconds = float(
        getattr(material_timing, "integration_with_tangent_seconds", 0.0)
    )
    kelvin_conversion_seconds = float(getattr(material_timing, "kelvin_conversion_seconds", 0.0))
    tensor_reconstruction_seconds = float(
        getattr(material_timing, "tensor_reconstruction_seconds", 0.0)
    )
    mfront_integration_without_tangent_calls = int(
        getattr(material_timing, "integration_without_tangent_calls", 0)
    )
    mfront_integration_with_tangent_calls = int(
        getattr(material_timing, "integration_with_tangent_calls", 0)
    )
    tensor_reconstruction_calls = int(getattr(material_timing, "tensor_reconstruction_calls", 0))
    nonlocal_fields = {}
    if nonlocal_plasticity_enabled:
        if accepted_nonlocal_evaluation is None:
            raise RuntimeError("converged coupled solve has no accepted nonlocal evaluation")
        nonlocal_fields = {
            "PEEQ_NONLOCAL": accepted_nonlocal_evaluation.nonlocal_peeq,
            "PEEQ_MISMATCH": accepted_nonlocal_evaluation.mismatch,
            "NONLOCAL_HARDENING_MPA": (accepted_nonlocal_evaluation.nonlocal_hardening_mpa),
            "YIELD_SURFACE_RADIUS_MPA": (accepted_nonlocal_evaluation.yield_surface_radius_mpa),
            "NONLOCAL_RESIDUAL": accepted_nonlocal_evaluation.residual_field,
        }

    output_seconds = time.perf_counter() - output_started_at
    linear_solver_statistics = linear_solver.statistics
    linear_solver.close()
    linear_solve_seconds = linear_solver_statistics.total_seconds
    elapsed_seconds = time.perf_counter() - started_at
    LOGGER.info(
        "nonlinear solve completed",
        extra={
            "event": "nonlinear_solve_completed",
            "elapsed_seconds": elapsed_seconds,
            "attempted_increments": inc,
            "converged_increments": converged_increments,
            "cutbacks": cutbacks,
            "total_newton_iterations": total_newton_iterations,
        },
    )
    return dict(
        U=U,
        S=S,
        E=E,
        PE=PE,
        PEEQ=tg(gm(ep_bar)),
        RF=RF,
        S_3D=stress_3d,
        E_3D=strain_3d,
        EE_3D=elastic_strain_3d,
        PE_3D=plastic_strain_3d,
        PLANE_STRESS_RESIDUAL_MPA=residual_vector,
        S33_RESIDUAL_MPA=residual_s33,
        **nonlocal_fields,
        mesh=mesh,
        frames=snaps,
        diagnostics=dict(
            backend=(f"{linear_solver.backend_name}; constitutive={material_batch.backend_name}"),
            elapsed_seconds=elapsed_seconds,
            initialization_seconds=initialization_seconds,
            elastic_assembly_seconds=elastic_assembly_seconds,
            constitutive_seconds=constitutive_seconds,
            tangent_assembly_seconds=tangent_assembly_seconds,
            linear_solve_seconds=linear_solve_seconds,
            output_seconds=output_seconds,
            attempted_increments=inc,
            converged_increments=converged_increments,
            cutbacks=cutbacks,
            total_newton_iterations=total_newton_iterations,
            maximum_newton_iterations=maximum_newton_iterations,
            final_residual_norm=final_residual_norm,
            final_relative_residual=final_relative_residual,
            final_convergence_criterion=final_convergence_criterion,
            newton_line_search_enabled=newton_line_search,
            line_search_evaluations=line_search_evaluations,
            line_search_reductions=line_search_reductions,
            line_search_failures=line_search_failures,
            minimum_accepted_line_search_factor=(minimum_accepted_line_search_factor),
            boundary_history_predictor=boundary_history_predictor,
            secant_predictor_uses=secant_predictor_uses,
            secant_predictor_fallbacks=secant_predictor_fallbacks,
            tensor_reconstruction_source=material_batch.completion_strategy,
            linear_system_matrix_type=linear_system_matrix_type,
            maximum_relative_constitutive_tangent_asymmetry=(
                maximum_relative_constitutive_tangent_asymmetry
            ),
            maximum_gauss_point_plane_stress_residual_mpa=(
                local_statistics.maximum_gauss_point_plane_stress_residual_mpa
            ),
            maximum_local_plane_stress_iterations=(
                local_statistics.maximum_local_plane_stress_iterations
            ),
            mean_local_plane_stress_iterations=(
                local_statistics.mean_local_plane_stress_iterations
            ),
            local_plane_stress_failures=local_statistics.local_plane_stress_failures,
            maximum_cbb_condition_number=local_statistics.maximum_cbb_condition_number,
            nonlocal_plasticity_enabled=nonlocal_plasticity_enabled,
            nonlocal_convergence_norm=(
                "mixed_relative_linf" if nonlocal_plasticity_enabled else "not_applicable"
            ),
            nonlocal_length_scale_mm=nonlocal_length_scale_mm,
            nonlocal_coupling_modulus_mpa=nonlocal_coupling_modulus_mpa,
            nonlocal_relaxation=nonlocal_relaxation,
            nonlocal_relaxation_strategy=nonlocal_relaxation_strategy,
            nonlocal_minimum_relaxation=nonlocal_minimum_relaxation,
            nonlocal_maximum_relaxation=nonlocal_maximum_relaxation,
            nonlocal_aitken_residual_growth_factor=(nonlocal_aitken_residual_growth_factor),
            nonlocal_fixed_point_history=tuple(nonlocal_fixed_point_history),
            nonlocal_iterations_per_newton=tuple(nonlocal_iterations_per_newton),
            nonlocal_iterations_per_increment=tuple(nonlocal_iterations_per_increment),
            total_nonlocal_iterations=nonlocal_total_iterations,
            maximum_nonlocal_iterations=nonlocal_maximum_iterations_observed,
            mean_nonlocal_iterations=(
                float(np.mean(nonlocal_iterations_per_newton))
                if nonlocal_iterations_per_newton
                else 0.0
            ),
            final_nonlocal_relative_residual=nonlocal_final_relative_residual,
            maximum_helmholtz_residual_relative=(nonlocal_maximum_helmholtz_residual_observed),
            maximum_absolute_nonlocal_mean_drift=(nonlocal_maximum_absolute_mean_drift),
            helmholtz_seconds=helmholtz_seconds,
            nonlocal_mfront_seconds=nonlocal_mfront_seconds,
            nonlocal_coupling_failures=nonlocal_coupling_failures,
            mfront_integration_without_tangent_seconds=(mfront_integration_without_tangent_seconds),
            mfront_integration_with_tangent_seconds=(mfront_integration_with_tangent_seconds),
            kelvin_conversion_seconds=kelvin_conversion_seconds,
            tensor_reconstruction_seconds=tensor_reconstruction_seconds,
            internal_force_seconds=internal_force_seconds,
            element_matrix_seconds=element_matrix_seconds,
            sparse_assembly_seconds=sparse_assembly_seconds,
            free_system_extraction_seconds=free_system_extraction_seconds,
            pardiso_seconds=linear_solve_seconds,
            pardiso_analysis_seconds=(linear_solver_statistics.analysis_seconds),
            pardiso_factorization_seconds=(linear_solver_statistics.factorization_seconds),
            pardiso_solve_seconds=linear_solver_statistics.solve_seconds,
            pardiso_analysis_calls=linear_solver_statistics.analysis_calls,
            pardiso_factorization_calls=(linear_solver_statistics.factorization_calls),
            pardiso_solve_calls=linear_solver_statistics.solve_calls,
            nonlocal_mfront_without_tangent_seconds=(nonlocal_mfront_without_tangent_seconds),
            nonlocal_mfront_with_tangent_seconds=(nonlocal_mfront_with_tangent_seconds),
            mfront_integration_without_tangent_calls=(mfront_integration_without_tangent_calls),
            mfront_integration_with_tangent_calls=(mfront_integration_with_tangent_calls),
            tensor_reconstruction_calls=tensor_reconstruction_calls,
        ),
    )


# ── Verification: equal biaxial tension (self-consistent BCs) ───────────────
def _verify():
    """
    Equal biaxial tension on 20x20 homogeneous mesh.
    BCs: u_x = eps*x, u_y = eps*y  where eps is computed analytically
    for a target sigma_vm = 400 MPa (above yield=250).
    Expected PEEQ = 0.00737, sigma_vm = 400 MPa.
    """
    print("=" * 55)
    print("VERIFICATION: equal biaxial tension, 20x20 mesh")
    print("=" * 55)
    nx, ny = 20, 20
    el = 0.001
    sf = 1.0
    E, nu = 205000.0, 0.3
    sy0, K, nexp = 250.0, 500.0, 0.245

    # Analytical solution for equal biaxial sigma_vm = sig_t
    sig_t = 400.0
    # ep from Ludwik: sig_t = sy0 + K*ep^n
    ep_ref = ((sig_t - sy0) / K) ** (1.0 / nexp)
    # For equal biaxial: ep_11=ep_22=ep/2, eeq=ep
    eps_e = sig_t * (1.0 - nu) / E
    eps_tot_val = eps_e + ep_ref / 2.0

    nxn, nyn = nx + 1, ny + 1
    xs = np.linspace(0, nx * el * sf, nxn)
    ys = np.linspace(0, ny * el * sf, nyn)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    disp_x = eps_tot_val * xx
    disp_y = eps_tot_val * yy

    result = run_fem(
        disp_x,
        disp_y,
        np.full((nx, ny), sy0),
        np.full((nx, ny), K),
        nexp,
        nx * el,
        ny * el,
        el,
        sf,
        E_mod=E,
        nu=nu,
        N_inc=30,
        max_nr=10,
        nr_tol=1e-8,
        constitutive_backend="python",
        verbose=True,
    )

    sv = von_mises(result["S"].reshape(-1, 3)).mean()
    ep = result["PEEQ"].mean()
    print(f"\n  sigma_vm  = {sv:.3f} MPa  (expected {sig_t:.1f})")
    print(f"  PEEQ      = {ep:.6f}     (expected {ep_ref:.6f})")
    err = abs(sv - sig_t) / sig_t * 100
    ep_err = abs(ep - ep_ref) / ep_ref * 100
    print(f"  Error     = {err:.3f} %")
    assert err < 0.5, f"FAILED: {err:.2f}%"
    assert ep_err < 0.5, f"FAILED PEEQ: {ep_err:.2f}%"
    print("  PASSED")


if __name__ == "__main__":
    _verify()
