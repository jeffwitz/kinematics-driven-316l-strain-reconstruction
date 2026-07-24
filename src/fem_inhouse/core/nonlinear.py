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

import numpy as np
from scipy.sparse.linalg import spsolve

from fem_inhouse.core.assembly import (
    assemble_stiffness,
    assembly_indices,
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
from fem_inhouse.core.mesh import StructuredMesh
from fem_inhouse.core.plane_stress_material import (
    ConstitutiveIntegrationError,
    ConstitutiveTrial,
    create_plane_stress_material_batch,
)

LOGGER = logging.getLogger(__name__)

# Optional fast multithreaded direct solver (MKL Pardiso).
#   pip install pypardiso
try:
    import pypardiso

    def _solve(A, b):
        return pypardiso.spsolve(A.tocsr(), b)

    _SOLVER_NAME = "pypardiso (MKL, multithreaded)"
except Exception:

    def _solve(A, b):
        return spsolve(A.tocsr(), b)

    _SOLVER_NAME = "scipy SuperLU (single-threaded; 'pip install pypardiso' for a large speed-up)"

GP_XI = GAUSS_POINTS
GP_W = GAUSS_WEIGHTS
N_GP = GAUSS_POINT_COUNT


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
    snapshot_fractions=None,
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
    constitutive_backend : 'python' uses the in-house return mapping; 'mfront'
                delegates stress, state update and consistent tangent to MGIS.
    """
    started_at = time.perf_counter()
    mesh = StructuredMesh(x_size, y_size, element_size, scale_factor)
    nx, ny = mesh.nx, mesh.ny
    nxn, nyn = nx + 1, ny + 1
    n_e = mesh.n_elems

    LOGGER.info(
        "nonlinear solve started",
        extra={
            "event": "nonlinear_solve_started",
            "elements": n_e,
            "free_dofs": len(mesh.dofs_free),
            "backend": _SOLVER_NAME,
        },
    )

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
    )

    C_ps = plane_stress_elasticity(E_mod, nu)
    operators = precompute_element(mesh, C_ps)
    Ke = operators.elastic_stiffness
    Bs = operators.strain_displacement
    dJs = operators.jacobian_determinants

    # DOF maps
    nd_all = mesh.node_ids
    ld = mesh.location_matrix()
    dof_I = mesh.dofs_free
    dof_B = mesh.dofs_bc
    initialization_seconds = time.perf_counter() - started_at

    # constant assembly index arrays (reused for every tangent assembly)
    assembly_started_at = time.perf_counter()
    _rc = assembly_indices(ld)

    # Elastic predictor matrix; PyPardiso reuses its factorization for each RHS.
    K_el = assemble_stiffness(mesh, Ke, ld, _rc)
    KII_el = K_el[dof_I][:, dof_I].tocsr()
    KIB_el = K_el[dof_I][:, dof_B].tocsr()
    elastic_assembly_seconds = time.perf_counter() - assembly_started_at
    linear_solve_seconds = 0.0
    constitutive_seconds = 0.0
    tangent_assembly_seconds = 0.0

    def timed_solve(matrix, right_hand_side):
        nonlocal linear_solve_seconds
        solve_started_at = time.perf_counter()
        solution = _solve(matrix, right_hand_side)
        linear_solve_seconds += time.perf_counter() - solve_started_at
        return solution

    def solve_el(b):
        return timed_solve(KII_el, b)

    # Prescribed displacements
    u_bc = np.zeros(mesh.n_dof)
    u_bc[2 * nd_all.ravel(order="F")] = disp_x.ravel(order="F")
    u_bc[2 * nd_all.ravel(order="F") + 1] = disp_y.ravel(order="F")

    # State
    u = np.zeros(mesh.n_dof)
    eps_p = np.zeros((n_e, N_GP, 3))
    ep_bar = np.zeros((n_e, N_GP))
    sig = np.zeros((n_e, N_GP, 3))
    eps_tot = np.zeros((n_e, N_GP, 3))
    accepted_constitutive_trial: ConstitutiveTrial | None = None

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
    snaps = {}
    pending = sorted(snapshot_fractions) if snapshot_fractions else []
    while t < 1.0 - 1e-12:
        dt = min(dt, 1.0 - t)
        if pending:  # land exactly on snapshot fractions
            dt = min(dt, max(pending[0] - t, 1e-12))
        inc += 1
        du_B = u_bc[dof_B] * dt
        u_save = u.copy()

        u[dof_B] += du_B
        # Elastic predictor (reused factorization)
        u[dof_I] += solve_el(-KIB_el @ du_B)

        KII = KII_el  # start with elastic tangent, replaced after 1st iter

        # Saved converged state (updated each NR iter before possible break)
        sf_acc = sig.copy()
        eps_p_acc = eps_p.copy()
        ep_new = ep_bar.copy()
        constitutive_trial_acc: ConstitutiveTrial | None = None
        converged = False

        for nrit in range(max_nr):
            total_newton_iterations += 1
            maximum_newton_iterations = max(maximum_newton_iterations, nrit + 1)
            u_e = u[ld]
            eps_tot = np.einsum("gak,ek->ega", Bs, u_e)
            if not np.isfinite(eps_tot).all():
                break
            # Constitutive trial from the last converged material state.
            constitutive_started_at = time.perf_counter()
            try:
                trial = material_batch.evaluate(
                    eps_tot.reshape(-1, 3),
                    time_increment=dt,
                    consistent_tangent=True,
                )
            except ConstitutiveIntegrationError as error:
                LOGGER.warning(
                    "constitutive trial failed",
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
            constitutive_seconds += time.perf_counter() - constitutive_started_at

            # Save state from this iteration (used if we break here)
            sf_acc = sf
            eps_p_acc = eps_p_trial
            constitutive_trial_acc = trial

            # Internal forces and residual
            R = internal_force(mesh, sf, Bs, dJs, ld)
            R_I = R[dof_I]
            res = float(np.linalg.norm(R_I))
            if not np.isfinite(res):
                break  # diverged -> cutback
            if nrit == 0:
                res0 = max(res, 1e-30)
            rel = res / res0
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
            pl_idx = np.arange(n_e * N_GP)
            plastic_tangents = material_tangents
            Ke_ep = element_tangent_stiffness(
                Ke,
                C_ps,
                plastic_tangents,
                pl_idx,
                Bs,
                dJs,
                element_count=n_e,
            )
            K_tang = assemble_stiffness(mesh, Ke_ep, ld, _rc)
            KII = K_tang[dof_I][:, dof_I].tocsr()
            tangent_assembly_seconds += time.perf_counter() - tangent_started_at

            du = timed_solve(KII, -R_I)
            if not np.isfinite(du).all():
                break  # singular tangent -> cutback
            u[dof_I] += du

        if not converged:
            material_batch.revert()
            u = u_save
            dt *= 0.5
            cutbacks += 1
            LOGGER.warning(
                "increment cutback",
                extra={
                    "event": "increment_cutback",
                    "increment": inc,
                    "next_step": dt,
                },
            )
            if dt < dt_min:
                raise RuntimeError(
                    f"run_fem: increment cutback below minimum ({dt:.2e}) - solution not converging"
                )
            continue

        if constitutive_trial_acc is None:
            raise RuntimeError("global Newton converged without a constitutive trial")
        material_batch.commit()
        accepted_constitutive_trial = constitutive_trial_acc
        eps_p = eps_p_acc
        ep_bar = ep_new
        sig = sf_acc
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
    F_all = internal_force(mesh, sig, Bs, dJs, ld)
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

    output_seconds = time.perf_counter() - output_started_at
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
        mesh=mesh,
        frames=snaps,
        diagnostics=dict(
            backend=f"{_SOLVER_NAME}; constitutive={material_batch.backend_name}",
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
            tensor_reconstruction_source=material_batch.completion_strategy,
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
