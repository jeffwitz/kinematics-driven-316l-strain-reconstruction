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
    internal_force,
)
from fem_inhouse.core.constitutive import (
    PLANE_STRESS_VON_MISES_METRIC,
    consistent_tangent,
    make_hardening,
    return_mapping,
    von_mises,
)
from fem_inhouse.core.element import (
    GAUSS_POINT_COUNT,
    GAUSS_POINTS,
    GAUSS_WEIGHTS,
    plane_stress_elasticity,
    precompute_element,
)
from fem_inhouse.core.mesh import StructuredMesh

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
    """
    t0 = time.time()
    hf, hfp = make_hardening(
        n_exp,
        hardening,
        ep_table_max,
        n_table,
        first_positive_plastic_strain,
    )
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
        return a.reshape(nx, ny, a.shape[1], order="F")

    # Material per GP
    sy0_gp = np.repeat(yield_map.ravel(order="F"), N_GP)
    K_gp = np.repeat(K_map.ravel(order="F"), N_GP)

    C_ps = plane_stress_elasticity(E_mod, nu)
    CM = C_ps @ PLANE_STRESS_VON_MISES_METRIC
    cm11, cm12, cm33 = CM[0, 0], CM[0, 1], CM[2, 2]
    operators = precompute_element(mesh, C_ps)
    Ke = operators.elastic_stiffness
    Bs = operators.strain_displacement
    dJs = operators.jacobian_determinants

    # DOF maps
    nd_all = mesh.node_ids
    ld = mesh.location_matrix()
    dof_I = mesh.dofs_free
    dof_B = mesh.dofs_bc

    # constant assembly index arrays (reused for every tangent assembly)
    _rc = assembly_indices(ld)

    # Elastic predictor matrix; PyPardiso reuses its factorization for each RHS.
    K_el = assemble_stiffness(mesh, Ke, ld, _rc)
    KII_el = K_el[dof_I][:, dof_I].tocsr()
    KIB_el = K_el[dof_I][:, dof_B].tocsr()

    def solve_el(b):
        return _solve(KII_el, b)

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
        dp_acc = np.zeros_like(eps_p)
        ep_new = ep_bar.copy()
        converged = False

        for nrit in range(max_nr):
            total_newton_iterations += 1
            maximum_newton_iterations = max(maximum_newton_iterations, nrit + 1)
            u_e = u[ld]
            eps_tot = np.einsum("gak,ek->ega", Bs, u_e)
            sig_tr = np.einsum("ij,egj->egi", C_ps, eps_tot - eps_p)

            # Return mapping
            sf, dp, dg = return_mapping(
                sig_tr.reshape(-1, 3), ep_bar.ravel(), sy0_gp, K_gp, hf, cm11, cm12, cm33
            )
            sf = sf.reshape(n_e, N_GP, 3)
            dp = dp.reshape(n_e, N_GP, 3)
            dg_ = dg.reshape(n_e, N_GP)
            ep_new = ep_bar + dg_

            # Save state from this iteration (used if we break here)
            sf_acc = sf
            dp_acc = dp

            # Internal forces and residual
            R = internal_force(mesh, sf, Bs, dJs, ld)
            R_I = R[dof_I]
            res = np.linalg.norm(R_I)
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
            N_flat = n_e * N_GP
            C_ep_flat = np.broadcast_to(C_ps, (N_flat, 3, 3)).copy()
            pl_idx = np.where(dg > 0)[0]
            if len(pl_idx):
                C_ep_flat[pl_idx] = consistent_tangent(
                    sf.reshape(-1, 3)[pl_idx],
                    dg[pl_idx],
                    ep_bar.ravel()[pl_idx],
                    sy0_gp[pl_idx],
                    K_gp[pl_idx],
                    hf,
                    hfp,
                    C_ps,
                    cm11,
                    cm12,
                    cm33,
                )
            C_ep_gp = C_ep_flat.reshape(n_e, N_GP, 3, 3)

            # Vectorised element tangent stiffness
            CB = np.einsum("egij,gjk->egik", C_ep_gp, Bs)
            Ke_ep = np.einsum("g,g,gik,egil->ekl", GP_W, dJs, Bs, CB)
            K_tang = assemble_stiffness(mesh, Ke_ep, ld, _rc)
            KII = K_tang[dof_I][:, dof_I].tocsr()

            du = _solve(KII, -R_I)
            if not np.isfinite(du).all():
                break  # singular tangent -> cutback
            u[dof_I] += du

        if not converged:
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

        eps_p += dp_acc
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
    F_all = internal_force(mesh, sig, Bs, dJs, ld)
    bc_m = np.zeros(mesh.n_dof, dtype=bool)
    bc_m[dof_B] = True

    U = np.zeros((nxn, nyn, 2))
    U[..., 0] = u[2 * nd_all].reshape(nxn, nyn)
    U[..., 1] = u[2 * nd_all + 1].reshape(nxn, nyn)
    RF = np.zeros((nxn, nyn, 2))
    RF[..., 0] = np.where(bc_m[2 * nd_all], F_all[2 * nd_all], 0.0).reshape(nxn, nyn)
    RF[..., 1] = np.where(bc_m[2 * nd_all + 1], F_all[2 * nd_all + 1], 0.0).reshape(nxn, nyn)

    elapsed_seconds = time.time() - t0
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
        S=tg(gm(sig)),
        E=tg(gm(eps_tot)),
        PE=tg(gm(eps_p)),
        PEEQ=tg(gm(ep_bar)),
        RF=RF,
        mesh=mesh,
        frames=snaps,
        diagnostics=dict(
            backend=_SOLVER_NAME,
            elapsed_seconds=elapsed_seconds,
            attempted_increments=inc,
            converged_increments=converged_increments,
            cutbacks=cutbacks,
            total_newton_iterations=total_newton_iterations,
            maximum_newton_iterations=maximum_newton_iterations,
            final_residual_norm=final_residual_norm,
            final_relative_residual=final_relative_residual,
            final_convergence_criterion=final_convergence_criterion,
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
