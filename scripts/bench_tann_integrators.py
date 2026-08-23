#!/usr/bin/env python3
"""Stiffness benchmark: RK4 (registered) vs implicit Euler vs Radau.

On the material states of the P43 increment 16 (the one the amended run
could not pass): the committed state after increment 16 and the trial
strain of increment 17, from the completed 25x25 smoke archive. Radau
IIA-5 (scipy) is the high-precision reference; the question is whether
the implicit one-step scheme lands on the same state with unconditional
stability where RK4 needs the slope limiter and explodes on scaled
excursions. sigma_ref = 200 MPa throughout -- the integrator adapts to
the law, never the reverse.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from fem_inhouse.constitutive.tann_fcc import TannFCCBatch, TannFCCConfig
from fem_inhouse.spectral2d import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D

SIGMA_REF = 200.0
PIXELS = 25
PIXEL_SIZE_MM = 1.84e-3
SCALES = (1.0, 2.0, 5.0, 10.0, 50.0)


def load_increment_16_states(npz_path: Path, point_count: int):
    archive = dict(np.load(npz_path, allow_pickle=False))
    grid = StructuredGrid2D(PIXELS, PIXELS, PIXEL_SIZE_MM * PIXELS, PIXEL_SIZE_MM * PIXELS)
    kinematics = TwoSubcellDiagnostic2D(grid)
    committed_state = archive["state_36_committed_state"]  # (P, 12, 3)
    strain_committed = kinematics.strain_samples(archive["state_36_u_sim"]).reshape(-1, 3)
    strain_trial = kinematics.strain_samples(archive["state_37_u_meas"]).reshape(-1, 3)
    increment = strain_trial - strain_committed
    norms = np.linalg.norm(increment, axis=1)
    order = np.argsort(-norms)
    chosen = np.concatenate(
        [
            order[: point_count // 2],
            order[np.linspace(0, len(order) - 1, point_count // 2, dtype=int)],
        ]
    )
    return (
        committed_state[chosen],
        strain_committed[chosen],
        increment[chosen],
        chosen,
    )


def radau_reference(
    batch: TannFCCBatch,
    systems_pt: np.ndarray,
    q0: np.ndarray,
    eps_n: np.ndarray,
    delta: np.ndarray,
) -> dict:
    """Per-point Radau IIA-5 (scipy) on the linear strain path."""

    from scipy.integrate import solve_ivp

    eps_n_t = torch.from_numpy(eps_n)
    delta_t = torch.from_numpy(delta)
    rate = float(np.linalg.norm(delta))
    systems = torch.from_numpy(systems_pt[None])

    def rhs(s: float, q: np.ndarray) -> np.ndarray:
        q_t = torch.from_numpy(q.reshape(1, 12, 3))
        strain = eps_n_t + s * delta_t
        gamma = q_t[..., 0:1]
        z = q_t[..., 1:]
        with torch.no_grad():
            force_gamma, _ = batch._forces(strain, gamma[..., 0], z, systems)
            _, flow = batch._mobility_flow(force_gamma, z, None)
        return (rate * flow).reshape(-1).numpy()

    solution = solve_ivp(rhs, (0.0, 1.0), q0.reshape(-1), method="Radau", rtol=1e-10, atol=1e-12)
    return {
        "q": solution.y[:, -1].reshape(12, 3),
        "rhs_calls": solution.nfev,
        "success": bool(solution.success),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=Path("/tmp/tann_smoke25_sr200.npz"))
    parser.add_argument("--points", type=int, default=24)
    arguments = parser.parse_args()

    q0_all, eps_n_all, delta_all, chosen = load_increment_16_states(arguments.npz, arguments.points)
    point_count = len(chosen)
    # The smoke archive carries no per-point systems; rebuild them from the
    # EBSD with the production builder so the benchmark runs on the real
    # geometry of the crop.
    import h5py

    from fem_inhouse.constitutive.tann_fcc_geometry import systems_from_bunge_node_grid

    ebsd_path = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")
    x0, y0 = 1580, 1030
    with h5py.File(ebsd_path, "r") as handle:
        angles = np.stack(
            [
                np.asarray(handle[f"/orientation/{name}"])[
                    x0 : x0 + PIXELS + 1, y0 : y0 + PIXELS + 1
                ]
                for name in ("phi1", "Phi", "phi2")
            ],
            axis=-1,
        )
        schmid = np.asarray(handle["/schmid/max_schmid_factor"])[
            x0 : x0 + PIXELS + 1, y0 : y0 + PIXELS + 1
        ]
    systems_full, _ = systems_from_bunge_node_grid(angles, max_schmid_factor=schmid)
    systems = systems_full[chosen]

    print(f"benchmark on {point_count} points of the increment-16 state, sigma_ref = {SIGMA_REF}")
    for scale in SCALES:
        print(f"--- trial increment scaled x{scale} ---")
        rk4_batch = TannFCCBatch(
            TannFCCConfig(sigma_ref_mpa=SIGMA_REF), point_count=point_count, systems_global=systems
        )
        ie_batch = TannFCCBatch(
            TannFCCConfig(sigma_ref_mpa=SIGMA_REF, integrator="implicit_euler"),
            point_count=point_count,
            systems_global=systems,
        )
        ie_batch.copy_weights_from(rk4_batch)
        rk4_batch.reset_committed(q0_all, eps_n_all)
        ie_batch.reset_committed(q0_all, eps_n_all)
        strain_trial = eps_n_all + scale * delta_all

        started = time.perf_counter()
        trial_rk4 = rk4_batch.evaluate(strain_trial, compute_tangent=False)
        rk4_seconds = time.perf_counter() - started

        started = time.perf_counter()
        trial_ie = ie_batch.evaluate(strain_trial, compute_tangent=False)
        ie_seconds = time.perf_counter() - started

        # Radau on the three most extreme points
        radau_q = []
        for index in range(min(3, point_count)):
            radau_q.append(
                radau_reference(
                    rk4_batch,
                    systems[index],
                    q0_all[index],
                    eps_n_all[index],
                    scale * delta_all[index],
                )
            )

        print(
            f"  rk4: finite={np.isfinite(trial_rk4.trial_state).all()} "
            f"state_max={np.abs(trial_rk4.trial_state).max():.3e} ({rk4_seconds:.2f}s)"
        )
        print(
            f"  implicit: finite={np.isfinite(trial_ie.trial_state).all()} "
            f"state_max={np.abs(trial_ie.trial_state).max():.3e} ({ie_seconds:.2f}s)"
        )
        for index, ref in enumerate(radau_q):
            if not ref["success"]:
                print(f"  radau point {index}: FAILED")
                continue
            q_rk4 = trial_rk4.trial_state[index]
            q_ie = trial_ie.trial_state[index]
            print(
                f"  radau point {index}: nfev={ref['rhs_calls']} "
                f"|rk4-radau|={np.linalg.norm(q_rk4 - ref['q']):.3e} "
                f"|ie-radau|={np.linalg.norm(q_ie - ref['q']):.3e} "
                f"|ie-rk4|={np.linalg.norm(q_ie - q_rk4):.3e}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
