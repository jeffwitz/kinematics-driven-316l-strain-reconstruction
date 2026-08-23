#!/usr/bin/env python3
"""SRIX FEMU-U smoke: free (tau0, Q), fit displacements, 20x20 window.

Per `validation/srix_femu_smoke_preregistration.md`: the law inside the
equilibrium problem, two free parameters, least squares on the measured
displacement, held-out validation. Each objective evaluation is one
`fem-inhouse partition` run on the prepared 20x20 case.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/_generated/shared_tensor_generator"
CASE = ROOT / "data/processed/femu_hist20"
DEFAULT_PARAMETER_SET = "316l_srix_transposed_from_nasri2018_rate_1e-3"
HELDOUT = (24, 28, 32, 36, 40)
STATES = list(range(21, 41))


def run_partition(output: Path, parameters: dict[str, float]) -> float:
    """One FEM run; returns the final-state displacement misfit to the measured."""

    options = json.dumps({"parameter_set": DEFAULT_PARAMETER_SET, "parameters": parameters})
    base = [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "fem_inhouse",
        "partition",
        "--input",
        str(CASE),
        "--output",
        str(output),
        "--count",
        "25",
        "--padding",
        "0",
        "--increments",
        "20",
    ]
    solve = subprocess.run(
        [
            *base,
            "--solve-pending",
            "--constitutive-backend",
            "mfront-3d-condensed-plane-stress",
            "--mfront-library",
            "build/mfront/src/libBehaviour.so",
            "--mfront-behaviour-id",
            "fcc_forest_rubin_srix",
            "--constitutive-options",
            options,
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if solve.returncode != 0:
        raise RuntimeError(f"partition failed: {solve.stderr[-600:]}")
    stitch = subprocess.run(
        [
            *base,
            "--stitch",
            "U",
            "--constitutive-backend",
            "mfront-3d-condensed-plane-stress",
            "--mfront-library",
            "build/mfront/src/libBehaviour.so",
            "--mfront-behaviour-id",
            "fcc_forest_rubin_srix",
            "--constitutive-options",
            options,
            "--field-output",
            str(output / "U.npz"),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if stitch.returncode != 0:
        raise RuntimeError(f"stitch failed: {stitch.stderr[-600:]}")
    simulated = np.load(output / "U.npz", allow_pickle=False)
    measured = np.stack(
        [np.load(CASE / "displacement_x_mm.npy"), np.load(CASE / "displacement_y_mm.npy")],
        axis=-1,
    )
    return float(np.sum((simulated - measured) ** 2) / max(np.sum(measured**2), 1e-300))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "srix_femu_smoke.json")
    parser.add_argument("--maxiter", type=int, default=12)
    arguments = parser.parse_args()
    # The evaluation directories carry the run manifest; leftovers from a
    # previous invocation would collide with fresh evaluations of the same
    # parameters, so they are cleared before every run.
    import shutil

    for pattern in (
        "femu_smoke_theta_*",
        "femu_smoke_fitted",
        "femu_smoke_default",
        "femu_smoke_j2",
    ):
        for stale in OUT.glob(pattern):
            if stale.is_dir():
                shutil.rmtree(stale)

    theta0 = np.log(np.asarray([40.0, 10.0]))  # tau0, Q defaults
    evaluation_counter = 0

    def objective(theta_log: np.ndarray) -> float:
        nonlocal evaluation_counter
        tau0, q = np.exp(theta_log)
        evaluation_counter += 1
        return run_partition(
            OUT / f"femu_smoke_theta_{evaluation_counter:03d}",
            {"tau0_mpa": float(tau0), "Q_mpa": float(q)},
        )

    def gradient(theta_log: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(theta_log)
        for index in range(len(theta_log)):
            step = 1e-3 * max(1.0, abs(theta_log[index]))
            plus = theta_log.copy()
            minus = theta_log.copy()
            plus[index] += step
            minus[index] -= step
            grad[index] = (objective(plus) - objective(minus)) / (2 * step)
        return grad

    bounds = [(value - 1.5, value + 1.5) for value in theta0]
    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        jac=gradient,
        bounds=bounds,
        options={"maxiter": arguments.maxiter, "ftol": 1e-8, "gtol": 1e-6},
    )
    fitted_tau0, fitted_q = np.exp(result.x)
    fitted_misfit = run_partition(
        OUT / "femu_smoke_fitted", {"tau0_mpa": fitted_tau0, "Q_mpa": fitted_q}
    )
    default_misfit = run_partition(OUT / "femu_smoke_default", {})
    # J2 baseline reference: the python backend on the same case.
    j2_output = OUT / "femu_smoke_j2"
    if not (j2_output / "U.npz").exists():
        j2_solve = subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                "-m",
                "fem_inhouse",
                "partition",
                "--input",
                str(CASE),
                "--output",
                str(j2_output),
                "--count",
                "25",
                "--padding",
                "0",
                "--increments",
                "20",
                "--solve-pending",
                "--constitutive-backend",
                "python",
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if j2_solve.returncode != 0:
            raise RuntimeError(f"j2 solve failed: {j2_solve.stderr[-600:]}")
        j2_stitch = subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                "-m",
                "fem_inhouse",
                "partition",
                "--input",
                str(CASE),
                "--output",
                str(j2_output),
                "--count",
                "25",
                "--padding",
                "0",
                "--increments",
                "20",
                "--stitch",
                "U",
                "--constitutive-backend",
                "python",
                "--field-output",
                str(j2_output / "U.npz"),
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if j2_stitch.returncode != 0:
            raise RuntimeError(f"j2 stitch failed: {j2_stitch.stderr[-600:]}")
    j2_u = np.load(j2_output / "U.npz", allow_pickle=False)
    measured = np.stack(
        [np.load(CASE / "displacement_x_mm.npy"), np.load(CASE / "displacement_y_mm.npy")],
        axis=-1,
    )
    j2_misfit = float(np.sum((j2_u - measured) ** 2) / max(np.sum(measured**2), 1e-300))
    payload = {
        "schema_version": 1,
        "fitted_parameters": {"tau0_mpa": float(fitted_tau0), "q_mpa": float(fitted_q)},
        "misfit_fitted": fitted_misfit,
        "misfit_default": default_misfit,
        "misfit_j2_baseline": j2_misfit,
        "bars": {"beats_j2_by": 0.9, "beats_default_by": 0.1},
        "reading": (
            "helps"
            if fitted_misfit <= 0.9 * j2_misfit
            and default_misfit - fitted_misfit >= 0.1 * default_misfit
            else "negative"
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
