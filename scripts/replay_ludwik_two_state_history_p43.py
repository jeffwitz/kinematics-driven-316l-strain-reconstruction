#!/usr/bin/env python3
"""What does Ludwik/J2 already reproduce of the measured P43 kinematics?

Every inverse run so far has explained `eps_DIC - eps_elastic`, which asks a
free eigenstrain field to invent an entire plastic history from nothing. The
question that was never asked is the cheaper and more informative one: how much
of that gap does the constitutive model we already have close on its own?

The comparison is exact rather than approximate, because the two-state solver
`solve_two_state_dirichlet_plane_stress` and the diagnostic used throughout the
inverse work share the very same kinematics, `TwoSubcellDiagnostic2D`. Measured
and simulated nodal fields therefore pass through one identical observation
operator; no interpolation, no state conversion, no mixing of integration
points. Nothing here compares an internal material field to a measured one.

Three fields per state, all observed the same way:

```text
eps_DIC     = B_obs u_DIC                     the measurement
eps_L       = B_obs u_Ludwik    Ludwik/J2, per-pixel yield and hardening maps
eps_el      = B_obs u_elastic   the same solver with plasticity switched off
```

and the number the project has lacked,

```text
E_L = |eps_L - eps_DIC| / |eps_el - eps_DIC|,        eta_L = 1 - E_L,
```

the fraction of the elastic defect that Ludwik already explains.

The plastic history is run from the undeformed state 0, not from state 20:
plasticity is path dependent, and starting mid-history would hand the model a
virgin material that the specimen no longer is. Increments relative to state 20
are reported as well, because that is the reference every earlier inverse run
used, but they are formed by differencing a single continuous trajectory.

The elastic reference needs no history -- it is path independent -- so it is
solved in one increment per state.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.core.kelvin import equivalent_plastic_strain, strain_from_engineering
from fem_inhouse.core.plane_stress_material import PythonJ2PlaneStressBatch
from fem_inhouse.spectral2d import EBISpectralSolverConfig, StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.spectral2d.newton_two_state import (
    TwoStateIncrementFields,
    solve_two_state_dirichlet_plane_stress,
)
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig

FloatArray = NDArray[np.float64]

ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "data/processed/case_study"
HISTORY_ROOT = ROOT / "validation/reference_data/dic_multistep_history_p0043_repaired_v1"
PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30
#: Ludwik exponent registered with the P43 yield-stress and hardening maps.
LUDWIK_EXPONENT = 0.245
#: A yield stress no attainable stress can reach, so the same code path is
#: elastic. Using the identical solver, element batch and kinematics is what
#: makes `eps_el` a fair denominator rather than a different discretisation.
ELASTIC_YIELD_STRESS_MPA = 1.0e12


def _relative(candidate: FloatArray, reference: FloatArray) -> float:
    return float(
        np.linalg.norm(candidate - reference) / max(float(np.linalg.norm(reference)), 1.0e-30)
    )


def _solver_config(
    *,
    tolerance: float,
    workers: int,
    linear_mode: str,
    reference_update: str,
    restart: int,
    maximum_newton_iterations: int,
) -> EBISpectralSolverConfig:
    return EBISpectralSolverConfig(
        relative_equilibrium_tolerance=tolerance,
        gmres_restart=restart,
        maximum_newton_iterations=maximum_newton_iterations,
        linear_tolerance_mode=linear_mode,  # type: ignore[arg-type]
        reference_update_mode=reference_update,  # type: ignore[arg-type]
        transform=SpectralTransformConfig(
            backend="fftw",
            workers=workers,
            fftw_planner_effort="measure",
            fftw_planning_time_limit_s=2.0,
            fftw_use_wisdom=False,
        ),
    )


def _load_history(origin: tuple[int, int], pixels: int) -> FloatArray:
    report = json.loads((HISTORY_ROOT / "report.json").read_text(encoding="utf-8"))
    bounds = list(map(int, report["solve_bounds"]))
    x0, y0 = origin
    source = np.load(HISTORY_ROOT / "repaired_history_mm.npy", mmap_mode="r", allow_pickle=False)
    history = np.asarray(
        source[
            :,
            x0 - bounds[0] : x0 + pixels - bounds[0] + 1,
            y0 - bounds[2] : y0 + pixels - bounds[2] + 1,
            :,
        ],
        dtype=np.float64,
    )
    if not np.allclose(history[0], 0.0):
        raise SystemExit("state 0 of the repaired history is not the undeformed reference")
    return history


def _load_ludwik_parameters(origin: tuple[int, int], pixels: int) -> tuple[FloatArray, FloatArray]:
    x0, y0 = origin
    yield_stress = np.load(CASE_ROOT / "yield_stress_mpa.npy", mmap_mode="r")
    hardening = np.load(CASE_ROOT / "hardening_coefficient_mpa.npy", mmap_mode="r")
    crop = (slice(x0, x0 + pixels), slice(y0, y0 + pixels))
    return (
        np.asarray(yield_stress[crop], dtype=np.float64).reshape(-1),
        np.asarray(hardening[crop], dtype=np.float64).reshape(-1),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", nargs=2, type=int, default=(1580, 1030))
    parser.add_argument("--pixels", type=int, default=100)
    parser.add_argument("--reference-state", type=int, default=20)
    parser.add_argument("--states", nargs="+", type=int, default=[25, 30, 35, 40])
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--workers", type=int, default=1)
    # The "optimized" policy of the TRI2 benchmark -- Eisenstat-Walker with a
    # per-Newton reference update -- was qualified on eight proportional
    # increments. On forty increments of the measured history it carries the
    # trajectory to increment 37 and then fails to converge, so the
    # conservative policy is the default here.
    parser.add_argument(
        "--linear-mode", choices=("fixed", "eisenstat_walker"), default="fixed"
    )
    parser.add_argument(
        "--reference-update",
        choices=("initial", "per_increment", "per_newton"),
        default="initial",
    )
    parser.add_argument("--restart", type=int, default=50)
    parser.add_argument("--max-newton", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    solver_config = _solver_config(
        tolerance=arguments.tolerance,
        workers=arguments.workers,
        linear_mode=arguments.linear_mode,
        reference_update=arguments.reference_update,
        restart=arguments.restart,
        maximum_newton_iterations=arguments.max_newton,
    )

    pixels = arguments.pixels
    origin = (int(arguments.origin[0]), int(arguments.origin[1]))
    observed_states = sorted({arguments.reference_state, *arguments.states})
    last_state = max(observed_states)

    grid = StructuredGrid2D(pixels, pixels, pixels * PIXEL_SIZE_MM, pixels * PIXEL_SIZE_MM)
    kinematics = TwoSubcellDiagnostic2D(grid)
    history = _load_history(origin, pixels)
    yield_stress, hardening = _load_ludwik_parameters(origin, pixels)

    def observe(displacement: FloatArray) -> FloatArray:
        """Nodal field to Kelvin strain, shape `(nx, ny, 2, 3)`.

        The one operator both the measurement and every simulation go through.
        """

        return strain_from_engineering(np.asarray(kinematics.strain(displacement)))

    # --- Ludwik, one continuous trajectory from the undeformed state ---------
    material = PythonJ2PlaneStressBatch(
        np.repeat(yield_stress, 2),
        np.repeat(hardening, 2),
        LUDWIK_EXPONENT,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
    )
    captured: dict[int, FloatArray] = {}
    captured_stress: dict[int, FloatArray] = {}
    captured_plastic: dict[int, FloatArray] = {}
    wanted = set(observed_states)

    def capture(fields: TwoStateIncrementFields) -> None:
        increment = fields.increment
        if increment in wanted:
            captured[increment] = np.array(fields.displacement, dtype=np.float64, copy=True)
            captured_stress[increment] = np.array(
                fields.stress_in_plane_mpa, dtype=np.float64, copy=True
            )
            if fields.plastic_strain_tensor is not None:
                captured_plastic[increment] = np.array(
                    fields.plastic_strain_tensor, dtype=np.float64, copy=True
                )
        print(f"  increment {increment:3d}/{last_state} done", flush=True)

    started = time.perf_counter()
    ludwik = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=material,
        boundary_displacement_history=history[: last_state + 1],
        config=solver_config,
        increment_observer=capture,
    )
    ludwik_seconds = time.perf_counter() - started
    captured[last_state] = np.asarray(ludwik.displacement, dtype=np.float64)
    missing = sorted(wanted - captured.keys())
    if missing:
        raise SystemExit(f"the solver never converged the requested states {missing}")
    print(f"Ludwik trajectory: {ludwik_seconds:.1f} s", flush=True)

    # --- Elastic reference, path independent, one increment per state -------
    elastic_displacement: dict[int, FloatArray] = {}
    for state in observed_states:
        elastic_material = PythonJ2PlaneStressBatch(
            np.full(2 * pixels * pixels, ELASTIC_YIELD_STRESS_MPA),
            np.repeat(hardening, 2),
            LUDWIK_EXPONENT,
            young_modulus_mpa=YOUNG_MPA,
            poisson_ratio=POISSON,
        )
        result = solve_two_state_dirichlet_plane_stress(
            grid=grid,
            material=elastic_material,
            boundary_displacement_history=np.stack([history[0], history[state]]),
            config=solver_config,
        )
        elastic_displacement[state] = np.asarray(result.displacement, dtype=np.float64)
        print(f"elastic state {state} solved", flush=True)

    # --- One observation operator, three fields, per state ------------------
    components = ("xx", "yy", "xy")
    measured_strain = {state: observe(history[state]) for state in observed_states}
    ludwik_strain = {state: observe(captured[state]) for state in observed_states}
    elastic_strain = {state: observe(elastic_displacement[state]) for state in observed_states}

    def compare(
        measured: FloatArray, ludwik_field: FloatArray, elastic_field: FloatArray
    ) -> dict[str, Any]:
        ludwik_gap = float(np.linalg.norm(ludwik_field - measured))
        elastic_gap = float(np.linalg.norm(elastic_field - measured))
        ratio = ludwik_gap / max(elastic_gap, 1.0e-30)
        return {
            "strain_norm_dic": float(np.linalg.norm(measured)),
            "ludwik_absolute_gap": ludwik_gap,
            "elastic_absolute_gap": elastic_gap,
            "ludwik_relative_error": ludwik_gap / max(float(np.linalg.norm(measured)), 1.0e-30),
            "elastic_relative_error": elastic_gap / max(float(np.linalg.norm(measured)), 1.0e-30),
            "E_L": ratio,
            "eta_L": 1.0 - ratio,
            "per_component": {
                name: {
                    "E_L": float(
                        np.linalg.norm(ludwik_field[..., index] - measured[..., index])
                        / max(
                            float(np.linalg.norm(elastic_field[..., index] - measured[..., index])),
                            1.0e-30,
                        )
                    ),
                    "ludwik_relative_error": _relative(
                        ludwik_field[..., index], measured[..., index]
                    ),
                    "elastic_relative_error": _relative(
                        elastic_field[..., index], measured[..., index]
                    ),
                }
                for index, name in enumerate(components)
            },
        }

    reference = arguments.reference_state
    per_state: dict[str, Any] = {}
    for state in arguments.states:
        absolute = compare(
            measured_strain[state], ludwik_strain[state], elastic_strain[state]
        )
        increment = compare(
            measured_strain[state] - measured_strain[reference],
            ludwik_strain[state] - ludwik_strain[reference],
            elastic_strain[state] - elastic_strain[reference],
        )
        nodal = {
            "ludwik_relative_error": _relative(captured[state], history[state]),
            "elastic_relative_error": _relative(elastic_displacement[state], history[state]),
            "ludwik_rms_mm": float(
                np.sqrt(np.mean((captured[state] - history[state]) ** 2))
            ),
            "elastic_rms_mm": float(
                np.sqrt(np.mean((elastic_displacement[state] - history[state]) ** 2))
            ),
            "dic_rms_mm": float(np.sqrt(np.mean(history[state] ** 2))),
        }
        per_state[str(state)] = {
            "absolute": absolute,
            f"increment_from_state_{reference}": increment,
            "nodal_displacement": nodal,
        }
        print(
            f"state {state}: E_L {absolute['E_L']:.4f} (absolute), "
            f"{increment['E_L']:.4f} (increment from {reference}), "
            f"nodal {nodal['ludwik_relative_error']:.4f} vs elastic "
            f"{nodal['elastic_relative_error']:.4f}",
            flush=True,
        )

    # --- Fields for the maps ------------------------------------------------
    field_path = arguments.output.with_suffix(".npz")
    fields: dict[str, FloatArray] = {}
    for state in observed_states:
        fields[f"dic_strain_{state}"] = measured_strain[state]
        fields[f"ludwik_strain_{state}"] = ludwik_strain[state]
        fields[f"elastic_strain_{state}"] = elastic_strain[state]
        fields[f"dic_displacement_{state}"] = history[state]
        fields[f"ludwik_displacement_{state}"] = captured[state]
        fields[f"elastic_displacement_{state}"] = elastic_displacement[state]
        # Incompressible equivalent of the *total* strain: a comparable scalar
        # for the maps, not a plastic measure -- the elastic part is not
        # incompressible, so this is a display convention, stated as one.
        if state in captured_stress:
            fields[f"ludwik_stress_{state}_mpa"] = captured_stress[state]
        if state in captured_plastic:
            fields[f"ludwik_plastic_strain_{state}"] = captured_plastic[state]
        fields[f"dic_equivalent_{state}"] = equivalent_plastic_strain(measured_strain[state])
        fields[f"ludwik_equivalent_{state}"] = equivalent_plastic_strain(ludwik_strain[state])
        fields[f"elastic_equivalent_{state}"] = equivalent_plastic_strain(elastic_strain[state])
    assert ludwik.plastic_strain_tensor is not None
    fields["ludwik_plastic_strain_final"] = np.asarray(ludwik.plastic_strain_tensor)
    fields["ludwik_equivalent_plastic_strain_final"] = np.asarray(
        ludwik.observables["equivalent_plastic_strain"]
    )
    fields["ludwik_stress_final_mpa"] = np.asarray(ludwik.stress_in_plane_mpa)
    fields["ludwik_reaction_forces_final"] = np.asarray(ludwik.reaction_forces)
    fields["yield_stress_mpa"] = yield_stress.reshape(pixels, pixels)
    fields["hardening_coefficient_mpa"] = hardening.reshape(pixels, pixels)
    field_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(field_path, **fields)

    report = {
        "schema_version": 1,
        "status": "completed_ludwik_two_state_replay_p43",
        "question": (
            "how much of the elastic defect in the measured kinematics does "
            "Ludwik/J2 already explain, both fields seen through the same "
            "TwoSubcellDiagnostic2D observation operator"
        ),
        "origin_nodes": list(origin),
        "mesh": [pixels, pixels],
        "pixel_size_mm": PIXEL_SIZE_MM,
        "reference_state": reference,
        "states": arguments.states,
        "increments_run": last_state,
        "ludwik": {
            "exponent": LUDWIK_EXPONENT,
            "young_modulus_mpa": YOUNG_MPA,
            "poisson_ratio": POISSON,
            "yield_stress_mpa": {
                "min": float(yield_stress.min()),
                "mean": float(yield_stress.mean()),
                "max": float(yield_stress.max()),
            },
            "hardening_coefficient_mpa": {
                "min": float(hardening.min()),
                "mean": float(hardening.mean()),
                "max": float(hardening.max()),
            },
        },
        "solver": {
            "kinematics": "TwoSubcellDiagnostic2D",
            "observation_operator": "TwoSubcellDiagnostic2D.strain, Kelvin coordinates",
            "tolerance": arguments.tolerance,
            "linear_tolerance_mode": arguments.linear_mode,
            "reference_update_mode": arguments.reference_update,
            "gmres_restart": arguments.restart,
            "elapsed_seconds_ludwik": ludwik_seconds,
            "newton_iterations": int(sum(ludwik.diagnostics.iterations_per_increment)),
            "final_residual": ludwik.diagnostics.verification_residual,
        },
        "per_state": per_state,
        "field_file": str(field_path),
        "cpu": platform.processor(),
    }
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\nwrote {arguments.output} and {field_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
