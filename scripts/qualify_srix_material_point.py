"""Material-point sensitivity of the SRIX law, section 11 of the specification.

Light on purpose: twenty-eight single-point runs, seconds each, no cluster and
no experimental data. It answers one question -- how much does the answer move
when the overstress modulus, the elasticity or the orientation changes -- and
deliberately does not identify anything. `(Q, b, C, d)` are held fixed
throughout, so nothing here can be read as a fit.

Writes the structured results section 17 asks for, into `--output`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from fem_inhouse.core.crystal_orientation import mgis_rotation_argument
from fem_inhouse.core.fcc_interaction_matrix import build_interaction_matrix, slip_systems
from fem_inhouse.core.srix_canonical import (
    kinematic_stored_energy,
    overstress_diagnostic,
    slip_dissipation_increments,
    uniaxial_001_plateau_stress,
)
from fem_inhouse.core.srix_parameters import (
    SRIX_PARAMETER_SETS,
    get_parameter_set,
    srix_provenance,
)

BEHAVIOUR = "Fcc316LForestRubinSrix"
ELASTIC_STRAIN = slice(0, 6)
PLASTIC_SLIP = slice(6, 18)
EQUIVALENT_SLIP = slice(18, 30)
BACK_STRAIN = slice(30, 42)

#: Crystal directions the specification names.
AXES: dict[str, tuple[float, float, float]] = {
    "001": (0.0, 0.0, 1.0),
    "011": (0.0, 1.0, 1.0),
    "111": (1.0, 1.0, 1.0),
    "123": (1.0, 2.0, 3.0),
}

AXIAL_STRAIN = 0.02
DEFAULT_STEPS = 100


def _rotation_for_axis(axis: tuple[float, float, float]) -> np.ndarray:
    third = np.asarray(axis, dtype=float)
    third = third / np.linalg.norm(third)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, third))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    first = np.cross(seed, third)
    first /= np.linalg.norm(first)
    return np.vstack([first, np.cross(third, first), third])


def _schmid_tensors() -> np.ndarray:
    tensors = []
    for system in slip_systems():
        burgers = system.burgers / np.linalg.norm(system.burgers)
        normal = system.normal / np.linalg.norm(system.normal)
        tensors.append(0.5 * (np.outer(burgers, normal) + np.outer(normal, burgers)))
    return np.array(tensors)


SCHMID = _schmid_tensors()
ROOT_TWO = math.sqrt(2.0)


def _kelvin_to_tensor(values: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [values[0], values[3] / ROOT_TWO, values[4] / ROOT_TWO],
            [values[3] / ROOT_TWO, values[1], values[5] / ROOT_TWO],
            [values[4] / ROOT_TWO, values[5] / ROOT_TWO, values[2]],
        ]
    )


def run_point(
    *,
    library: str,
    parameter_set: str,
    axis: str,
    steps: int = DEFAULT_STEPS,
    axial: float = AXIAL_STRAIN,
) -> dict[str, Any]:
    """One uniaxial tension, recording everything section 11 lists."""

    import mgis.behaviour as mgis

    preset = get_parameter_set(parameter_set)
    behaviour = mgis.load(library, BEHAVIOUR, mgis.Hypothesis.Tridimensional)
    # Written on every run: MGIS shares behaviour handles process-wide, so a
    # previous run's set would otherwise still be in force.
    for name, value in preset.mfront_overrides().items():
        mgis.setParameter(behaviour, name, value)
    data = mgis.MaterialDataManager(behaviour, 1)
    for state in (data.s0, data.s1):
        mgis.setExternalStateVariable(state, "Temperature", preset.temperature_k)

    rotation = _rotation_for_axis(AXES[axis])
    argument = mgis_rotation_argument(rotation[None, :, :])
    hardening = build_interaction_matrix(preset.interaction_matrix)

    curve: list[tuple[float, float]] = []
    diagnostics: list[dict[str, Any]] = []
    dissipation = 0.0
    for index in range(steps):
        value = axial * (index + 1) / steps
        strain = np.zeros(6)
        strain[2] = value
        strain[0] = strain[1] = -0.5 * value
        crystal = np.ascontiguousarray(strain.copy())
        mgis.rotateGradients(crystal, behaviour, argument)
        previous = data.s1.internal_state_variables[0, PLASTIC_SLIP].copy()
        data.s1.gradients[:, :] = crystal.reshape(1, 6)
        mgis.integrate(
            data, mgis.IntegrationType.IntegrationWithConsistentTangentOperator, 1.0, 0, 1
        )
        mgis.update(data)

        internal = data.s1.internal_state_variables[0]
        crystal_stress = _kelvin_to_tensor(
            np.asarray(data.s1.thermodynamic_forces[0], dtype=float)
        )
        resolved = np.einsum("ij,sij->s", crystal_stress, SCHMID)
        back = preset.c_mpa * internal[BACK_STRAIN]
        equivalent = internal[EQUIVALENT_SLIP]
        resistance = preset.tau0_mpa + preset.q_mpa * (
            hardening @ (1.0 - np.exp(-preset.b * equivalent))
        )
        increment = internal[PLASTIC_SLIP] - previous
        dissipation += float(
            slip_dissipation_increments(
                resolved_stress_mpa=resolved,
                back_stress_mpa=back,
                slip_increment=increment,
            ).sum()
        )
        global_stress = np.ascontiguousarray(
            np.asarray(data.s1.thermodynamic_forces[0], dtype=float).copy()
        )
        mgis.rotateThermodynamicForces(global_stress, behaviour, argument)
        axial_stress = float(
            global_stress[2] - 0.5 * (global_stress[0] + global_stress[1])
        )
        curve.append((value, axial_stress))
        diagnostics.append(
            {
                "increment": index + 1,
                "axial_strain": value,
                **overstress_diagnostic(
                    resolved_stress_mpa=resolved,
                    back_stress_mpa=back,
                    critical_resistance_mpa=resistance,
                ).as_dict(),
            }
        )

    internal = data.s1.internal_state_variables[0]
    elastic = internal[ELASTIC_STRAIN]
    elastic_tensor = _kelvin_to_tensor(elastic)
    stress_tensor = _kelvin_to_tensor(
        np.asarray(data.s1.thermodynamic_forces[0], dtype=float)
    )
    equivalent = internal[EQUIVALENT_SLIP]
    resistance = preset.tau0_mpa + preset.q_mpa * (
        hardening @ (1.0 - np.exp(-preset.b * equivalent))
    )
    stored_isotropic = float(
        preset.q_mpa * np.sum(equivalent + (np.exp(-preset.b * equivalent) - 1.0) / preset.b)
    )
    return {
        "parameter_set": parameter_set,
        "axis": axis,
        "steps": steps,
        "overstress_modulus_mpa": preset.overstress_modulus_mpa,
        "overstress_ratio": preset.overstress_ratio,
        "c11_mpa": preset.elasticity.c11_mpa,
        "tau0_mpa": preset.tau0_mpa,
        "final_axial_stress_mpa": curve[-1][1],
        "curve": curve,
        "slip": internal[PLASTIC_SLIP].tolist(),
        "equivalent_slip": equivalent.tolist(),
        "cumulated_slip": float(np.sum(equivalent)),
        "back_stress_mpa": (preset.c_mpa * internal[BACK_STRAIN]).tolist(),
        "critical_resistance_mpa": resistance.tolist(),
        "active_systems": int(np.sum(np.abs(internal[PLASTIC_SLIP]) > 1e-12)),
        "plastic_dissipation": dissipation,
        "elastic_energy": float(0.5 * np.sum(stress_tensor * elastic_tensor)),
        "stored_isotropic": stored_isotropic,
        "stored_kinematic": kinematic_stored_energy(
            back_strain=internal[BACK_STRAIN], c_mpa=preset.c_mpa
        ),
        "diagnostics": diagnostics,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("validation/_generated/srix"))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args()

    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        raise SystemExit("MFRONT_BEHAVIOUR_LIBRARY must be set")
    args.output.mkdir(parents=True, exist_ok=True)

    results = [
        run_point(library=library, parameter_set=name, axis=axis, steps=args.steps)
        for name in SRIX_PARAMETER_SETS
        for axis in AXES
    ]

    _write_csv(
        args.output / "srix_r_sensitivity.csv",
        results,
        [
            "parameter_set",
            "axis",
            "overstress_modulus_mpa",
            "overstress_ratio",
            "tau0_mpa",
            "c11_mpa",
            "final_axial_stress_mpa",
            "cumulated_slip",
            "active_systems",
            "plastic_dissipation",
        ],
    )
    _write_csv(
        args.output / "srix_orientation_sensitivity.csv",
        results,
        [
            "axis",
            "parameter_set",
            "final_axial_stress_mpa",
            "cumulated_slip",
            "active_systems",
        ],
    )
    _write_csv(
        args.output / "srix_dissipation_balance.csv",
        results,
        [
            "parameter_set",
            "axis",
            "elastic_energy",
            "stored_isotropic",
            "stored_kinematic",
            "plastic_dissipation",
        ],
    )

    overstress_rows = [
        {"parameter_set": item["parameter_set"], "axis": item["axis"], **record}
        for item in results
        for record in item["diagnostics"]
        if record["increment"] % max(args.steps // 10, 1) == 0
    ]
    _write_csv(
        args.output / "srix_overstress_diagnostics.csv",
        overstress_rows,
        [
            "parameter_set",
            "axis",
            "increment",
            "axial_strain",
            "maximum",
            "q99",
            "q95",
            "mean_active",
            "fraction_above_1pc",
            "fraction_above_5pc",
            "fraction_above_10pc",
            "active_count",
        ],
    )

    convergence: list[dict[str, Any]] = []
    reference_steps = 400
    for axis in AXES:
        finest = run_point(
            library=library,
            parameter_set="316l_srix_transposed_from_nasri2018_rate_1e-3",
            axis=axis,
            steps=reference_steps,
        )["final_axial_stress_mpa"]
        for steps in (10, 20, 40, 80, 160):
            value = run_point(
                library=library,
                parameter_set="316l_srix_transposed_from_nasri2018_rate_1e-3",
                axis=axis,
                steps=steps,
            )["final_axial_stress_mpa"]
            convergence.append(
                {
                    "axis": axis,
                    "steps": steps,
                    "final_axial_stress_mpa": value,
                    "reference_steps": reference_steps,
                    "relative_error": abs(value - finest) / abs(finest),
                }
            )
    _write_csv(
        args.output / "srix_time_step_convergence.csv",
        convergence,
        ["axis", "steps", "final_axial_stress_mpa", "reference_steps", "relative_error"],
    )

    (args.output / "srix_parameter_provenance.json").write_text(
        json.dumps(
            {
                name: srix_provenance(
                    parameter_set=name,
                    mfront_source=Path("mfront") / f"{BEHAVIOUR}.mfront",
                )
                for name in SRIX_PARAMETER_SETS
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    historical = get_parameter_set("316l_srix_transposed_from_nasri2018_rate_1e-3")
    (args.output / "srix_canonical_tests.json").write_text(
        json.dumps(
            {
                "closed_form_001_plateau_no_hardening": {
                    "expression": "sigma = sqrt(6) tau0 + (6/8) R",
                    "tau0_mpa": historical.tau0_mpa,
                    "R_mpa": historical.overstress_modulus_mpa,
                    "value_mpa": uniaxial_001_plateau_stress(
                        tau0_mpa=historical.tau0_mpa,
                        overstress_modulus_mpa=historical.overstress_modulus_mpa,
                    ),
                    "active_systems": 8,
                },
                "runs": [
                    {
                        key: item[key]
                        for key in (
                            "parameter_set",
                            "axis",
                            "final_axial_stress_mpa",
                            "cumulated_slip",
                            "active_systems",
                            "plastic_dissipation",
                            "elastic_energy",
                            "stored_isotropic",
                            "stored_kinematic",
                        )
                    }
                    for item in results
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(f"wrote seven result files to {args.output}")
    for item in results:
        if item["parameter_set"].endswith("nasri2018_rate_1e-3"):
            print(
                f"  {item['axis']:<5} sigma={item['final_axial_stress_mpa']:9.3f} MPa  "
                f"slip={item['cumulated_slip']:.5f}  active={item['active_systems']:2d}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
