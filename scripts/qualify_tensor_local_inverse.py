#!/usr/bin/env python3
"""Milestone 3: the free tensor plastic increment, qualified before any network.

```text
a  ->  v(x) = sum_j w_j(x) a_j  ->  P_H  ->  A  ->  J(u; u_obs)  ->  grad_a J
```

Milestone 2 qualified the inverse plumbing with a **scalar** `Delta p` whose
direction J2 dictated. That family is too rigid for the real DIC, so the
representation changes and the plumbing is reused: three Kelvin coefficients per
patch node, admissibility applied to the **assembled** field, and the mechanics
supplied by the already-qualified matrix-free tensor operator rather than the
scalar driven-J2 batch.

Gates are registered in `validation/tensor_local_inverse_preregistration.md`
before any run. Gate 7 is the one that matters scientifically: the same twin
fitted by the free tensor family and by the `Delta p + n_J2` family through
identical mechanics and an identical objective, so the only difference is the
freedom of the representation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from fem_inhouse.identification.tensor_local_inverse import (
    DissipativeProjection,
    TensorLocalBasis,
    TensorLocalInverse,
    j2_flow_direction,
    plastic_gauge_norm,
)
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D

PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30
UNIAXIAL_STRESS_MPA = 320.0


class _Identity:
    """No measurement chain: this milestone is about the representation."""

    def apply(self, values):
        return np.asarray(values, dtype=np.float64)

    def adjoint(self, values):
        return np.asarray(values, dtype=np.float64)


def build(pixels: int, patches: int):
    grid = StructuredGrid2D(
        pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels
    )
    operator = TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
        transfer=_Identity(),
        whitener=_Identity(),
    )
    basis = TensorLocalBasis.build(grid.nx, grid.ny, patches)
    return grid, operator, basis


def frozen_stress(operator, seed: int) -> np.ndarray:
    """`sigma_pred`, fixed before the run and independent of `a`.

    A uniform uniaxial predictor carries the loading; the elastic response to an
    **independently seeded** smooth eigenstrain gives it spatial structure, so
    the half-space varies across the domain without leaking the field being
    recovered.
    """

    points = operator.kinematics.material_point_count
    base = np.zeros((points, 3))
    base[:, 1] = UNIAXIAL_STRESS_MPA

    generator = np.random.default_rng(seed)
    grid = operator.grid
    coarse = generator.standard_normal((4, 4, 3))
    x = np.linspace(0.0, 3.0, grid.nx)
    y = np.linspace(0.0, 3.0, grid.ny)
    nodes = np.arange(4, dtype=np.float64)
    wx = np.clip(1.0 - np.abs(x[:, None] - nodes[None, :]), 0.0, None)
    wy = np.clip(1.0 - np.abs(y[:, None] - nodes[None, :]), 0.0, None)
    wx /= wx.sum(axis=1, keepdims=True)
    wy /= wy.sum(axis=1, keepdims=True)
    field = np.einsum("xi,yj,ijc->xyc", wx, wy, coarse)
    subcells = points // (grid.nx * grid.ny)
    reference = 3.0e-4 * np.repeat(
        field[:, :, None, :], subcells, axis=2
    ).reshape(-1, 3)
    strain = np.asarray(operator.kelvin_response(reference)).reshape(-1, 3)
    variation = np.einsum("pi,pij->pj", strain - reference, operator.elasticity)
    return base + variation


def truth_coefficients(basis: TensorLocalBasis, peak: float, seed: int) -> np.ndarray:
    """Smooth, three independent components, and deliberately not J2-parallel.

    Under a uniaxial `sigma_yy` the J2 direction is exactly `e_xx = -0.5 e_yy`
    with zero shear, so a truth that varies that ratio and carries real shear is
    what makes the family comparison of gate 7 meaningful. A truth inside the J2
    cone would let the restricted family match it and prove nothing.
    """

    generator = np.random.default_rng(seed)
    patches = basis.coefficient_shape[0]
    values = np.zeros(basis.coefficient_shape)
    axial = np.abs(generator.standard_normal((patches, patches))) + 0.4
    values[:, :, 1] = axial
    values[:, :, 0] = -axial * (0.5 + 0.45 * generator.standard_normal((patches, patches)))
    values[:, :, 2] = 0.6 * axial * generator.standard_normal((patches, patches))
    return values * (peak / float(np.abs(values[:, :, 1]).max()))


# -- gate 1 ---------------------------------------------------------------


def gate_transpose(arguments) -> dict:
    _, operator, _ = build(arguments.small_pixels, 4)
    generator = np.random.default_rng(21)
    worst = 0.0
    for _ in range(4):
        x = generator.standard_normal(operator.plastic_size)
        y = generator.standard_normal(operator.observation_size)
        left = float(np.dot(operator.matvec(x), y))
        right = float(np.dot(x, operator.rmatvec(y)))
        worst = max(worst, abs(left - right) / max(abs(left), abs(right), 1e-300))
    print(f"  gate 1  A/A^T dot product, worst relative: {worst:.3e}", flush=True)
    return {"worst_relative": worst, "passed": worst <= 1e-10}


# -- gate 2 ---------------------------------------------------------------


def gate_gradient(arguments) -> dict:
    grid, operator, basis = build(arguments.small_pixels, arguments.small_patches)
    stress = frozen_stress(operator, 91)
    projection = DissipativeProjection(stress=stress)
    truth = truth_coefficients(basis, arguments.peak, 5)
    inverse = TensorLocalInverse(
        operator=operator, basis=basis, projection=projection,
        observed_displacement=np.zeros((*grid.node_shape, 2)),
    )
    target = inverse.evaluate(truth)
    inverse.observed = target.displacement.copy()

    start = truth * 0.55
    base_evaluation, gradient = inverse.gradient(start)
    _, base_active, _ = inverse.plastic_from(start)
    generator = np.random.default_rng(13)
    per_direction = []
    for index in range(2):
        direction = generator.standard_normal(basis.coefficient_shape)
        direction = direction / np.linalg.norm(direction) * np.linalg.norm(start)
        analytic = float(np.sum(gradient * direction))
        sweep = []
        for exponent in range(2, 9):
            step = 10.0 ** (-exponent)
            plus = inverse.evaluate(start + step * direction)
            minus = inverse.evaluate(start - step * direction)
            # P_H is piecewise linear. A step that moves a point across the kink
            # makes the central difference measure a different function than the
            # gradient does, so the crossing is detected rather than averaged in.
            _, active_plus, _ = inverse.plastic_from(start + step * direction)
            _, active_minus, _ = inverse.plastic_from(start - step * direction)
            crossed = int(
                np.sum(active_plus != base_active) + np.sum(active_minus != base_active)
            )
            difference = (plus.objective - minus.objective) / (2.0 * step)
            relative = abs(difference - analytic) / max(abs(analytic), 1e-300)
            sweep.append({"step": step, "relative_error": relative,
                          "activity_crossings": crossed})
            print(f"  gate 2  dir {index} h=1e-{exponent}  fd={difference:+.8e}  "
                  f"adj={analytic:+.8e}  rel={relative:.3e}  crossings={crossed}",
                  flush=True)
        clean = [row for row in sweep if row["activity_crossings"] == 0]
        errors = [row["relative_error"] for row in clean]
        best = min(errors) if errors else float("inf")
        position = errors.index(best) if errors else -1
        per_direction.append({
            "analytic": analytic, "sweep": sweep, "best_relative_error": best,
            "v_shaped": 0 < position < len(errors) - 1,
        })
    worst = max(d["best_relative_error"] for d in per_direction)
    shaped = all(d["v_shaped"] for d in per_direction)
    print(f"  gate 2  worst {worst:.3e}, V-shaped {shaped}, "
          f"projection active at base {base_evaluation.active_fraction:.3f}", flush=True)
    return {"directions": per_direction, "worst_best_relative_error": worst,
            "v_shaped": shaped, "base_active_fraction": base_evaluation.active_fraction,
            "passed": worst <= 1e-5 and shaped}


# -- gate 3 ---------------------------------------------------------------


def gate_spectrum(arguments) -> dict:
    grid, operator, basis = build(arguments.spectrum_pixels, arguments.patches)
    stress = frozen_stress(operator, 91)
    projection = DissipativeProjection(stress=stress)
    truth = truth_coefficients(basis, arguments.peak, 5)
    inverse = TensorLocalInverse(
        operator=operator, basis=basis, projection=projection,
        observed_displacement=np.zeros((*grid.node_shape, 2)),
    )
    started = time.time()
    count = basis.coefficient_count
    columns = np.empty((int(np.prod(grid.node_shape)) * 2, count))
    seed = np.zeros(count)
    for index in range(count):
        seed[:] = 0.0
        seed[index] = 1.0
        # Applied exactly, not differenced: a quotient would be exact away from
        # the projection's kinks and wrong across one, and there is no reason to
        # accept that ambiguity when the Jacobian is available.
        columns[:, index] = inverse.sensitivity_column(
            truth, seed.reshape(basis.coefficient_shape)
        ).ravel()
    singular = np.linalg.svd(columns, compute_uv=False)
    thresholds = {
        f"above_1e-{d}": int(np.sum(singular >= singular[0] * 10.0**-d))
        for d in (1, 2, 3, 4, 6)
    }
    fraction = thresholds["above_1e-6"] / count
    print(f"  gate 3  {count} coefficients, condition "
          f"{singular[0] / max(singular[-1], 1e-300):.3e}", flush=True)
    print(f"    above 1e-1 / 1e-2 / 1e-3 / 1e-4 / 1e-6 of the leading: "
          f"{thresholds['above_1e-1']} / {thresholds['above_1e-2']} / "
          f"{thresholds['above_1e-3']} / {thresholds['above_1e-4']} / "
          f"{thresholds['above_1e-6']}", flush=True)
    return {
        "coefficients": count,
        "condition_number": float(singular[0] / max(singular[-1], 1e-300)),
        "effective_rank": thresholds,
        "fraction_above_1e-6": fraction,
        "singular_values": singular.tolist(),
        "seconds": time.time() - started,
        "passed": fraction >= 0.90,
    }


# -- gates 4 to 7 ---------------------------------------------------------


def recover(inverse, basis, truth_field, peak, iterations, label):
    start = np.zeros(basis.coefficient_shape)
    start[:, :, 1] = 0.3 * peak
    start[:, :, 0] = -0.15 * peak
    truth_norm = max(plastic_gauge_norm(truth_field), 1e-300)
    scale = max(0.5 * float(np.sum(inverse.observed**2)), 1e-300)
    history: list[dict[str, float]] = []

    def objective_and_gradient(flat):
        values = flat.reshape(basis.coefficient_shape) * peak
        evaluation, gradient = inverse.gradient(values)
        error = plastic_gauge_norm(evaluation.plastic_field - truth_field) / truth_norm
        history.append({
            "objective": evaluation.objective,
            "gauge_relative_error": error,
            "active_fraction": evaluation.active_fraction,
            "minimum_dissipation": evaluation.minimum_dissipation,
        })
        if len(history) % 25 == 1:
            print(f"    {label} {len(history):>4}  J={evaluation.objective:.6e}  "
                  f"gauge={error:.5f}  active={evaluation.active_fraction:.3f}",
                  flush=True)
        return evaluation.objective / scale, (gradient * peak / scale).ravel()

    started = time.time()
    outcome = minimize(
        objective_and_gradient, (start / peak).ravel(), jac=True, method="L-BFGS-B",
        options={"maxiter": iterations, "ftol": 1e-18, "gtol": 1e-16},
    )
    final = inverse.evaluate(outcome.x.reshape(basis.coefficient_shape) * peak)
    error = plastic_gauge_norm(final.plastic_field - truth_field) / truth_norm
    decades = (
        float(np.log10(history[0]["objective"] / final.objective))
        if final.objective > 0 else float("inf")
    )
    return {
        "evaluations": len(history),
        "initial_objective": history[0]["objective"],
        "final_objective": final.objective,
        "decades": decades,
        "gauge_relative_error": error,
        "final_active_fraction": final.active_fraction,
        "maximum_active_fraction": max(row["active_fraction"] for row in history),
        "minimum_dissipation": min(row["minimum_dissipation"] for row in history),
        "seconds": time.time() - started,
        "history": history,
    }


def gate_twin(arguments) -> dict:
    grid, operator, basis = build(arguments.twin_pixels, arguments.patches)
    stress = frozen_stress(operator, 91)
    projection = DissipativeProjection(stress=stress)
    truth = truth_coefficients(basis, arguments.peak, 5)

    reference = TensorLocalInverse(
        operator=operator, basis=basis, projection=projection,
        observed_displacement=np.zeros((*grid.node_shape, 2)),
    )
    target = reference.evaluate(truth)
    truth_field = target.plastic_field.copy()
    observed = target.displacement.copy()

    # How far the truth sits from the J2 cone, in the gauge. If this is small the
    # comparison of gate 7 is not testing anything.
    flow = j2_flow_direction(stress)
    amplitude = np.einsum("pi,pi->p", truth_field, flow) / np.maximum(
        np.einsum("pi,pi->p", flow, flow), 1e-300
    )
    departure = plastic_gauge_norm(
        truth_field - amplitude[:, None] * flow
    ) / max(plastic_gauge_norm(truth_field), 1e-300)
    print(f"  the truth departs from the J2 cone by {departure:.3f} in the gauge",
          flush=True)

    report: dict[str, object] = {
        "coefficients": basis.coefficient_count,
        "j2_cone_departure": departure,
        "truth_minimum_dissipation": float(np.min(projection.dissipation(truth_field))),
    }
    for label, scalar in (("tensor", False), ("scalar_j2", True)):
        inverse = TensorLocalInverse(
            operator=operator, basis=basis, projection=projection,
            observed_displacement=observed, scalar_j2_family=scalar,
        )
        entry = recover(inverse, basis, truth_field, arguments.peak,
                        arguments.iterations, label)
        # A restricted arm that never moves measures a dead ReLU, not a family
        # limitation, and would hand the comparison a spectacular false result.
        entry["moved"] = entry["decades"] > 0.05
        report[label] = entry
        print(f"  {label}: {entry['decades']:.2f} decades, gauge error "
              f"{entry['gauge_relative_error']:.4f}, "
              f"min dissipation {entry['minimum_dissipation']:.3e}", flush=True)
    tensor, scalar_arm = report["tensor"], report["scalar_j2"]
    separation = float(
        np.log10(scalar_arm["final_objective"] / max(tensor["final_objective"], 1e-300))
    )
    report["family_separation_decades"] = separation
    report["passed_gate_4"] = tensor["gauge_relative_error"] <= 0.05
    report["passed_gate_5"] = tensor["minimum_dissipation"] >= -1e-9
    report["passed_gate_7"] = bool(
        separation >= 1.0 and scalar_arm["moved"] and tensor["moved"]
    )
    print(f"  gate 7  the tensor family reaches {separation:.2f} decades below "
          f"the J2-restricted one", flush=True)
    return report


def gate_observability(arguments) -> dict:
    """What the displacement data can determine, with every optimiser removed.

    Gate 4 failing at a low objective is either the optimiser or the data. A
    truncated SVD of the sensitivity matrix answers it without appeal: at each
    truncation level the reconstruction is the best any method could produce
    from this observation, so a floor here is a property of the physics.
    """

    grid, operator, basis = build(arguments.spectrum_pixels, arguments.patches)
    stress = frozen_stress(operator, 91)
    projection = DissipativeProjection(stress=stress)
    base = truth_coefficients(basis, arguments.peak, 5)
    reference = TensorLocalInverse(
        operator=operator, basis=basis, projection=projection,
        observed_displacement=np.zeros((*grid.node_shape, 2)),
    )
    count = basis.coefficient_count
    matrix = np.empty((int(np.prod(grid.node_shape)) * 2, count))
    seed = np.zeros(count)
    for index in range(count):
        seed[:] = 0.0
        seed[index] = 1.0
        matrix[:, index] = reference.sensitivity_column(
            base, seed.reshape(basis.coefficient_shape)
        ).ravel()
    left, singular, right = np.linalg.svd(matrix, compute_uv=True, full_matrices=False)

    # A spatially uniform eigenstrain has a uniform eigenstress, whose interior
    # divergence vanishes identically, so it moves nothing. Measured rather than
    # argued, because it is the cleanest member of the invisible subspace.
    uniform = {}
    for channel, name in enumerate(("xx", "yy", "xy_kelvin")):
        probe = np.zeros(basis.coefficient_shape)
        probe[:, :, channel] = 1.0
        uniform[name] = float(
            np.linalg.norm(reference.sensitivity_column(base, probe))
        )

    report: dict[str, object] = {
        "coefficients": count,
        "condition_number": float(singular[0] / max(singular[-1], 1e-300)),
        "uniform_eigenstrain_response": uniform,
        "truths": {},
    }
    for name, truth in (
        ("registered", base),
        ("zero_mean", base - base.mean(axis=(0, 1), keepdims=True)),
    ):
        inverse = TensorLocalInverse(
            operator=operator, basis=basis, projection=projection,
            observed_displacement=np.zeros((*grid.node_shape, 2)),
        )
        target = inverse.evaluate(truth)
        inverse.observed = target.displacement.copy()
        field = target.plastic_field.copy()
        norm = max(plastic_gauge_norm(field), 1e-300)
        coordinates = right @ truth.ravel()
        invisible = float(
            np.linalg.norm(coordinates[singular < singular[0] * 1e-6])
            / max(np.linalg.norm(coordinates), 1e-300)
        )
        levels = []
        for decade in (2, 3, 4, 6):
            keep = singular >= singular[0] * 10.0**-decade
            solution = right[keep].T @ (
                (left[:, keep].T @ inverse.observed.ravel()) / singular[keep]
            )
            recovered = inverse.evaluate(solution.reshape(basis.coefficient_shape))
            levels.append({
                "truncation": f"1e-{decade}",
                "directions_kept": int(keep.sum()),
                "gauge_relative_error": plastic_gauge_norm(
                    recovered.plastic_field - field
                ) / norm,
            })
            print(f"    {name:<12} cut 1e-{decade}  kept {int(keep.sum()):>4}  "
                  f"gauge error {levels[-1]['gauge_relative_error']:.4f}", flush=True)
        best = min(row["gauge_relative_error"] for row in levels)
        report["truths"][name] = {
            "truth_fraction_in_null_subspace": invisible,
            "levels": levels,
            "best_gauge_relative_error": best,
        }
        print(f"  {name}: best {best:.4f}, truth in the invisible subspace "
              f"{invisible:.4f}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", default="all",
                        choices=("transpose", "gradient", "spectrum", "twin",
                                 "observability", "all"))
    parser.add_argument("--small-pixels", type=int, default=24)
    parser.add_argument("--small-patches", type=int, default=4)
    parser.add_argument("--spectrum-pixels", type=int, default=48)
    parser.add_argument("--twin-pixels", type=int, default=64)
    parser.add_argument("--patches", type=int, default=8)
    parser.add_argument("--peak", type=float, default=4.0e-4)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report: dict[str, object] = {}
    if arguments.gate in ("transpose", "all"):
        print("gate 1 -- the transpose of A", flush=True)
        report["gate_1_transpose"] = gate_transpose(arguments)
    if arguments.gate in ("gradient", "all"):
        print("gate 2 -- the gradient through the assembled projection", flush=True)
        report["gate_2_gradient"] = gate_gradient(arguments)
    if arguments.gate in ("spectrum", "all"):
        print("gate 3 -- the spectrum of du/da", flush=True)
        report["gate_3_spectrum"] = gate_spectrum(arguments)
    if arguments.gate in ("twin", "all"):
        print("gates 4-7 -- twin recovery, admissibility, and the family comparison",
              flush=True)
        report["gate_4_twin"] = gate_twin(arguments)
    if arguments.gate in ("observability", "all"):
        print("observability -- what the data determine, every optimiser removed",
              flush=True)
        report["observability"] = gate_observability(arguments)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\nwrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
