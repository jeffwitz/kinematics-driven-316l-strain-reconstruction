#!/usr/bin/env python3
"""Build the displacement-space M20 Krylov/SRIX intersection fixture.

Only the archived raw-displacement target and the existing linear
``TensorPlasticObservabilityOperator`` are used here.  This is intentionally
not a SRIX forward: the Krylov block is generated with ``rmatvec``/``matvec``
of the homogeneous linear elastic observation operator, followed by small
least-squares solves and the registered dissipative projector.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from fem_inhouse.core.kelvin import KELVIN_SCALE_2D, PLANE_STRESS_PLASTIC_GAUGE
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from scripts.krylov_projected_control import project

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "validation/reference_data/p0043_experimental_raw_femu_m20_v1"
OUTPUT = ROOT / "validation/reference_data/p0043_krylov_srix_intersection_m20_v1"
PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30
SCORED_INDICES = np.asarray([3, 7, 11, 15, 19, 23, 27, 31], dtype=int)


class _Identity:
    def apply(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)

    def adjoint(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)


def _elastic_extension(
    operator: TensorPlasticObservabilityOperator, field: np.ndarray
) -> np.ndarray:
    """Return the homogeneous elastic solution with the field's boundary data."""
    kelvin_strain = operator.kelvin_strain(field).reshape(-1, 3)
    stress_kelvin = np.einsum("pi,pij->pj", kelvin_strain, operator.elasticity)
    stress_voigt = stress_kelvin / KELVIN_SCALE_2D
    weak_residual = operator.weak_equilibrium_residual(
        stress_voigt.reshape((*operator.grid.pixel_shape, 2, 3))
    )
    return field + operator.correction_from_weak_residual(weak_residual)


def _affine_gate(operator: TensorPlasticObservabilityOperator) -> float:
    nx, ny = operator.grid.node_shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    probes = []
    for component in range(2):
        field = np.zeros((nx, ny, 2), dtype=np.float64)
        field[..., component] = 1.0
        probes.append(field)
    for component in range(2):
        for coordinate in (x, y):
            field = np.zeros((nx, ny, 2), dtype=np.float64)
            field[..., component] = coordinate
            probes.append(field)
    errors = [
        np.linalg.norm(_elastic_extension(operator, field) - field)
        / max(np.linalg.norm(field), 1e-300)
        for field in probes
    ]
    return float(max(errors))


def _stacked_mode_contributions(
    responses: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    """Stack each mode's eight state contributions as one observable vector."""
    state_count, rank = coefficients.shape
    observed = responses.shape[0]
    return np.column_stack(
        [
            (coefficients[:, mode, None] * responses[:, mode][None, :]).reshape(-1)
            for mode in range(rank)
        ]
    ).reshape(state_count * observed, rank)


def _observable_from_kelvin(
    operator: TensorPlasticObservabilityOperator, field: np.ndarray
) -> np.ndarray:
    """Apply ``matvec`` to a physical Kelvin plastic field."""
    physical = np.asarray(field, dtype=np.float64).reshape(-1, 3)
    normalized = np.linalg.solve(operator.inverse_gauge_root.T, physical.T).T
    return operator.matvec(normalized)


def _relative_errors(predicted: np.ndarray, target: np.ndarray) -> list[float]:
    return [
        float(
            np.linalg.norm(predicted[index] - target[index])
            / max(np.linalg.norm(target[index]), 1e-300)
        )
        for index in range(target.shape[0])
    ]


def main() -> int:
    started = time.perf_counter()
    source_fields = np.load(SOURCE / "fields.npz", allow_pickle=False)
    source_report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    target_all = np.asarray(source_fields["target_displacement"], dtype=np.float64)
    target = target_all[SCORED_INDICES]
    if target.shape != (8, 21, 21, 2):
        raise RuntimeError(f"unexpected scored target shape: {target.shape}")

    grid = StructuredGrid2D(20, 20, PIXEL_SIZE_MM * 20, PIXEL_SIZE_MM * 20)
    operator = TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
        transfer=_Identity(),
        whitener=_Identity(),
    )
    elastic_reference = np.stack([_elastic_extension(operator, field) for field in target])
    defect = target - elastic_reference
    defect_observed = defect.reshape(8, operator.observation_size)
    border = np.zeros((21, 21), dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    boundary_values = defect[:, border, :]
    interior_values = defect[:, ~border, :]
    boundary_max = float(np.abs(boundary_values).max())
    interior_l2 = float(np.linalg.norm(interior_values))
    affine_error = _affine_gate(operator)

    seeds = np.column_stack([operator.rmatvec(item) for item in defect_observed])
    seed_basis, seed_qr_r = np.linalg.qr(seeds, mode="reduced")
    enriched = np.column_stack(
        [operator.rmatvec(operator.matvec(seed_basis[:, index])) for index in range(8)]
    )
    combined = np.column_stack((seed_basis, enriched))
    phi_core, combined_qr_r = np.linalg.qr(combined, mode="reduced")
    if phi_core.shape != (operator.plastic_size, 16):
        raise RuntimeError(f"rank-16 block Krylov collapsed to {phi_core.shape}")

    core_response = np.column_stack(
        [operator.matvec(phi_core[:, index]) for index in range(16)]
    )
    _, response_singular, response_vt = np.linalg.svd(core_response, full_matrices=False)
    phi_normalized = phi_core @ response_vt.T
    mode_response = core_response @ response_vt.T
    phi_kelvin = np.column_stack(
        [operator.plastic_mode(phi_normalized[:, index]) for index in range(16)]
    )

    coefficients: dict[str, np.ndarray] = {}
    raw_by_state: dict[str, np.ndarray] = {}
    raw_eps: dict[str, np.ndarray] = {}
    raw_contributions: dict[str, np.ndarray] = {}
    raw_errors: dict[str, list[float]] = {}
    dissipative_by_state: dict[str, np.ndarray] = {}
    dissipative_eps: dict[str, np.ndarray] = {}
    dissipative_errors: dict[str, list[float]] = {}
    dissipative_power: dict[str, dict[str, float]] = {}
    plastic_gauge = np.asarray(PLANE_STRESS_PLASTIC_GAUGE, dtype=np.float64)

    # The projection stress is the homogeneous linear-elastic predictor stress
    # associated with each measured boundary state.  No constitutive forward is
    # used or implied by this diagnostic.
    predictor_stress = np.stack(
        [
            np.einsum(
                "pi,pij->pj",
                operator.kelvin_strain(field).reshape(-1, 3),
                operator.elasticity,
            )
            for field in elastic_reference
        ]
    )

    for rank in (8, 16):
        responses = mode_response[:, :rank]
        coeff = np.stack(
            [np.linalg.lstsq(responses, item, rcond=None)[0] for item in defect_observed]
        )
        predicted = coeff @ responses.T
        eps = (coeff @ phi_kelvin[:, :rank].T).reshape(
            8, operator.kinematics.material_point_count, 3
        )
        contributions = _stacked_mode_contributions(responses, coeff)

        increments = np.empty_like(eps)
        increments[0] = eps[0]
        increments[1:] = np.diff(eps, axis=0)
        projected_increments = []
        active = []
        powers = []
        for state in range(8):
            projected, mask = project(
                increments[state], predictor_stress[state], "gp"
            )
            projected_increments.append(projected)
            active.append(mask)
            powers.append(np.einsum("pi,pi->p", predictor_stress[state], projected))
        projected_increments_array = np.asarray(projected_increments)
        eps_diss = np.cumsum(projected_increments_array, axis=0)
        delta_diss_by_state = np.stack(
            [_observable_from_kelvin(operator, item) for item in eps_diss]
        )
        power = np.asarray(powers)
        scale = max(float(np.abs(power).max()), 1e-300)
        dissipative_by_state[str(rank)] = delta_diss_by_state
        dissipative_eps[str(rank)] = eps_diss
        dissipative_errors[str(rank)] = _relative_errors(
            delta_diss_by_state, defect_observed
        )
        dissipative_power[str(rank)] = {
            "negative_fraction_below_minus_1e-12_scale": float(np.mean(power < -1e-12 * scale)),
            "minimum": float(power.min()),
            "maximum": float(power.max()),
            "active_fraction": float(np.mean(active)),
            "raw_final_rms_gauge": float(
                np.sqrt(np.mean(np.einsum("pi,ij,pj->p", eps[-1], plastic_gauge, eps[-1])))
            ),
            "dissipative_final_rms_gauge": float(
                np.sqrt(
                    np.mean(
                        np.einsum(
                            "pi,ij,pj->p", eps_diss[-1], plastic_gauge, eps_diss[-1]
                        )
                    )
                )
            ),
        }
        coefficients[str(rank)] = coeff
        raw_by_state[str(rank)] = predicted
        raw_eps[str(rank)] = eps
        raw_contributions[str(rank)] = contributions
        raw_errors[str(rank)] = _relative_errors(predicted, defect_observed)

    rng = np.random.default_rng(20260830)
    probe_x = rng.normal(size=operator.plastic_size)
    probe_y = rng.normal(size=operator.observation_size)
    lhs = float(operator.matvec(probe_x) @ probe_y)
    rhs = float(probe_x @ operator.rmatvec(probe_y))
    dot_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)

    fields: dict[str, np.ndarray] = {
        "phi_normalized_r16": phi_normalized,
        "phi_kelvin_r16": phi_kelvin,
        "mode_response_displacement_r16": mode_response,
        "elastic_reference_displacement": elastic_reference,
        "observable_defect": defect,
    }
    for rank in (8, 16):
        key = str(rank)
        fields.update(
            {
                f"coefficients_raw_r{rank}": coefficients[key],
                f"trajectory_mode_contributions_raw_r{rank}": raw_contributions[key],
                f"delta_y_raw_by_state_r{rank}": raw_by_state[key],
                f"delta_y_raw_r{rank}": raw_by_state[key].reshape(-1),
                f"eps_p_raw_r{rank}": raw_eps[key],
                f"eps_p_dissipative_r{rank}": dissipative_eps[key],
                f"delta_y_dissipative_by_state_r{rank}": dissipative_by_state[key],
                f"delta_y_dissipative_r{rank}": dissipative_by_state[key].reshape(-1),
            }
        )

    gates = {
        "target_shape": list(target.shape),
        "final_jacobian_row_contract": int(source_fields["final_jacobian"].shape[0]),
        "observed_block_size": int(operator.observation_size),
        "boundary_max_abs_mm": boundary_max,
        "interior_l2_mm": interior_l2,
        "boundary_to_interior_ratio": boundary_max / max(interior_l2, 1e-300),
        "affine_extension_relative_error": affine_error,
        "dot_test_relative_error": float(dot_error),
        "raw_contribution_reconstruction_relative_error": {
            str(rank): float(
                np.linalg.norm(
                    raw_by_state[str(rank)].reshape(-1)
                    - raw_contributions[str(rank)].sum(axis=1)
                )
                / max(np.linalg.norm(raw_by_state[str(rank)]), 1e-300)
            )
            for rank in (8, 16)
        },
        "dissipative_predictor_power": dissipative_power,
    }
    report = {
        "schema_version": 1,
        "status": "fixture_built",
        "method": "observable-matched displacement-space block Krylov on M20",
        "source": {
            "fields": "validation/reference_data/p0043_experimental_raw_femu_m20_v1/fields.npz",
            "report": "validation/reference_data/p0043_experimental_raw_femu_m20_v1/report.json",
            "source_git_sha": source_report.get("git_sha"),
            "source_observation_profile": source_report.get("observation_profile"),
            "source_observation_weighting": source_report.get("observation_weighting"),
            "source_noise_model_used": source_report.get("noise_model_used"),
        },
        "observable_contract": {
            "scored_source_indices": SCORED_INDICES.tolist(),
            "scored_source_steps": source_report.get("scored_steps"),
            "state_order": "score index 0..7 follows source scored_steps",
            "nodal_shape_per_state": [21, 21, 2],
            "observed_components_per_state": 882,
            "stacked_observed_components": 7056,
            "component_order": "source displacement last axis [u_x, u_y]",
            "flatten_order": "NumPy C-order, state then node axes then component",
            "displacement_units": "mm",
            "transfer": "identity",
            "whitener": "identity",
        },
        "operator": {
            "implementation": "TensorPlasticObservabilityOperator.matvec/rmatvec",
            "elasticity": {"young_mpa": YOUNG_MPA, "poisson": POISSON},
            "grid": [20, 20],
            "material_points": int(operator.kinematics.material_point_count),
            "plastic_size": int(operator.plastic_size),
            "observation_size_per_state": int(operator.observation_size),
            "normalised_latent_coordinates": True,
            "physical_conversion": "operator.plastic_mode",
        },
        "basis": {
            "construction": (
                "eight displacement residual seeds with rmatvec, one A^T A block "
                "enrichment, QR reorthogonalisation"
            ),
            "ranks_archived": [8, 16],
            "ordering": "SVD of A_u Phi_core response, descending displacement singular value",
            "response_singular_values_r16": response_singular.tolist(),
            "seed_qr_r": seed_qr_r.tolist(),
            "combined_qr_r": combined_qr_r.tolist(),
        },
        "dissipative_projection": {
            "implementation": "scripts.krylov_projected_control.project",
            "corrected_projector_commit": "2e92c0f8a5d5bb88d7a0d033667beabb4d3a765b",
            "projector": "gp",
            "stress_definition": (
                "homogeneous linear-elastic predictor stress of each elastic boundary "
                "extension; no constitutive forward"
            ),
            "increment_definition": (
                "first cumulative increment, then state-to-state differences of the "
                "raw fitted plastic fields"
            ),
        },
        "gates": gates,
        "fit": {
            "raw_relative_observable_error_by_state": raw_errors,
            "dissipative_relative_observable_error_by_state": dissipative_errors,
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "claims": {
            "reproduces_old_strain_space_krylov_bitwise": False,
            "observable_matched_fixture": True,
            "new_srix_forward": False,
            "registered_case_methodological_diagnostic_only": True,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / "fields.npz", **fields)
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
