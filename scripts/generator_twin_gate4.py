#!/usr/bin/env python3
"""Gate 4 construction: the twin with a controlled invisible share.

Builds the milestone-4 twin on the campaign grid (100x100), measures the
kernel share of the truth against the null directions of the displacement
operator, and writes the twin observed history that
`learn_flow_direction_p43.py --observed-history` consumes.

The construction is deterministic and registered:

* the same frozen predictor stress as the milestone-3 observability gate
  (seed 91), so the null directions and the admissibility cone live in the
  same frame;
* the null directions are the right singular vectors of the 192-column
  sensitivity matrix with `sigma < sigma_1 * 1e-6`, the identical code path as
  the milestone-3 measurement, recomputed on the campaign grid (the frozen
  artifact stores spectra and shares, not vectors);
* the truth pattern is `truth_coefficients(basis, peak=4e-4, seed=5)` with its
  patch mean removed (the milestone-3 zero-mean variant); if its kernel share
  exceeds the registered 20 %, the null component is projected out — a
  deterministic construction step, no data involved;
* per-state truth increments are `w_s * pattern` with `w_s = (s - 20) / 20`
  for states 21-40;
* the observed history is `reference + elastic_lift(history_s - reference)
  + A(sum of truth increments)`, so the elastic baseline is the real one and
  only the plastic part is synthetic.

The script runs no training. It writes the twin and its construction report,
then exits. The elastic-lifting residual guard replicates the margin script:
if it does not equilibrate, the twin would embed a convention bug.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fem_inhouse.core.kelvin import KELVIN_SCALE_2D
from fem_inhouse.identification.tensor_local_inverse import (
    DissipativeProjection,
    TensorLocalBasis,
    TensorLocalInverse,
    plastic_gauge_norm,
)
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior

ROOT = Path(__file__).resolve().parents[1]
HISTORY = (
    ROOT
    / "validation/reference_data/dic_multistep_history_p0043_repaired_v1"
    / "repaired_history_mm.npy"
)
OUT = ROOT / "validation/_generated/shared_tensor_generator"
PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30
UNIAXIAL_STRESS_MPA = 205.0
ORIGIN = (1580, 1030)
PIXELS = 100
PATCHES = 8
REFERENCE_STATE = 20
STATES = list(range(21, 41))
PEAK = 4.0e-4
TRUTH_SEED = 5
STRESS_SEED = 91
MAX_KERNEL_SHARE = 0.20


class _Identity:
    def apply(self, values):
        return np.asarray(values, dtype=np.float64)

    def adjoint(self, values):
        return np.asarray(values, dtype=np.float64)


def frozen_stress(operator, seed: int) -> np.ndarray:
    """The milestone-3 `frozen_stress`, reproduced verbatim for the same frame."""

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
    return base + reference


def truth_coefficients(basis: TensorLocalBasis, peak: float, seed: int) -> np.ndarray:
    """The milestone-3 smooth, three-component, deliberately non-J2 truth."""

    generator = np.random.default_rng(seed)
    patches = basis.coefficient_shape[0]
    values = np.zeros(basis.coefficient_shape)
    axial = np.abs(generator.standard_normal((patches, patches))) + 0.4
    values[:, :, 1] = axial
    values[:, :, 0] = -axial * (0.5 + 0.45 * generator.standard_normal((patches, patches)))
    values[:, :, 2] = 0.6 * axial * generator.standard_normal((patches, patches))
    return values * (peak / float(np.abs(values[:, :, 1]).max()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "twin_gate4.npz")
    arguments = parser.parse_args()

    grid = StructuredGrid2D(
        PIXELS, PIXELS, PIXEL_SIZE_MM * PIXELS, PIXEL_SIZE_MM * PIXELS
    )
    operator = TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
        transfer=_Identity(),
        whitener=_Identity(),
    )
    basis = TensorLocalBasis.build(grid.nx, grid.ny, PATCHES)

    # -- elastic lifting guard ------------------------------------------------
    def kelvin_strain(field) -> np.ndarray:
        return operator.kelvin_strain(field).reshape(-1, 3)

    def stress_of(strain: np.ndarray) -> np.ndarray:
        return np.einsum("pi,pij->pj", strain.reshape(-1, 3), operator.elasticity)

    def divergence(stress_kelvin: np.ndarray) -> np.ndarray:
        voigt = stress_kelvin.reshape(-1, 3) / KELVIN_SCALE_2D
        return pack_interior(
            operator.kinematics.divergence_from_sample_stress(
                voigt.reshape((PIXELS, PIXELS, 2, 3))
            )
        )

    def elastic_lift(field: np.ndarray) -> np.ndarray:
        forcing = -divergence(stress_of(kelvin_strain(field))) / operator.quadrature_weight
        lifted = field.copy()
        lifted[1:-1, 1:-1, :] -= operator.solve_stiffness(forcing).reshape(
            PIXELS - 1, PIXELS - 1, 2
        )
        return lifted

    report = json.loads((HISTORY.with_name("report.json")).read_text(encoding="utf-8"))
    bounds = list(map(int, report["solve_bounds"]))
    source = np.load(HISTORY, mmap_mode="r", allow_pickle=False)
    x0, y0 = ORIGIN
    history = np.asarray(
        source[
            :,
            x0 - bounds[0] : x0 + PIXELS - bounds[0] + 1,
            y0 - bounds[2] : y0 + PIXELS - bounds[2] + 1,
            :,
        ],
        dtype=np.float64,
    )
    reference = history[REFERENCE_STATE]
    residual = np.linalg.norm(
        divergence(stress_of(kelvin_strain(elastic_lift(history[40] - reference))))
    )
    print(f"elastic lifting residual (guard): {residual:.3e}", flush=True)
    if residual > 1e-8:
        raise RuntimeError(
            f"replicated elastic lift does not equilibrate ({residual:.3e})"
        )

    # -- null directions, the milestone-3 code path ---------------------------
    # The sensitivity map is measured at the raw truth as base point, exactly as
    # the milestone-3 observability gate did; the projection active set at the
    # base is what defines the map, so the base must come first.
    raw_truth = truth_coefficients(basis, PEAK, TRUTH_SEED)
    stress = frozen_stress(operator, STRESS_SEED)
    projection = DissipativeProjection(stress=stress)
    inverse = TensorLocalInverse(
        operator=operator,
        basis=basis,
        projection=projection,
        observed_displacement=np.zeros((*grid.node_shape, 2)),
    )
    count = basis.coefficient_count
    matrix = np.empty((int(np.prod(grid.node_shape)) * 2, count))
    seed = np.zeros(count)
    for index in range(count):
        seed[:] = 0.0
        seed[index] = 1.0
        matrix[:, index] = inverse.sensitivity_column(
            raw_truth, seed.reshape(basis.coefficient_shape)
        ).ravel()
    _, singular, right = np.linalg.svd(matrix, compute_uv=True, full_matrices=False)
    null_mask = singular < singular[0] * 1e-6
    null_vectors = right[null_mask]
    print(
        f"null directions on the campaign grid: {int(null_mask.sum())} of {count}, "
        f"condition {singular[0] / max(singular[-1], 1e-300):.3e}",
        flush=True,
    )

    # -- the truth, with a registered invisible share -------------------------
    truth = raw_truth - raw_truth.mean(axis=(0, 1), keepdims=True)
    coordinates = right @ truth.ravel()
    raw_share = float(
        np.linalg.norm(coordinates[null_mask]) / max(np.linalg.norm(coordinates), 1e-300)
    )
    if raw_share > MAX_KERNEL_SHARE:
        truth = truth - (null_vectors.T @ (null_vectors @ truth.ravel())).reshape(
            basis.coefficient_shape
        )
        coordinates = right @ truth.ravel()
    share = float(
        np.linalg.norm(coordinates[null_mask]) / max(np.linalg.norm(coordinates), 1e-300)
    )
    print(f"kernel share: raw {raw_share:.4f} -> registered {share:.4f} "
          f"(<= {MAX_KERNEL_SHARE})", flush=True)

    # -- strict dissipation against the frozen predictor ----------------------
    evaluation = inverse.evaluate(truth)
    minimum = float(evaluation.minimum_dissipation)
    gauge = plastic_gauge_norm(evaluation.plastic_field)
    print(f"truth: gauge {gauge:.3e}, minimum dissipation {minimum:.3e}", flush=True)
    if minimum < -1e-9:
        # The milestone-3 gate-5 criterion, carried over: the projection leaves
        # a numerical floor of ~1e-17; a real violation is orders larger.
        raise RuntimeError("the truth is not strictly dissipative; construction fails")

    # -- per-state observed history -------------------------------------------
    observed = np.zeros_like(history)
    observed[REFERENCE_STATE] = reference
    cumulative = np.zeros(basis.coefficient_shape)
    increments: list[np.ndarray] = []
    for state in STATES:
        weight = (state - REFERENCE_STATE) / (STATES[-1] - REFERENCE_STATE)
        cumulative = cumulative + weight * truth
        increments.append((weight * truth).copy())
        state_inverse = TensorLocalInverse(
            operator=operator,
            basis=basis,
            projection=projection,
            observed_displacement=np.zeros((*grid.node_shape, 2)),
        )
        plastic_displacement = state_inverse.evaluate(cumulative).displacement
        observed[state] = (
            reference + elastic_lift(history[state] - reference) + plastic_displacement
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output,
        observed_history=observed,
        truth_pattern=truth,
        truth_increments=np.stack(increments),
        null_vectors=null_vectors,
        singular_values=singular,
        frozen_stress=stress,
    )
    construction = {
        "schema_version": 1,
        "pixels": PIXELS,
        "patches": PATCHES,
        "coefficients": int(count),
        "null_directions": int(null_mask.sum()),
        "condition_number": float(singular[0] / max(singular[-1], 1e-300)),
        "kernel_share_raw": raw_share,
        "kernel_share_registered": share,
        "max_kernel_share": MAX_KERNEL_SHARE,
        "truth_gauge": gauge,
        "minimum_dissipation": minimum,
        "peak": PEAK,
        "truth_seed": TRUTH_SEED,
        "stress_seed": STRESS_SEED,
        "elastic_lifting_residual_guard": residual,
        "weights": [(s, (s - REFERENCE_STATE) / (STATES[-1] - REFERENCE_STATE)) for s in STATES],
    }
    (arguments.output.with_suffix(".construction.json")).write_text(
        json.dumps(construction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(construction, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
