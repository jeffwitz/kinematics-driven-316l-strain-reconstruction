#!/usr/bin/env python3
"""Freeze margin(E) for the milestone-4 preregistration, before any run.

Computes the operational noise margin defined in
`validation/shared_tensor_generator_preregistration.md`: the archived DIC
repetition residual (`centred_repeat_flow_pixels.npy`), cropped to the same
100x100-element window as `learn_flow_direction_p43.py`, converted to
canonical millimetres and passed through the same Kelvin strain derivation,
divided by each held-out state's elastic defect; the median over the five
held-out states is the value to inscribe into the preregistration.

This script runs no training and is executed exactly once, before the first
run. The frozen value is what the preregistration's margins refer to; the
preregistration itself is never modified afterwards.

The elastic-lifting residual guard replicates the check in
`learn_flow_direction_p43.py`: if the replicated derivation here does not
equilibrate, the margin would measure a convention bug, not a noise floor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fem_inhouse.core.kelvin import KELVIN_SCALE_2D
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.measurement.coordinates import image_flow_to_canonical
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior

ROOT = Path(__file__).resolve().parents[1]
HISTORY = (
    ROOT
    / "validation/reference_data/dic_multistep_history_p0043_repaired_v1"
    / "repaired_history_mm.npy"
)
NOISE = (
    ROOT
    / "validation/reference_data/dic_uncertainty_propagation_p0043_v1"
    / "centred_repeat_flow_pixels.npy"
)
OUT = ROOT / "validation/_generated/shared_tensor_generator"
PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30
ORIGIN = (1580, 1030)
PIXELS = 100
REFERENCE_STATE = 20
HELDOUT = (24, 28, 32, 36, 40)


class _Identity:
    def apply(self, values):
        return np.asarray(values, dtype=np.float64)

    def adjoint(self, values):
        return np.asarray(values, dtype=np.float64)


def main() -> int:
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

    report = json.loads(
        (HISTORY.with_name("report.json")).read_text(encoding="utf-8")
    )
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
            f"replicated elastic lift does not equilibrate ({residual:.3e}): "
            "the margin would measure a convention bug, not a noise floor"
        )

    defects: dict[int, float] = {}
    for state in HELDOUT:
        measured = kelvin_strain(history[state] - reference)
        elastic = kelvin_strain(elastic_lift(history[state] - reference))
        defects[state] = float(np.linalg.norm(measured - elastic))

    noise = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    window = (slice(x0, x0 + PIXELS + 1), slice(y0, y0 + PIXELS + 1))
    noise_canonical = image_flow_to_canonical(
        np.asarray(noise[window]), pixel_size_mm=PIXEL_SIZE_MM
    )
    noise_norm = float(np.linalg.norm(kelvin_strain(noise_canonical)))
    # Placement sanity check: the noise grid's physical alignment with the
    # history grid cannot be asserted by a residual; a second window 300 nodes
    # away along the first axis bounds the sensitivity of the margin to it.
    shifted = (slice(x0 + 300, x0 + 300 + PIXELS + 1), window[1])
    shifted_canonical = image_flow_to_canonical(
        np.asarray(noise[shifted]), pixel_size_mm=PIXEL_SIZE_MM
    )
    shifted_norm = float(np.linalg.norm(kelvin_strain(shifted_canonical)))

    margins = {state: noise_norm / defects[state] for state in HELDOUT}
    margin = float(np.median(list(margins.values())))
    payload = {
        "schema_version": 1,
        "definition": "median over held-out states of "
        "|eps_noise| / |eps_measured - eps_elastic|",
        "heldout_states": list(HELDOUT),
        "defects": defects,
        "noise_strain_norm": noise_norm,
        "noise_strain_norm_shifted_300": shifted_norm,
        "per_state_margins": margins,
        "margin_frozen": margin,
        "elastic_lifting_residual_guard": residual,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "margin_frozen.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
