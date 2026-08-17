#!/usr/bin/env python3
"""Pre-run verification of the P43 200x200 campaign window.

Two checks, both required before any 200x200 run:

1. **Alignment.** The 200x200 crop centred on the historical 100x100 zone
   (origin `(1530, 980)`; the 100x100 at `(1580, 1030)` is its exact centre)
   must reproduce the historical crop on its central 101x101 nodes, bit for
   bit, and must lie inside the repaired sub-ROI.
2. **Elastic lifting on the new boundary.** The same guarded derivation as the
   qualified pipeline, on the 200x200 window: the interior residual of the
   elastic lift of the final measured state must stay at the qualified level
   (`<= 1e-8`; the broken historical conversion reads `0.32`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fem_inhouse.core.kelvin import KELVIN_SCALE_2D
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
ORIGIN_200 = (1530, 980)
ORIGIN_100 = (1580, 1030)
PIXELS_200 = 200
PIXELS_100 = 100
REFERENCE_STATE = 20
FINAL_STATE = 40


class _Identity:
    def apply(self, values):
        return np.asarray(values, dtype=np.float64)

    def adjoint(self, values):
        return np.asarray(values, dtype=np.float64)


def crop(source: np.ndarray, bounds: list[int], origin, pixels) -> np.ndarray:
    x0, y0 = origin
    return np.asarray(
        source[
            :,
            x0 - bounds[0] : x0 + pixels - bounds[0] + 1,
            y0 - bounds[2] : y0 + pixels - bounds[2] + 1,
            :,
        ],
        dtype=np.float64,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "window200_verification.json")
    arguments = parser.parse_args()

    report = json.loads((HISTORY.with_name("report.json")).read_text(encoding="utf-8"))
    bounds = list(map(int, report["solve_bounds"]))
    source = np.load(HISTORY, mmap_mode="r", allow_pickle=False)
    grid_200 = StructuredGrid2D(
        PIXELS_200, PIXELS_200, PIXEL_SIZE_MM * PIXELS_200, PIXEL_SIZE_MM * PIXELS_200
    )
    operator = TensorPlasticObservabilityOperator.build(
        grid_200,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
        transfer=_Identity(),
        whitener=_Identity(),
    )

    # -- check 1: bounds and alignment ----------------------------------------
    x0, y0 = ORIGIN_200
    inside = (
        x0 >= bounds[0]
        and x0 + PIXELS_200 <= bounds[1]
        and y0 >= bounds[2]
        and y0 + PIXELS_200 <= bounds[3]
    )
    history_200 = crop(source, bounds, ORIGIN_200, PIXELS_200)
    history_100 = crop(source, bounds, ORIGIN_100, PIXELS_100)
    # A 201-node window centred on the historical centre spans [1530, 1730];
    # its central 101-node block is indices 50..150, not 100..200.
    centre = history_200[:, 50:151, 50:151, :]
    alignment = float(np.max(np.abs(centre - history_100)))
    print(f"window inside sub-ROI: {inside}", flush=True)
    print(f"alignment max |centre_200 - historical_100|: {alignment:.3e}", flush=True)

    # -- check 2: elastic lifting residual on the new boundary ----------------
    def kelvin_strain(field) -> np.ndarray:
        return operator.kelvin_strain(field).reshape(-1, 3)

    def stress_of(strain: np.ndarray) -> np.ndarray:
        return np.einsum("pi,pij->pj", strain.reshape(-1, 3), operator.elasticity)

    def divergence(stress_kelvin: np.ndarray) -> np.ndarray:
        voigt = stress_kelvin.reshape(-1, 3) / KELVIN_SCALE_2D
        return pack_interior(
            operator.kinematics.divergence_from_sample_stress(
                voigt.reshape((PIXELS_200, PIXELS_200, 2, 3))
            )
        )

    def elastic_lift(field: np.ndarray) -> np.ndarray:
        forcing = -divergence(stress_of(kelvin_strain(field))) / operator.quadrature_weight
        lifted = field.copy()
        lifted[1:-1, 1:-1, :] -= operator.solve_stiffness(forcing).reshape(
            PIXELS_200 - 1, PIXELS_200 - 1, 2
        )
        return lifted

    reference = history_200[REFERENCE_STATE]
    residual = np.linalg.norm(
        divergence(stress_of(kelvin_strain(elastic_lift(history_200[FINAL_STATE] - reference))))
    )
    print(f"elastic lifting residual at 200x200 (guard): {residual:.3e}", flush=True)

    payload = {
        "schema_version": 1,
        "origin_200": list(ORIGIN_200),
        "origin_100_historical": list(ORIGIN_100),
        "pixels": PIXELS_200,
        "solve_bounds": bounds,
        "window_inside_sub_roi": bool(inside),
        "alignment_max_abs_diff": alignment,
        "alignment_pass": bool(inside and alignment == 0.0),
        "elastic_lifting_residual": residual,
        "elastic_lifting_pass": bool(residual <= 1e-8),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
