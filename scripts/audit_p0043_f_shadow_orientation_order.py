#!/usr/bin/env python3
"""Check that F base and shadow material batches receive identical orientations."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET, get_parameter_set
from fem_inhouse.identification.srix_parameter_coordinates import SrixTheta9
from scripts.qualify_srix_p0043_synthetic_smoke import CROP, _factory, _load_inputs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/reference_data/p0043_f_mapping_reidentification_v1/shadow_orientation_audit.json"


def main() -> int:
    _, angles, _ = _load_inputs(CROP)
    lib = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so"))
    eta = SrixTheta9.from_parameter_set(get_parameter_set(DEFAULT_PARAMETER_SET)).log_coordinates()
    basis = np.eye(9)
    factory = _factory(angles, lib, 1, "F")
    base = factory(SrixTheta9.from_log_coordinates(eta).as_runtime_overrides())
    plus = factory(SrixTheta9.from_log_coordinates(eta + 1.5e-3 * basis[:, 0]).as_runtime_overrides())
    minus = factory(SrixTheta9.from_log_coordinates(eta - 1.5e-3 * basis[:, 0]).as_runtime_overrides())
    rb = base._bridge.rotations_global_to_material
    rp = plus._bridge.rotations_global_to_material
    rm = minus._bridge.rotations_global_to_material
    report = {
        "element_order": "F", "point_count": int(rb.shape[0]),
        "base_plus_rotations_identical": bool(np.array_equal(rb, rp)),
        "base_minus_rotations_identical": bool(np.array_equal(rb, rm)),
        "max_base_plus_rotation_difference": float(np.max(np.abs(rb - rp))),
        "max_base_minus_rotation_difference": float(np.max(np.abs(rb - rm))),
        "orientation_order_suspect": bool(not np.array_equal(rb, rp) or not np.array_equal(rb, rm)),
        "note": "This closes only the orientation-copy hypothesis; it does not validate the shadow forcing assembly.",
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
