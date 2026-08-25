#!/usr/bin/env python3
"""Qualify the F nine-parameter shadow Jacobian against centered F finite differences."""

from __future__ import annotations

import json
import os
import argparse
from pathlib import Path

import numpy as np

from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET, get_parameter_set
from fem_inhouse.identification.srix_parameter_coordinates import SrixTheta9
from scripts.qualify_srix_p0043_synthetic_smoke import CROP, _forward, _load_inputs, _make_path, _vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/reference_data/p0043_f_mapping_reidentification_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.source if args.source.is_absolute() else ROOT / args.source
    measured, angles, provenance = _load_inputs(CROP)
    path = _make_path(measured, 4)
    scored = tuple(4 * i for i in range(1, 9))
    target = [np.asarray(s.boundary, float).copy() for s in path]
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so"))
    eta0 = SrixTheta9.from_parameter_set(get_parameter_set(DEFAULT_PARAMETER_SET)).log_coordinates()
    h = 1.5e-3
    columns = []
    timings = []
    for j in range(9):
        ep, em = eta0.copy(), eta0.copy()
        ep[j] += h; em[j] -= h
        fp, tp = _forward(SrixTheta9.from_log_coordinates(ep), path, angles, library, 4, "F")
        fm, tm = _forward(SrixTheta9.from_log_coordinates(em), path, angles, library, 4, "F")
        columns.append((_vector(fp, scored, target) - _vector(fm, scored, target)) / (2.0 * h))
        timings.append({"parameter": j, "plus": tp, "minus": tm})
    jfd = np.column_stack(columns)
    jshadow = np.load(out / "full_jacobian_f.npy")
    errors = [float(np.linalg.norm(jshadow[:,j]-jfd[:,j]) / np.linalg.norm(jfd[:,j])) for j in range(9)]
    cos = [float(np.dot(jshadow[:,j],jfd[:,j])/(np.linalg.norm(jshadow[:,j])*np.linalg.norm(jfd[:,j]))) for j in range(9)]
    np.save(out / "full_jacobian_f_fd.npy", jfd)
    report = {"method": "centered F finite differences", "h": h, "element_order": "F",
              "relative_column_errors": errors, "column_cosines": cos,
              "max_relative_error": max(errors), "min_cosine": min(cos), "timings": timings,
              "shadow_qualified": bool(max(errors) <= 0.015 and min(cos) >= 0.999),
              "provenance": provenance}
    (out / "projected_shadow_f_qualification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"max_relative_error": max(errors), "min_cosine": min(cos), "shadow_qualified": report["shadow_qualified"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
