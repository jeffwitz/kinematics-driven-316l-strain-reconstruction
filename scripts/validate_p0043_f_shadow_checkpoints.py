#!/usr/bin/env python3
"""Spot-check repaired F shadow columns against centred FD at GN checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET, get_parameter_set
from fem_inhouse.identification.srix_parameter_coordinates import SrixTheta9
from scripts.qualify_srix_p0043_synthetic_smoke import (
    CROP,
    _forward,
    _load_inputs,
    _make_path,
    _vector,
)
from scripts.qualify_srix_svd_shadow import _direct_shadow, _step_sizes

ROOT = Path(__file__).resolve().parents[1]
OPT = ROOT / "validation/reference_data/p0043_f_mapping_reidentification_shadow_v1"
SVD = ROOT / "validation/reference_data/p0043_f_mapping_reidentification_v1/svd_f.json"
FD = ROOT / "validation/reference_data/p0043_f_mapping_reidentification_v1/full_jacobian_f_fd.npy"
DEFAULT_OUTPUT = OPT / "shadow_fd_checkpoints.json"


def _compare(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    nl = float(np.linalg.norm(left))
    nr = float(np.linalg.norm(right))
    return float(np.linalg.norm(left - right) / nr), float(np.dot(left, right) / (nl * nr))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")

    report = json.loads((OPT / "optimization_report.json").read_text())
    svd = json.loads(SVD.read_text())
    vectors = np.asarray(svd["right_singular_vectors"], dtype=float)
    steps = _step_sizes(np.asarray(svd["singular_values"], dtype=float), 9)
    modes = (0, 2)
    eta_ref = SrixTheta9.from_parameter_set(
        get_parameter_set(DEFAULT_PARAMETER_SET)
    ).log_coordinates()
    accepted = [entry for entry in report["history"] if entry.get("accepted")]
    if not accepted:
        raise RuntimeError("no accepted optimization checkpoint")
    checkpoints = {
        "prior": eta_ref,
        "after_first_accepted_step": np.asarray(
            next(entry["eta"] for entry in accepted if entry["evaluation"] == 5)
        ),
        "final": np.asarray(report["final_eta"], dtype=float),
    }

    measured, angles, _ = _load_inputs(CROP)
    path = _make_path(measured, 4)
    scored = tuple(4 * i for i in range(1, 9))
    target = [np.asarray(step.boundary, dtype=float) for step in path]
    library = os.environ.get(
        "MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so")
    )
    fd_full = np.load(FD)
    results: dict[str, object] = {}
    for name, eta in checkpoints.items():
        base_fields, base_timing = _forward(
            SrixTheta9.from_log_coordinates(eta), path, angles, library, args.threads, "F"
        )
        basis = vectors[:, modes]
        shadow, shadow_timing = _direct_shadow(
            fields=base_fields,
            basis=basis,
            eta=eta,
            step_sizes=steps[list(modes)],
            angles=angles,
            scored=scored,
            library=library,
            threads=args.threads,
            element_order="F",
        )
        point: dict[str, object] = {
            "verification_residual": float(base_timing["verification_residual"]),
            "modes": list(modes),
            "columns": [],
            "timing": {"forward": base_timing, "shadow": shadow_timing},
        }
        for local, mode in enumerate(modes):
            if name == "prior":
                fd_column = fd_full @ vectors[:, mode]
            else:
                plus, _ = _forward(
                    SrixTheta9.from_log_coordinates(eta + steps[mode] * vectors[:, mode]),
                    path, angles, library, args.threads, "F"
                )
                minus, _ = _forward(
                    SrixTheta9.from_log_coordinates(eta - steps[mode] * vectors[:, mode]),
                    path, angles, library, args.threads, "F"
                )
                fd_column = (
                    _vector(plus, scored, target) - _vector(minus, scored, target)
                ) / (2.0 * steps[mode])
            error, cosine = _compare(shadow[:, local], fd_column)
            point["columns"].append({
                "mode": mode,
                "step": float(steps[mode]),
                "relative_error": error,
                "cosine": cosine,
            })
        results[name] = point

    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
