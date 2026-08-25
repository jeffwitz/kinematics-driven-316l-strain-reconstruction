#!/usr/bin/env python3
"""Run one experimental P43 M100 rank-three scale-up from the M20 result."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np

from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET
from fem_inhouse.identification.dic_noise_reference import load_dic_noise_reference
from fem_inhouse.identification.srix_equilibrium_gap import SrixTheta4
from fem_inhouse.identification.svd_parameter_basis import svd_parameter_basis
from scripts.qualify_srix_femu_direct_sensitivity import _direct_jacobian
from scripts.qualify_srix_p43_experimental_rank3 import (
    _run_start,
)
from scripts.qualify_srix_p0043_synthetic_smoke import (
    _factory,
    _forward,
    _git,
    _load_inputs,
    _make_path,
    _vector,
)
from scripts.qualify_srix_regm_twin import _theta_from_preset

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "validation/reference_data/p0043_experimental_srix_m100_v1"
M20_REPORT = ROOT / "validation/reference_data/p0043_experimental_srix_m20_v1/report.json"
FULL_CROP = (1580, 1680, 1030, 1130)
PIXEL_SIZE_MM = 0.00184
DIC_NOISE_REPORT = (
    ROOT / "validation/reference_data/dic_boundary_loading_subspace_p0043_v1/report.json"
)
H = 1.5e-3


def _theta(payload: dict[str, float]) -> SrixTheta4:
    return SrixTheta4(
        tau0_mpa=payload["tau0_mpa"],
        r_mpa=payload["R_mpa"],
        q_mpa=payload["Q_mpa"],
        b=payload["b"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    noise_reference = load_dic_noise_reference(
        DIC_NOISE_REPORT, pixel_size_mm=PIXEL_SIZE_MM
    )
    pixel_noise_mm = float(noise_reference["robust_mm"])
    m20 = json.loads(M20_REPORT.read_text(encoding="utf-8"))
    if not m20.get("gate", {}).get("m100_authorized", False):
        raise SystemExit("M20 gate did not authorize M100; refusing scale-up")
    starts = m20["starts"]
    best = min(starts, key=lambda item: item["cost"]["final_whitened_rms"])
    initial = _theta(best["identified"])
    measured_macro, angles, provenance = _load_inputs(FULL_CROP)
    path = _make_path(measured_macro, 4)
    scored = tuple(4 * index for index in range(1, 9))
    target = [np.asarray(step.boundary, dtype=np.float64).copy() for step in path]
    prior = _theta_from_preset()
    eta_ref = prior.log_coordinates()
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so"))
    factory = _factory(angles, library, args.threads)
    prior_fields, prior_timing = _forward(prior, path, angles, library, args.threads)
    prior_matrix, _ = _direct_jacobian(
        fields=prior_fields,
        scored=scored,
        orientations=angles,
        theta=prior,
        library=library,
        threads=args.threads,
        transfer=__import__("scripts.qualify_srix_p0043_synthetic_smoke", fromlist=["_Identity"])._Identity(),
        h=H,
        material_factory=factory,
    )
    basis = svd_parameter_basis(prior_matrix / pixel_noise_mm, fixed_rank=3)
    initial_eta = initial.log_coordinates()
    z_all = basis.right_singular_vectors.T @ (initial_eta - eta_ref)
    result = _run_start(
        "M100_from_M20",
        z_all[:3],
        eta_ref,
        basis,
        target,
        path,
        scored,
        angles,
        library,
        args.threads,
        args.max_nfev,
        discarded_direction=basis.discarded_basis[:, 0],
        discarded_coordinate=float(z_all[3]),
    )
    report = {
        "schema_version": 1,
        "method": "P43 experimental direct FEMU rank-three SVD identification M100",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "crop": list(FULL_CROP),
        "mesh": list(angles.shape[:2]),
        "path_steps": len(path),
        "scored_steps": list(scored),
        "observation_profile": "measured displacement identity plus scalar DIC whitening",
        "dic_uncertainty_mm": pixel_noise_mm,
        "dic_noise_reference": noise_reference,
        "parameter_preset": DEFAULT_PARAMETER_SET,
        "shadow_fd_step": H,
        "m20_report": str(M20_REPORT),
        "m20_initialization": initial.as_runtime_overrides(),
        "prior": prior.as_runtime_overrides(),
        "provenance": provenance,
        "prior_forward": prior_timing,
        "prior_svd": {
            "singular_values": basis.singular_values.tolist(),
            "normalized_singular_values": basis.normalized_singular_values.tolist(),
            "right_singular_vectors": basis.right_singular_vectors.tolist(),
        },
        "run": result,
        "claims": {
            "experimental_m100_completed": True,
            "experimental_m100_parameters_identified": False,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cost": result["cost"], "optimizer": result["optimizer"], "identified": result["identified"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
