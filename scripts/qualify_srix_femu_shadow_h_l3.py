#!/usr/bin/env python3
"""Qualify the two prospective shadow finite-difference steps on full L3."""

from __future__ import annotations

import json
import subprocess
import time
from fractions import Fraction

import numpy as np
from scipy.linalg import subspace_angles

from fem_inhouse.identification.dic_whitening import DICSpectralTransfer
from scripts.qualify_srix_femu_common_path_gate import _common_path
from scripts.qualify_srix_femu_direct_sensitivity import (
    ROOT,
    _direct_jacobian,
    _geometry,
    _oracle_config,
)
from scripts.qualify_srix_femu_fixed_path_gate import _fixed_path_trajectory
from scripts.qualify_srix_femu_path_convergence_rebaseline import (
    TRANSFER,
    _mandatory_refine,
    _nearest_indices,
    _observed_forward,
)
from scripts.qualify_srix_regm_transfer_noise import _WrapFreeTransfer
from scripts.qualify_srix_regm_twin import _orientation_map, _theta_from_preset

SOURCE = ROOT / "validation/reference_data/srix_femu_path_convergence_v3"
OUTPUT = ROOT / "validation/reference_data/srix_femu_shadow_h_l3_v1"
REPAIR_FRACTIONS = (
    "0.254882812", "0.254394531", "0.274414062", "0.361328125",
    "0.375976562", "0.383789062", "0.395507812", "0.395019531",
    "0.401367188", "0.418945312", "0.424804688", "0.438964844",
    "0.477050781", "0.481933594", "0.485839844", "0.495605469",
    "0.502929688", "0.504882812", "0.522460938", "0.567382812",
    "0.571289062", "0.570800781", "0.616699219", "0.830078125",
    "0.876953125",
)
H_VALUES = (1.5e-3, 1.0e-3)


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _l3_path(pixels: int) -> list:
    arrays = np.load(SOURCE / "path_convergence.npz")
    l2 = _common_path(arrays["end_fractions_L2"].tolist(), pixels=pixels)
    mandatory = _mandatory_refine(l2, pixels=pixels)
    repairs = [float(Fraction(value).limit_denominator(4096)) for value in REPAIR_FRACTIONS]
    return _common_path([*(step.end_fraction for step in mandatory), *repairs], pixels=pixels)


def _column_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, list[float]]:
    errors = []
    cosines = []
    for index in range(left.shape[1]):
        a = left[:, index]
        b = right[:, index]
        errors.append(float(np.linalg.norm(a - b) / np.linalg.norm(b)))
        cosines.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    angles = np.degrees(subspace_angles(left[:, :3], right[:, :3])).tolist()
    return {
        "column_relative_l2": errors,
        "column_cosines": cosines,
        "rank3_principal_angles_degrees": angles,
    }


def main() -> None:
    output = OUTPUT
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pixels = 8
    path = _l3_path(pixels)
    source_report = json.loads((SOURCE / "report.json").read_text())
    fields = _fixed_path_trajectory(
        theta=_theta_from_preset(),
        path=path,
        initial_displacement=None,
        pixels=pixels,
        library="build/mfront/src/libBehaviour.so",
        threads=4,
        config=_oracle_config(),
    )
    scored = _nearest_indices(fields, source_report["scored_target_fractions"])
    transfer = _WrapFreeTransfer(DICSpectralTransfer.from_sinusoidal_csv(TRANSFER))
    matrices: dict[str, np.ndarray] = {}
    timings: dict[str, dict[str, float | int]] = {}
    for h in H_VALUES:
        key = f"{h:.4g}"
        matrix, timing = _direct_jacobian(
            fields=fields,
            scored=scored,
            orientations=_orientation_map(pixels),
            theta=_theta_from_preset(),
            library="build/mfront/src/libBehaviour.so",
            threads=4,
            transfer=transfer,
            h=h,
        )
        matrices[key] = matrix
        timings[key] = timing
        print(f"completed h={key}: {timing}", flush=True)

    primary = matrices["0.0015"]
    control = matrices["0.001"]
    stability = _column_comparison(primary, control)
    arrays = np.load(SOURCE / "path_convergence.npz")
    l2 = arrays["jacobian_L2"]
    l3_geometries = {key: _geometry(value) for key, value in matrices.items()}
    l2_to_l3 = {}
    for key, matrix in matrices.items():
        errors = _column_comparison(l2, matrix)
        l2_to_l3[key] = {
            **errors,
            "forward_observed_relative_l2": float(
                np.linalg.norm(
                    arrays["forward_L2"]
                    - _observed_forward(fields, scored, transfer)
                )
                / np.linalg.norm(_observed_forward(fields, scored, transfer))
            ),
        }
    stable = (
        max(stability["column_relative_l2"]) < 5e-3
        and min(stability["column_cosines"]) > 0.99999
        and max(stability["rank3_principal_angles_degrees"]) < 2.0
    )
    report = {
        "schema_version": 1,
        "method": "full L3 shadow h qualification",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "steps": len(path),
        "scored": list(scored),
        "h_values": list(H_VALUES),
        "timings": timings,
        "geometries": {key: value for key, value in l3_geometries.items()},
        "stability_h_0015_vs_h_001": stability,
        "l2_to_l3": l2_to_l3,
        "claims": {
            "both_l3_complete": True,
            "h_0015_stable_against_h_001": stable,
            "identification_authorized": False,
            "p43_authorized": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    np.savez_compressed(
        output / "jacobians.npz",
        **{f"jacobian_h_{key.replace('.', 'p')}": value for key, value in matrices.items()},
    )
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["claims"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
