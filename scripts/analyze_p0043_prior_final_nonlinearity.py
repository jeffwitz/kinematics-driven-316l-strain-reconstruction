#!/usr/bin/env python3
"""Offline prior-to-final, frame-resolved linearity check for P43.

This compares archived prior/final displacement fields with the archived
prior Jacobian in raw and wrap-free DIC-surrogate coordinates.  It is a local
post-processing diagnostic only; no forward or finite-difference calculation
is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles

ROOT = Path(__file__).resolve().parents[1]
NOISE = ROOT / "validation/reference_data/dic_uncertainty_propagation_p0043_v1/centred_repeat_flow_pixels.npy"
TRANSFER = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
REPORT = ROOT / "validation/reference_data/p0043_experimental_raw_femu_m20_v1/report.json"
FIELDS = ROOT / "validation/reference_data/p0043_experimental_raw_femu_m20_v1/fields.npz"
PIXEL_SIZE_MM = 0.00184
SIDE = 21
STATES = 8
COMPONENTS = 2
PARAMETER_ORDER = ("tau0_mpa", "R_mpa", "Q_mpa", "b")
SCORED_STATE_INDICES = (3, 7, 11, 15, 19, 23, 27, 31)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_WHITENING = _load_module("offline_dic_whitening_prior_final", ROOT / "src/fem_inhouse/identification/dic_whitening.py")
_COORDINATES = _load_module("offline_coordinates_prior_final", ROOT / "src/fem_inhouse/measurement/coordinates.py")
DICSpectralTransfer = _WHITENING.DICSpectralTransfer
DICSpectralWhitener = _WHITENING.DICSpectralWhitener
image_flow_to_canonical = _COORDINATES.image_flow_to_canonical


def _support() -> np.ndarray:
    support = np.ones((SIDE, SIDE, COMPONENTS), dtype=np.float64)
    support[[0, -1], :, :] = 0.0
    support[:, [0, -1], :] = 0.0
    return support


def _whitener(noise: np.ndarray) -> DICSpectralWhitener:
    corner = image_flow_to_canonical(np.asarray(noise[:512, :512]), pixel_size_mm=PIXEL_SIZE_MM)
    return DICSpectralWhitener.from_stationary_noise_field(
        corner, target_shape=(SIDE, SIDE), sample_count=256, seed=42,
        remove_spatial_mean=False, support_mask=_support(),
    )


def _transform(jacobian: np.ndarray, transfer: DICSpectralTransfer,
               whitener: DICSpectralWhitener | None) -> np.ndarray:
    fields = jacobian.reshape(STATES, SIDE, SIDE, COMPONENTS, len(PARAMETER_ORDER))
    transformed = np.empty_like(fields)
    for state in range(STATES):
        for parameter in range(len(PARAMETER_ORDER)):
            value = transfer.apply_without_wrap(fields[state, ..., parameter])
            transformed[state, ..., parameter] = value if whitener is None else whitener.apply(value)
    return transformed.reshape(jacobian.shape)


def _transform_displacement(fields: np.ndarray, transfer: DICSpectralTransfer,
                            whitener: DICSpectralWhitener) -> np.ndarray:
    output = np.empty_like(fields, dtype=np.float64)
    for state in range(STATES):
        output[state] = whitener.apply(transfer.apply_without_wrap(fields[state]))
    return output


def _state_svd_angles(prior: np.ndarray, final: np.ndarray) -> list[dict[str, object]]:
    records = []
    for state in range(STATES):
        _, _, prior_vh = np.linalg.svd(prior[state], full_matrices=False)
        _, _, final_vh = np.linalg.svd(final[state], full_matrices=False)
        prior_v = prior_vh.T
        final_v = final_vh.T
        records.append({
            "state": state + 1,
            "source_scored_state_index": SCORED_STATE_INDICES[state],
            "prior_normalised": (np.linalg.svd(prior[state], compute_uv=False) / np.linalg.svd(prior[state], compute_uv=False)[0]).tolist(),
            "final_normalised": (np.linalg.svd(final[state], compute_uv=False) / np.linalg.svd(final[state], compute_uv=False)[0]).tolist(),
            "angles_prior_to_final_deg": {
                str(rank): np.degrees(subspace_angles(prior_v[:, :rank], final_v[:, :rank])).tolist()
                for rank in (1, 2, 3)
            },
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    noise = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    transfer = DICSpectralTransfer.from_sinusoidal_csv(TRANSFER)
    whitener = _whitener(noise)
    with np.load(FIELDS, allow_pickle=False) as archive:
        prior_jacobian = np.asarray(archive["prior_jacobian"], dtype=np.float64)
        final_jacobian = np.asarray(archive["final_jacobian"], dtype=np.float64)
        prior_displacement = np.asarray(archive["prior_displacement"], dtype=np.float64)[list(SCORED_STATE_INDICES)]
        best_displacement = np.asarray(archive["best_displacement"], dtype=np.float64)[list(SCORED_STATE_INDICES)]
    metadata = json.loads(REPORT.read_text(encoding="utf-8"))
    prior_parameters = np.array([metadata["prior"][key] for key in PARAMETER_ORDER], dtype=np.float64)
    best_start = next(start for start in metadata["starts"] if start["name"] == metadata["best_start"])
    final_parameters = np.array([best_start["identified"][key] for key in PARAMETER_ORDER], dtype=np.float64)
    delta_log_theta = np.log(final_parameters / prior_parameters)
    prior_raw = prior_jacobian.reshape(STATES, -1, len(PARAMETER_ORDER))
    prior_observed = _transform(prior_jacobian, transfer, whitener).reshape(STATES, -1, len(PARAMETER_ORDER))
    final_raw = final_jacobian.reshape(STATES, -1, len(PARAMETER_ORDER))
    final_observed = _transform(final_jacobian, transfer, whitener).reshape(STATES, -1, len(PARAMETER_ORDER))
    observed_delta = _transform_displacement(best_displacement - prior_displacement, transfer, whitener)
    records = []
    for state in range(STATES):
        pred_raw = prior_raw[state] @ delta_log_theta
        pred_observed = prior_observed[state] @ delta_log_theta
        actual = observed_delta[state].reshape(-1)
        predicted = pred_observed
        denominator = np.linalg.norm(actual)
        records.append({
            "state": state + 1,
            "source_scored_state_index": SCORED_STATE_INDICES[state],
            "raw_predicted_norm": float(np.linalg.norm(pred_raw)),
            "observed_predicted_norm": float(np.linalg.norm(predicted)),
            "archived_observed_difference_norm": float(denominator),
            "observed_relative_linearisation_error": float(np.linalg.norm(predicted - actual) / denominator) if denominator else None,
            "observed_correlation": float(np.corrcoef(predicted, actual)[0, 1]) if np.std(predicted) and np.std(actual) else None,
        })
    report = {
        "schema_version": 1,
        "method": "frame-resolved prior-to-final local linearity check",
        "no_forward_or_finite_difference": True,
        "sources": {
            "fields": str(FIELDS.relative_to(ROOT)),
            "report": str(REPORT.relative_to(ROOT)),
            "noise": str(NOISE.relative_to(ROOT)),
            "noise_sha256": hashlib.sha256(NOISE.read_bytes()).hexdigest(),
        },
        "parameter_order": list(PARAMETER_ORDER),
        "prior_parameters": prior_parameters.tolist(),
        "final_parameters": final_parameters.tolist(),
        "delta_log_theta": delta_log_theta.tolist(),
        "scored_state_indices_zero_based": list(SCORED_STATE_INDICES),
        "frame_linearity": records,
        "per_frame_svd_angles": {
            "raw_mechanical": _state_svd_angles(prior_raw, final_raw),
            "wrap_free_dicom_surrogate": _state_svd_angles(prior_observed, final_observed),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
