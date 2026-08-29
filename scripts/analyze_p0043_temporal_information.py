#!/usr/bin/env python3
"""Offline temporal information analysis of archived P43 FEMU Jacobians.

The eight scored states are decomposed from the archived displacement
Jacobians.  Per-state and cumulative SVD/Fisher summaries are computed for
the raw displacement sensitivity and for the registered wrap-free DIC
surrogate.  This is algebra on archived arrays only: no forward solve or
finite-difference sensitivity is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles

ROOT = Path(__file__).resolve().parents[1]
NOISE_PATH = ROOT / "validation/reference_data/dic_uncertainty_propagation_p0043_v1/centred_repeat_flow_pixels.npy"
TRANSFER_PATH = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
FIELDS_PATH = ROOT / "validation/reference_data/p0043_experimental_raw_femu_m20_v1/fields.npz"
PIXEL_SIZE_MM = 0.00184
SIDE = 21
STATES = 8
COMPONENTS = 2
PARAMETERS = 4
SCORED_STATE_INDICES = (3, 7, 11, 15, 19, 23, 27, 31)
SAMPLE_COUNT = 256
SEED = 42


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_WHITENING = _load_module(
    "offline_dic_whitening_temporal_information",
    ROOT / "src/fem_inhouse/identification/dic_whitening.py",
)
_COORDINATES = _load_module(
    "offline_measurement_coordinates_temporal_information",
    ROOT / "src/fem_inhouse/measurement/coordinates.py",
)
DICSpectralTransfer = _WHITENING.DICSpectralTransfer
DICSpectralWhitener = _WHITENING.DICSpectralWhitener
image_flow_to_canonical = _COORDINATES.image_flow_to_canonical


def _support() -> np.ndarray:
    support = np.ones((SIDE, SIDE, COMPONENTS), dtype=np.float64)
    support[[0, -1], :, :] = 0.0
    support[:, [0, -1], :] = 0.0
    return support


def _corner_whitener(noise: np.ndarray) -> DICSpectralWhitener:
    corner = image_flow_to_canonical(
        np.asarray(noise[:512, :512]), pixel_size_mm=PIXEL_SIZE_MM
    )
    return DICSpectralWhitener.from_stationary_noise_field(
        corner,
        target_shape=(SIDE, SIDE),
        sample_count=SAMPLE_COUNT,
        seed=SEED,
        remove_spatial_mean=False,
        support_mask=_support(),
    )


def _state_matrices(
    jacobian: np.ndarray,
    transfer: DICSpectralTransfer,
    whitener: DICSpectralWhitener,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    expected = STATES * SIDE * SIDE * COMPONENTS
    if jacobian.shape != (expected, PARAMETERS):
        raise ValueError(f"unexpected Jacobian shape {jacobian.shape}")
    fields = jacobian.reshape(STATES, SIDE, SIDE, COMPONENTS, PARAMETERS)
    raw: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    for state in range(STATES):
        raw.append(fields[state].reshape(-1, PARAMETERS).copy())
        transformed = np.empty_like(fields[state])
        for parameter in range(PARAMETERS):
            transformed[..., parameter] = whitener.apply(
                transfer.apply_without_wrap(fields[state, ..., parameter])
            )
        observed.append(transformed.reshape(-1, PARAMETERS))
    return raw, observed


def _svd_summary(matrix: np.ndarray) -> dict[str, object]:
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    vectors = vh.T
    cumulative = np.cumsum(singular**2) / np.sum(singular**2)
    return {
        "shape": list(matrix.shape),
        "singular_values": singular.tolist(),
        "normalised": (singular / singular[0]).tolist(),
        "condition_number": float(singular[0] / singular[-1]),
        "cumulative_information": cumulative.tolist(),
        "right_singular_vectors": vectors.tolist(),
        "rank_above_threshold": {
            str(level): int(np.count_nonzero(singular / singular[0] >= level))
            for level in (1.0e-2, 1.0e-3, 1.0e-4)
        },
    }


def _fisher_summary(matrices: list[np.ndarray]) -> dict[str, object]:
    fisher = sum((matrix.T @ matrix for matrix in matrices), np.zeros((PARAMETERS, PARAMETERS)))
    eigenvalues, vectors = np.linalg.eigh(fisher)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    vectors = vectors[:, order]
    singular = np.sqrt(eigenvalues)
    cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    return {
        "singular_values": singular.tolist(),
        "normalised": (singular / singular[0]).tolist(),
        "condition_number": float(singular[0] / singular[-1]) if singular[-1] else None,
        "cumulative_information": cumulative.tolist(),
        "right_singular_vectors": vectors.tolist(),
        "trace": float(np.trace(fisher)),
        "smallest_eigenvalue": float(eigenvalues[-1]),
    }


def _angles(vectors: np.ndarray, reference: np.ndarray) -> dict[str, list[float]]:
    return {
        str(rank): np.degrees(subspace_angles(vectors[:, :rank], reference[:, :rank])).tolist()
        for rank in (1, 2, 3)
    }


def _subset_summary(matrices: list[np.ndarray], full_vectors: np.ndarray) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for size in range(1, STATES + 1):
        candidates = []
        for subset in itertools.combinations(range(STATES), size):
            summary = _fisher_summary([matrices[index] for index in subset])
            vectors = np.asarray(summary["right_singular_vectors"], dtype=np.float64)
            rank3_angle = float(np.max(np.degrees(subspace_angles(vectors[:, :3], full_vectors[:, :3]))))
            trace_fraction = float(summary["trace"] / _fisher_summary(matrices)["trace"])
            candidates.append((trace_fraction, rank3_angle, subset, summary))
        best_trace = max(candidates, key=lambda item: item[0])
        best_angle = min(candidates, key=lambda item: item[1])
        records.append({
            "subset_size": size,
            "best_trace_fraction": best_trace[0],
            "best_trace_subset_one_based": [index + 1 for index in best_trace[2]],
            "best_trace_rank3_angle_deg": best_trace[1],
            "best_rank3_angle_deg": best_angle[1],
            "best_rank3_angle_subset_one_based": [index + 1 for index in best_angle[2]],
        })
    return {
        "interpretation": "geometric subset summaries; no temporal covariance or statistical optimality claim",
        "by_subset_size": records,
    }


def _analyse_family(matrices: list[np.ndarray]) -> dict[str, object]:
    per_state = [_svd_summary(matrix) for matrix in matrices]
    cumulative: list[dict[str, object]] = []
    current: list[np.ndarray] = []
    full = _fisher_summary(matrices)
    full_vectors = np.asarray(full["right_singular_vectors"], dtype=np.float64)
    for state, matrix in enumerate(matrices):
        current.append(matrix)
        summary = _fisher_summary(current)
        vectors = np.asarray(summary["right_singular_vectors"], dtype=np.float64)
        summary["through_state"] = state + 1
        summary["source_scored_state_index"] = SCORED_STATE_INDICES[state]
        summary["angles_to_full_cumulative_subspace_deg"] = _angles(vectors, full_vectors)
        cumulative.append(summary)
    return {
        "per_state": per_state,
        "cumulative": cumulative,
        "full": full,
        "subset_summary": _subset_summary(matrices, full_vectors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    noise = np.load(NOISE_PATH, mmap_mode="r", allow_pickle=False)
    transfer = DICSpectralTransfer.from_sinusoidal_csv(TRANSFER_PATH)
    whitener = _corner_whitener(noise)
    with np.load(FIELDS_PATH, allow_pickle=False) as archive:
        prior = np.asarray(archive["prior_jacobian"], dtype=np.float64)
        final = np.asarray(archive["final_jacobian"], dtype=np.float64)
    prior_raw, prior_observed = _state_matrices(prior, transfer, whitener)
    final_raw, final_observed = _state_matrices(final, transfer, whitener)
    report = {
        "schema_version": 1,
        "method": "per-state and cumulative SVD/Fisher analysis of archived P43 FEMU Jacobians",
        "no_forward_or_finite_difference": True,
        "jacobian": {
            "source": str(FIELDS_PATH.relative_to(ROOT)),
            "keys": ["prior_jacobian", "final_jacobian"],
            "shape": list(final.shape),
            "state_count": STATES,
            "scored_state_indices_zero_based": list(SCORED_STATE_INDICES),
            "parameter_order": ["tau0_mpa", "R_mpa", "Q_mpa", "b"],
        },
        "noise_and_observation": {
            "noise_source": str(NOISE_PATH.relative_to(ROOT)),
            "noise_sha256": hashlib.sha256(NOISE_PATH.read_bytes()).hexdigest(),
            "transfer": "DICSpectralTransfer.apply_without_wrap",
            "whitener": "registered corner noise[:512,:512], sample_count=256, seed=42",
        },
        "prior": {
            "raw_mechanical": _analyse_family(prior_raw),
            "wrap_free_dicom_surrogate": _analyse_family(prior_observed),
        },
        "final": {
            "raw_mechanical": _analyse_family(final_raw),
            "wrap_free_dicom_surrogate": _analyse_family(final_observed),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
