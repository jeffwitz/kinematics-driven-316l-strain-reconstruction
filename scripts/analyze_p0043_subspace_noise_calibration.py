#!/usr/bin/env python3
"""Calibrate repeat-frame noise in the affine-cleaned SRIX sensitivity subspace.

The calculation uses archived noise and final/prior Jacobians only.  It avoids
the registered full spatial whitener and estimates a 3x3 covariance in the
rank-3 N2 sensitivity subspace.  T0 uses distinct spatial windows per scored
state; T1 repeats one window across states as a common-mode extreme.
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
NOISE_PATH = ROOT / (
    "validation/reference_data/dic_uncertainty_propagation_p0043_v1/centred_repeat_flow_pixels.npy"
)
TRANSFER_PATH = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
FIELDS_PATH = ROOT / "validation/reference_data/p0043_experimental_raw_femu_m20_v1/fields.npz"
PARAMETERS_REPORT = (
    ROOT / "validation/reference_data/p0043_experimental_raw_femu_m20_v1/report.json"
)
PIXEL_SIZE_MM = 0.00184
SIDE = 21
STATES = 8
COMPONENTS = 2
PARAMETERS = 4
SAMPLE_COUNT = 256
SEED = 42
PROJECTION_SEED = 20260829
PARAMETER_ORDER = ["tau0_mpa", "R_mpa", "Q_mpa", "b"]
ZONES = {
    "p43_solve_excluding_core": (1290, 1950, 780, 1390),
    "p43_core": (1440, 1800, 930, 1240),
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PROJECTION = _load_module(
    "offline_nuisance_projection_subspace_noise",
    ROOT / "scripts/analyze_p0043_noise_nuisance_projection.py",
)
DICSpectralTransfer = _PROJECTION.DICSpectralTransfer
image_flow_to_canonical = _PROJECTION.image_flow_to_canonical


def _origins(
    bounds: tuple[int, int, int, int],
    stride: int = 32,
    exclude: tuple[int, int, int, int] | None = None,
) -> list[tuple[int, int]]:
    x0, x1, y0, y1 = bounds
    origins = []
    for x in range(x0, x1 - SIDE + 1, stride):
        for y in range(y0, y1 - SIDE + 1, stride):
            if exclude is not None:
                ex0, ex1, ey0, ey1 = exclude
                if x < ex1 and x + SIDE > ex0 and y < ey1 and y + SIDE > ey0:
                    continue
            origins.append((x, y))
    return origins


def _canonical(noise: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
    x, y = origin
    return image_flow_to_canonical(
        np.asarray(noise[x : x + SIDE, y : y + SIDE]), pixel_size_mm=PIXEL_SIZE_MM
    )


def _project_affine(field: np.ndarray) -> np.ndarray:
    return _PROJECTION._project(field, "affine")


def _sensitivity_matrix(jacobian: np.ndarray, transfer: DICSpectralTransfer) -> np.ndarray:
    fields = jacobian.reshape(STATES, SIDE, SIDE, COMPONENTS, PARAMETERS)
    transformed = np.empty_like(fields)
    for state in range(STATES):
        for parameter in range(PARAMETERS):
            value = transfer.apply_without_wrap(fields[state, ..., parameter])
            transformed[state, ..., parameter] = _project_affine(value)
    return transformed.reshape(STATES * SIDE * SIDE * COMPONENTS, PARAMETERS)


def _sensitivity_states(jacobian: np.ndarray, transfer: DICSpectralTransfer) -> list[np.ndarray]:
    full = _sensitivity_matrix(jacobian, transfer).reshape(STATES, -1, PARAMETERS)
    return [full[state] for state in range(STATES)]


def _subspace(matrix: np.ndarray, rank: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, singular, right_transposed = np.linalg.svd(matrix, full_matrices=False)
    return left[:, :rank], singular[:rank], right_transposed.T[:, :rank]


def _stack_noise(
    noise: np.ndarray,
    origins: list[tuple[int, int]],
    count: int,
    temporal: str,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = rng.choice(len(origins), count if temporal == "T0" else 1, replace=False)
    if temporal == "T1":
        selected = np.repeat(selected, count)
    return np.stack(
        [_project_affine(_canonical(noise, origins[int(index)])) for index in selected], axis=0
    ).reshape(-1)


def _modal_samples(
    noise: np.ndarray,
    origins: list[tuple[int, int]],
    basis: np.ndarray,
    state_count: int,
    temporal: str,
    draws: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, basis.shape[1]), dtype=np.float64)
    for draw in range(draws):
        stacked = _stack_noise(noise, origins, state_count, temporal, rng)
        samples[draw] = basis.T @ stacked
    return samples


def _inverse_sqrt(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(covariance)
    floor = max(float(np.max(values)) * 1.0e-12, np.finfo(float).eps)
    clipped = np.maximum(values, floor)
    transform = (vectors * (1.0 / np.sqrt(clipped))) @ vectors.T
    return transform, values


def _calibration(training: np.ndarray, validation: np.ndarray) -> dict[str, object]:
    mean = training.mean(axis=0)
    covariance = np.cov(training, rowvar=False, ddof=1)
    inverse_sqrt, train_eigenvalues = _inverse_sqrt(covariance)
    whitened = (validation - mean) @ inverse_sqrt.T
    covariance_white = np.cov(whitened, rowvar=False, ddof=1)
    eigenvalues = np.linalg.eigvalsh(covariance_white)
    return {
        "training_draws": int(training.shape[0]),
        "validation_draws": int(validation.shape[0]),
        "training_mean": mean.tolist(),
        "training_covariance": covariance.tolist(),
        "training_covariance_eigenvalues": train_eigenvalues.tolist(),
        "validation_whitened_mean": whitened.mean(axis=0).tolist(),
        "validation_whitened_std": whitened.std(axis=0, ddof=1).tolist(),
        "validation_whitened_covariance": covariance_white.tolist(),
        "validation_covariance_eigenvalues": eigenvalues.tolist(),
        "mean_norm": float(np.linalg.norm(whitened.mean(axis=0))),
        "covariance_minus_identity_frobenius": float(np.linalg.norm(covariance_white - np.eye(3))),
        "min_max_covariance_eigenvalue": [float(eigenvalues[0]), float(eigenvalues[-1])],
    }


def _parametric_svd(
    matrix: np.ndarray,
    basis: np.ndarray,
    covariance: np.ndarray,
    geometric_vectors: np.ndarray,
    delta_log_theta: np.ndarray,
) -> dict[str, object]:
    inverse_sqrt, _ = _inverse_sqrt(covariance)
    projected = basis.T @ matrix
    scaled = inverse_sqrt @ projected
    _, singular, right_transposed = np.linalg.svd(scaled, full_matrices=False)
    right = right_transposed.T
    return {
        "status": "conditional_on_training_covariance_and_temporal_surrogate",
        "singular_values": singular.tolist(),
        "normalised": (singular / singular[0]).tolist(),
        "right_singular_vectors": right.tolist(),
        "angles_to_geometric_deg": {
            str(rank): np.degrees(
                subspace_angles(geometric_vectors[:, :rank], right[:, :rank])
            ).tolist()
            for rank in (1, 2, 3)
        },
        "delta_q_1sigma": (1.0 / singular).tolist(),
        "delta_q_3sigma": (3.0 / singular).tolist(),
        "local_signal_scale_delta_eta": (scaled @ delta_log_theta).tolist(),
    }


def _conclusion(families: dict[str, dict[str, object]]) -> dict[str, object]:
    """Summarise the conservative decision without inventing a hard gate."""
    full8 = families["full8"]["temporal_surrogates"]
    t0 = full8["T0_independent_windows"]["parametric_svd"]["singular_values"]
    t1 = full8["T1_common_window"]["parametric_svd"]["singular_values"]
    temporal_ratio = (np.asarray(t1) / np.asarray(t0)).tolist()
    return {
        "verdict": "B",
        "label": (
            "spatial covariance is provisionally usable, but temporal correlation "
            "remains the absolute-scale blocker"
        ),
        "spatial_validation": (
            "P43-core and disjoint solve-neighbourhood validation have small modal "
            "means and O(1) covariance eigenvalues, especially for T0; this supports a "
            "local 3x3 spatial surrogate rather than the corner as the sole calibration."
        ),
        "temporal_bracket": (
            "T0 independent-window and T1 common-window assumptions materially change "
            "the conditional singular-value scale, so absolute detectability is not "
            "qualified as an experimental likelihood."
        ),
        "t1_to_t0_singular_value_ratio_full8": temporal_ratio,
        "rank_statement": (
            "The rank-3 sensitivity geometry remains the working search subspace; "
            "the absolute experimentally detectable rank is not fixed."
        ),
        "next_step": "qualify temporal covariance or retain relative/robust objectives before FEMU",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    noise = np.load(NOISE_PATH, mmap_mode="r", allow_pickle=False)
    transfer = DICSpectralTransfer.from_sinusoidal_csv(TRANSFER_PATH)
    with np.load(FIELDS_PATH, allow_pickle=False) as archive:
        final_jacobian = np.asarray(archive["final_jacobian"], dtype=np.float64)
    solve_origins = _origins(ZONES["p43_solve_excluding_core"], exclude=ZONES["p43_core"])
    core_origins = _origins(ZONES["p43_core"])
    split = 1620
    split_train = _origins((1290, split, 780, 1390))
    split_validation = _origins((split, 1950, 780, 1390))
    geometric_states = _sensitivity_states(final_jacobian, transfer)
    metadata = json.loads(PARAMETERS_REPORT.read_text(encoding="utf-8"))
    prior_parameters = np.array(
        [metadata["prior"][key] for key in PARAMETER_ORDER], dtype=np.float64
    )
    best_start = next(
        start for start in metadata["starts"] if start["name"] == metadata["best_start"]
    )
    final_parameters = np.array(
        [best_start["identified"][key] for key in PARAMETER_ORDER], dtype=np.float64
    )
    delta_log_theta = np.log(final_parameters / prior_parameters)
    families: dict[str, dict[str, object]] = {}
    for family, states in (("full8", geometric_states), ("late3", geometric_states[5:])):
        state_count = len(states)
        matrix = np.vstack(states)
        basis, singular, geometric_vectors = _subspace(matrix)
        full_geometric_singular_values = np.linalg.svd(matrix, compute_uv=False)
        family_results: dict[str, object] = {
            "state_count": state_count,
            "state_selection_one_based": list(range(1, 9)) if family == "full8" else [6, 7, 8],
            "sensitivity_shape": list(matrix.shape),
            "geometric_singular_values": singular.tolist(),
            "geometric_singular_values_full_rank4_diagnostic": (
                full_geometric_singular_values.tolist()
            ),
            "geometric_right_singular_vectors": geometric_vectors.tolist(),
            "temporal_surrogates": {},
        }
        for temporal in ("T0_independent_windows", "T1_common_window"):
            temporal_code = "T0" if temporal.startswith("T0") else "T1"
            train = _modal_samples(
                noise,
                solve_origins,
                basis,
                state_count,
                temporal_code,
                SAMPLE_COUNT,
                PROJECTION_SEED,
            )
            validation_core = _modal_samples(
                noise,
                core_origins,
                basis,
                state_count,
                temporal_code,
                SAMPLE_COUNT,
                PROJECTION_SEED + 1,
            )
            validation_split = _modal_samples(
                noise,
                split_validation,
                basis,
                state_count,
                temporal_code,
                SAMPLE_COUNT,
                PROJECTION_SEED + 2,
            )
            family_results["temporal_surrogates"][temporal] = {
                "calibration_zone": "p43_solve excluding p43_core",
                "validation_zone": "p43_core",
                "calibration": _calibration(train, validation_core),
                "independent_split_validation": _calibration(
                    _modal_samples(
                        noise,
                        split_train,
                        basis,
                        state_count,
                        temporal_code,
                        SAMPLE_COUNT,
                        PROJECTION_SEED + 3,
                    ),
                    validation_split,
                ),
            }
            calibration = family_results["temporal_surrogates"][temporal]["calibration"]
            family_results["temporal_surrogates"][temporal]["parametric_svd"] = _parametric_svd(
                matrix,
                basis,
                np.asarray(calibration["training_covariance"], dtype=np.float64),
                geometric_vectors,
                delta_log_theta,
            )
        families[family] = family_results
    report = {
        "schema_version": 1,
        "method": "3x3 N2 affine-cleaned sensitivity-subspace noise calibration",
        "no_forward_or_finite_difference": True,
        "noise_source": str(NOISE_PATH.relative_to(ROOT)),
        "noise_sha256": hashlib.sha256(NOISE_PATH.read_bytes()).hexdigest(),
        "jacobian_source": str(FIELDS_PATH.relative_to(ROOT)),
        "parameter_order": PARAMETER_ORDER,
        "observation_chain": (
            "archived Jacobian -> wrap-free transfer -> N2 affine projection -> rank-3 basis"
        ),
        "noise_chain": (
            "repeat-frame window -> canonical conversion -> N2 affine "
            "projection -> state stack -> Q^T"
        ),
        "calibration": {
            "primary_train_origins": len(solve_origins),
            "primary_validation_origins": len(core_origins),
            "split_train_origins": len(split_train),
            "split_validation_origins": len(split_validation),
            "stride": 32,
            "draws": SAMPLE_COUNT,
            "calibration_excludes_core": True,
        },
        "families": families,
    }
    report["conclusion"] = _conclusion(families)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
