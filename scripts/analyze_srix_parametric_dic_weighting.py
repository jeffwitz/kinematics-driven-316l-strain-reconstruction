#!/usr/bin/env python3
"""Apply the registered DIC transfer and whitener to archived SRIX Jacobians.

This is an offline post-processing tool.  It never calls a constitutive or
mechanical solver and never recomputes finite differences.  The transfer is
applied to each archived displacement-Jacobian column, followed by the
stationary whitener used by the registered repeated-frame qualification.
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

_WHITENING_SPEC = importlib.util.spec_from_file_location(
    "offline_dic_whitening", Path(__file__).resolve().parents[1]
    / "src/fem_inhouse/identification/dic_whitening.py"
)
if _WHITENING_SPEC is None or _WHITENING_SPEC.loader is None:
    raise ImportError("cannot load the registered DIC whitening implementation")
_WHITENING_MODULE = importlib.util.module_from_spec(_WHITENING_SPEC)
sys.modules[_WHITENING_SPEC.name] = _WHITENING_MODULE
_WHITENING_SPEC.loader.exec_module(_WHITENING_MODULE)
DICSpectralTransfer = _WHITENING_MODULE.DICSpectralTransfer
DICSpectralWhitener = _WHITENING_MODULE.DICSpectralWhitener

_COORDINATES_SPEC = importlib.util.spec_from_file_location(
    "offline_measurement_coordinates",
    Path(__file__).resolve().parents[1] / "src/fem_inhouse/measurement/coordinates.py",
)
if _COORDINATES_SPEC is None or _COORDINATES_SPEC.loader is None:
    raise ImportError("cannot load the registered measurement coordinate conversion")
_COORDINATES_MODULE = importlib.util.module_from_spec(_COORDINATES_SPEC)
sys.modules[_COORDINATES_SPEC.name] = _COORDINATES_MODULE
_COORDINATES_SPEC.loader.exec_module(_COORDINATES_MODULE)
image_flow_to_canonical = _COORDINATES_MODULE.image_flow_to_canonical

ROOT = Path(__file__).resolve().parents[1]
NOISE = (
    ROOT
    / "validation/reference_data/dic_uncertainty_propagation_p0043_v1"
    / "centred_repeat_flow_pixels.npy"
)
TRANSFER = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
EXPERIMENTAL_REPORT = (
    ROOT / "validation/reference_data/p0043_experimental_raw_femu_m20_v1/report.json"
)
EXPERIMENTAL_FIELDS = (
    ROOT / "validation/reference_data/p0043_experimental_raw_femu_m20_v1/fields.npz"
)
PARAMETER_ORDER = ["tau0_mpa", "R_mpa", "Q_mpa", "b"]
SCORED_STATE_INDICES = (3, 7, 11, 15, 19, 23, 27, 31)

CASES = {
    "synthetic_m20_truth": (
        ROOT / "validation/reference_data/p0043_synthetic_identification_v1/fields.npz",
        "jacobian_truth",
    ),
    "synthetic_m100_truth": (
        ROOT / "validation/reference_data/p0043_synthetic_scaleup_v1/fields.npz",
        "jacobian",
    ),
    "experimental_raw_m20_prior": (
        ROOT / "validation/reference_data/p0043_experimental_raw_femu_m20_v1/fields.npz",
        "prior_jacobian",
    ),
    "experimental_raw_m20_final": (
        ROOT / "validation/reference_data/p0043_experimental_raw_femu_m20_v1/fields.npz",
        "final_jacobian",
    ),
}


def _whitener(shape: tuple[int, int, int]) -> DICSpectralWhitener:
    support = np.ones(shape, dtype=np.float64)
    support[[0, -1], :, :] = 0.0
    support[:, [0, -1], :] = 0.0
    noise = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    canonical = image_flow_to_canonical(
        np.asarray(noise[:512, :512]), pixel_size_mm=0.00184
    )
    return DICSpectralWhitener.from_stationary_noise_field(
        canonical,
        target_shape=shape[:2],
        sample_count=256,
        seed=42,
        remove_spatial_mean=False,
        support_mask=support,
    )


def _transform(
    matrix: np.ndarray,
    transfer: DICSpectralTransfer,
    whitener: DICSpectralWhitener | None,
    *,
    wrap_free: bool,
) -> np.ndarray:
    rows = matrix.shape[0]
    states = 8
    components = 2
    nodes = rows // (states * components)
    side = round(nodes**0.5)
    if side * side != nodes or rows != states * side * side * components:
        raise ValueError(f"unexpected Jacobian shape {matrix.shape}; expected 8 square states")
    fields = matrix.reshape(states, side, side, components, matrix.shape[1])
    output = np.empty_like(fields)
    for state in range(states):
        for parameter in range(matrix.shape[1]):
            field = fields[state, ..., parameter]
            transformed = (
                transfer.apply_without_wrap(field)
                if wrap_free
                else transfer.apply(field)
            )
            if whitener is not None:
                transformed = whitener.apply(transformed)
            output[state, ..., parameter] = transformed
    return output.reshape(matrix.shape)


def _svd(matrix: np.ndarray) -> dict[str, object]:
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    vectors = vh.T
    normalised = singular / singular[0]
    cumulative = np.cumsum(singular**2) / np.sum(singular**2)

    def safe_exp(value: float) -> float | None:
        if value > np.log(np.finfo(np.float64).max):
            return None
        return float(np.exp(value))

    one_sigma = 1.0 / singular
    three_sigma = 3.0 / singular
    delta_log_theta_one = vectors * one_sigma[None, :]
    delta_log_theta_three = vectors * three_sigma[None, :]

    return {
        "shape": list(matrix.shape),
        "singular_values": singular.tolist(),
        "normalised_singular_values": normalised.tolist(),
        "cumulative_information": cumulative.tolist(),
        "modes_for_information": {
            str(level): int(np.searchsorted(cumulative, level) + 1)
            for level in (0.90, 0.95, 0.99)
        },
        "condition_number": float(singular[0] / singular[-1]),
        "delta_q_1sigma": one_sigma.tolist(),
        "delta_q_3sigma": three_sigma.tolist(),
        "delta_log_theta_1sigma": delta_log_theta_one.tolist(),
        "delta_log_theta_3sigma": delta_log_theta_three.tolist(),
        "parameter_factor_1sigma": [
            [safe_exp(float(value)) for value in delta_log_theta_one[:, mode]]
            for mode in range(vectors.shape[1])
        ],
        "parameter_factor_3sigma": [
            [safe_exp(float(value)) for value in delta_log_theta_three[:, mode]]
            for mode in range(vectors.shape[1])
        ],
        "rank_above_threshold": {
            str(level): int(np.count_nonzero(normalised >= level))
            for level in (1.0e-2, 1.0e-3, 1.0e-4)
        },
        "right_singular_vectors": vectors.tolist(),
    }


def _angles(first: np.ndarray, second: np.ndarray) -> dict[str, list[float]]:
    return {
        str(rank): np.degrees(subspace_angles(first[:, :rank], second[:, :rank])).tolist()
        for rank in (1, 2, 3)
    }


def _heldout_noise_projection(
    noise: np.ndarray,
    whitener: DICSpectralWhitener,
    left_vectors: np.ndarray,
    *,
    draws: int = 256,
    seed: int = 20260829,
) -> dict[str, object]:
    """Project spatially separated held-out noise windows onto FEMU modes."""

    rng = np.random.default_rng(seed)
    side = whitener.field_shape[0]
    states = 8
    spacing = side + 11
    origins = [
        (x, y)
        for x in range(512, noise.shape[0] - side + 1, spacing)
        for y in range(0, noise.shape[1] - side + 1, spacing)
    ]
    if len(origins) < states:
        raise ValueError("not enough held-out windows for eight scored states")
    projections = np.empty((draws, left_vectors.shape[1]), dtype=np.float64)
    norms = np.empty(draws, dtype=np.float64)
    component_values = np.empty((draws, states, side, side, 2), dtype=np.float64)
    for draw in range(draws):
        selected = rng.choice(len(origins), size=states, replace=False)
        stacked = np.empty((states, side, side, 2), dtype=np.float64)
        for state, index in enumerate(selected):
            x, y = origins[int(index)]
            canonical = image_flow_to_canonical(
                np.asarray(noise[x : x + side, y : y + side]),
                pixel_size_mm=0.00184,
            )
            stacked[state] = whitener.apply(canonical)
        component_values[draw] = stacked
        flattened = stacked.reshape(-1)
        projections[draw] = left_vectors.T @ flattened
        norms[draw] = np.linalg.norm(flattened)

    per_mode = {}
    for mode in range(projections.shape[1]):
        values = projections[:, mode]
        per_mode[str(mode + 1)] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "median_abs": float(np.median(np.abs(values))),
            "q95_abs": float(np.quantile(np.abs(values), 0.95)),
        }
    return {
        "draws": draws,
        "seed": seed,
        "calibration_exclusion": (
            "all held-out window origins have x >= 512; calibration crop is "
            "noise[:512,:512]"
        ),
        "window_spacing": spacing,
        "mode_projection_statistics": per_mode,
        "whitened_noise_norm": {
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms, ddof=1)),
            "median": float(np.median(norms)),
            "q95": float(np.quantile(norms, 0.95)),
        },
        "whitened_noise_component_statistics": {
            str(component): {
                "mean": float(np.mean(component_values[..., component])),
                "std": float(np.std(component_values[..., component], ddof=1)),
                "rms": float(np.sqrt(np.mean(component_values[..., component] ** 2))),
                "variance": float(np.var(component_values[..., component], ddof=1)),
            }
            for component in range(2)
        },
    }


def _prior_to_final_check(
    prior_matrix: np.ndarray,
    transfer: DICSpectralTransfer,
    whitener: DICSpectralWhitener,
) -> dict[str, object]:
    """Compare a local prior-to-final linear prediction with archived fields."""

    metadata = json.loads(EXPERIMENTAL_REPORT.read_text(encoding="utf-8"))
    prior_values = metadata["prior"]
    selected = next(
        start for start in metadata["starts"] if start["name"] == metadata["best_start"]
    )
    final_values = selected["identified"]
    prior = np.array([prior_values[key] for key in PARAMETER_ORDER], dtype=np.float64)
    final = np.array([final_values[key] for key in PARAMETER_ORDER], dtype=np.float64)
    delta_eta = np.log(final / prior)
    left, singular, right_transposed = np.linalg.svd(prior_matrix, full_matrices=False)
    q = right_transposed @ delta_eta
    predicted = singular * q

    with np.load(EXPERIMENTAL_FIELDS, allow_pickle=False) as archive:
        prior_displacement = np.asarray(archive["prior_displacement"], dtype=np.float64)
        best_displacement = np.asarray(archive["best_displacement"], dtype=np.float64)
    observed = []
    for state in SCORED_STATE_INDICES:
        field = best_displacement[state] - prior_displacement[state]
        observed.append(whitener.apply(transfer.apply_without_wrap(field)))
    observed_vector = np.asarray(observed, dtype=np.float64).reshape(-1)
    actual = left.T @ observed_vector
    return {
        "parameter_order": PARAMETER_ORDER,
        "prior": prior.tolist(),
        "final": final.tolist(),
        "delta_log_theta": delta_eta.tolist(),
        "modal_coordinates_q": q.tolist(),
        "predicted_modal_signal_sigma_q": predicted.tolist(),
        "archived_field_difference_modal_projection": actual.tolist(),
        "predicted_signal_norm": float(np.linalg.norm(predicted)),
        "archived_field_difference_norm": float(np.linalg.norm(observed_vector)),
        "modal_projection_difference_norm": float(np.linalg.norm(actual - predicted)),
        "scored_state_indices_zero_based": list(SCORED_STATE_INDICES),
        "interpretation": "local linearity check only; no identification validation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    transfer = DICSpectralTransfer.from_sinusoidal_csv(TRANSFER)
    report: dict[str, object] = {
        "schema_version": 2,
        "method": "offline archived displacement Jacobian post-processing",
        "no_forward_or_finite_difference": True,
        "transfer": {
            "source": str(TRANSFER.relative_to(ROOT)),
            "periodic_variant": "DICSpectralTransfer.apply",
            "wrap_free_variant": "DICSpectralTransfer.apply_without_wrap",
            "interpretation": "spectral DIC surrogates; not a new image-level DIC run",
        },
        "whitener": {
            "source": str(NOISE.relative_to(ROOT)),
            "source_shape": list(np.load(NOISE, mmap_mode="r", allow_pickle=False).shape),
            "source_dtype": str(np.load(NOISE, mmap_mode="r", allow_pickle=False).dtype),
            "source_sha256": hashlib.sha256(NOISE.read_bytes()).hexdigest(),
            "canonical_conversion": (
                "image_flow_to_canonical(noise[:512,:512], "
                "pixel_size_mm=0.00184)"
            ),
            "canonical_shape": [512, 512, 2],
            "canonical_dtype": "float64",
            "sample_count": 256,
            "seed": 42,
            "remove_spatial_mean": False,
            "support": "zero on one-node boundary, one in the interior",
        },
        "cases": {},
        "angles": {},
    }
    right_vectors: dict[str, dict[str, np.ndarray]] = {}
    whiteners: dict[int, DICSpectralWhitener] = {}
    wrap_free_full_matrices: dict[str, np.ndarray] = {}
    for name, (path, key) in CASES.items():
        with np.load(path, allow_pickle=False) as archive:
            raw = np.asarray(archive[key], dtype=np.float64)
        transfer_only = _transform(raw, transfer, None, wrap_free=False)
        side = round((raw.shape[0] / 16) ** 0.5)
        whitener = whiteners.setdefault(side, _whitener((side, side, 2)))
        full = _transform(raw, transfer, whitener, wrap_free=False)
        wrap_free_transfer = _transform(raw, transfer, None, wrap_free=True)
        wrap_free_full = _transform(raw, transfer, whitener, wrap_free=True)
        levels = {
            "raw_mechanical": _svd(raw),
            "dic_periodic_transfer": _svd(transfer_only),
            "dic_periodic_transfer_plus_spatial_whitening": _svd(full),
            "dic_wrap_free_transfer": _svd(wrap_free_transfer),
            "dic_wrap_free_transfer_plus_spatial_whitening": _svd(wrap_free_full),
        }
        report["cases"][name] = {  # type: ignore[index]
            "source": str(path.relative_to(ROOT)),
            "key": key,
            "parameter_order": ["tau0", "R", "Q", "b"],
            "coordinate_system": "log(theta)",
            "levels": levels,
        }
        right_vectors[name] = {
            level: np.asarray(values["right_singular_vectors"], dtype=np.float64)
            for level, values in levels.items()
        }
        wrap_free_full_matrices[name] = wrap_free_full

    for name, vectors in right_vectors.items():
        report["angles"][name] = {  # type: ignore[index]
            "raw_to_periodic_transfer": _angles(
                vectors["raw_mechanical"], vectors["dic_periodic_transfer"]
            ),
            "periodic_transfer_to_periodic_full": _angles(
                vectors["dic_periodic_transfer"],
                vectors["dic_periodic_transfer_plus_spatial_whitening"],
            ),
            "raw_to_periodic_full": _angles(
                vectors["raw_mechanical"],
                vectors["dic_periodic_transfer_plus_spatial_whitening"],
            ),
            "periodic_to_wrap_free_transfer": _angles(
                vectors["dic_periodic_transfer"],
                vectors["dic_wrap_free_transfer"],
            ),
            "periodic_full_to_wrap_free_full": _angles(
                vectors["dic_periodic_transfer_plus_spatial_whitening"],
                vectors["dic_wrap_free_transfer_plus_spatial_whitening"],
            ),
            "raw_to_wrap_free_full": _angles(
                vectors["raw_mechanical"],
                vectors["dic_wrap_free_transfer_plus_spatial_whitening"],
            ),
        }

    final_vectors = {
        name: vectors["dic_wrap_free_transfer_plus_spatial_whitening"]
        for name, vectors in right_vectors.items()
    }
    for first, second in (
        ("experimental_raw_m20_prior", "experimental_raw_m20_final"),
        ("synthetic_m20_truth", "experimental_raw_m20_final"),
        ("synthetic_m100_truth", "experimental_raw_m20_final"),
    ):
        report["angles"][f"{first}_vs_{second}"] = _angles(  # type: ignore[index]
            final_vectors[first], final_vectors[second]
        )

    report["angles"]["experimental_raw_m20_prior_vs_final_periodic_full"] = _angles(
        right_vectors["experimental_raw_m20_prior"][
            "dic_periodic_transfer_plus_spatial_whitening"
        ],
        right_vectors["experimental_raw_m20_final"][
            "dic_periodic_transfer_plus_spatial_whitening"
        ],
    )
    report["angles"]["experimental_raw_m20_prior_vs_final_wrap_free_full"] = _angles(
        right_vectors["experimental_raw_m20_prior"][
            "dic_wrap_free_transfer_plus_spatial_whitening"
        ],
        right_vectors["experimental_raw_m20_final"][
            "dic_wrap_free_transfer_plus_spatial_whitening"
        ],
    )

    # The left singular vectors are needed for the held-out noise projection.
    left_vectors, _, _ = np.linalg.svd(
        wrap_free_full_matrices["experimental_raw_m20_final"], full_matrices=False
    )
    noise = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    calibration = _heldout_noise_projection(
        noise,
        whiteners[21],
        left_vectors,
    )
    full_singular = np.linalg.svd(
        wrap_free_full_matrices["experimental_raw_m20_final"],
        compute_uv=False,
    )
    for mode, singular in enumerate(full_singular, start=1):
        statistics = calibration["mode_projection_statistics"][str(mode)]  # type: ignore[index]
        projection_std = float(statistics["std"])  # type: ignore[index]
        statistics["empirical_delta_q_1sigma"] = projection_std / float(singular)  # type: ignore[index]
        statistics["empirical_delta_q_3sigma"] = 3.0 * projection_std / float(singular)  # type: ignore[index]
    calibration["sigma_interpretation"] = (
        "nominal 1/sigma is not calibrated as one standard deviation because "
        "held-out modal projection std differs materially from one"
    )
    report["heldout_noise_calibration"] = calibration
    report["prior_to_final_local_check"] = _prior_to_final_check(
        wrap_free_full_matrices["experimental_raw_m20_prior"],
        transfer,
        whiteners[21],
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(arguments.output), "cases": list(CASES)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
