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

    def relative_amplitude(delta: float) -> float | None:
        # exp(delta)-1 is not useful once it overflows; retain the log-scale
        # one-sigma quantity in that case instead of emitting a warning.
        if delta > np.log(np.finfo(np.float64).max):
            return None
        return float(np.expm1(delta))

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
        "one_sigma_log_amplitude": (1.0 / singular).tolist(),
        "three_sigma_log_amplitude": (3.0 / singular).tolist(),
        "one_sigma_relative_amplitude": [
            relative_amplitude(float(value)) for value in (1.0 / singular)
        ],
        "three_sigma_relative_amplitude": [
            relative_amplitude(float(value)) for value in (3.0 / singular)
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

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(arguments.output), "cases": list(CASES)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
