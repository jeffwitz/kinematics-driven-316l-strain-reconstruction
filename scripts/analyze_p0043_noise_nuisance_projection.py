#!/usr/bin/env python3
"""Offline nuisance-projection audit for the P43 DIC modal metric.

The script applies the same declared nuisance projection to repeat-frame
noise and archived displacement sensitivities.  It compares no projection
(N0), translation removal (N1), and affine removal (N2) in the exact order
wrap-free transfer -> nuisance projection -> registered corner whitener.
No mechanical solve or finite difference is performed.
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
PIXEL_SIZE_MM = 0.00184
SIDE = 21
STATES = 8
COMPONENTS = 2
SAMPLE_COUNT = 256
SEED = 42
PROJECTION_SEED = 20260829
ZONES = {
    "corner_adjacent_holdout": (0, 512, 512, 1024),
    "p43_solve": (1290, 1950, 780, 1390),
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


_WHITENING = _load_module(
    "offline_dic_whitening_nuisance_projection",
    ROOT / "src/fem_inhouse/identification/dic_whitening.py",
)
_COORDINATES = _load_module(
    "offline_measurement_coordinates_nuisance_projection",
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


def _origins(bounds: tuple[int, int, int, int], stride: int = 32) -> list[tuple[int, int]]:
    x0, x1, y0, y1 = bounds
    return [
        (x, y) for x in range(x0, x1 - SIDE + 1, stride) for y in range(y0, y1 - SIDE + 1, stride)
    ]


def _canonical(noise: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
    x, y = origin
    return image_flow_to_canonical(
        np.asarray(noise[x : x + SIDE, y : y + SIDE]), pixel_size_mm=PIXEL_SIZE_MM
    )


def _design(kind: str) -> np.ndarray:
    """Return the scalar nuisance basis on the retained interior support."""
    rows = np.linspace(-1.0, 1.0, SIDE, dtype=np.float64)
    columns = np.linspace(-1.0, 1.0, SIDE, dtype=np.float64)
    xx, yy = np.meshgrid(rows, columns, indexing="ij")
    retained = np.ones((SIDE, SIDE), dtype=bool)
    retained[[0, -1], :] = False
    retained[:, [0, -1]] = False
    coordinate = np.stack((np.ones_like(xx), xx, yy), axis=-1)
    terms = 1 if kind == "translation" else 3 if kind == "affine" else 0
    if terms == 0:
        return np.zeros((SIDE, SIDE, COMPONENTS, 0), dtype=np.float64)
    return coordinate[retained, :terms]


def _project(field: np.ndarray, kind: str) -> np.ndarray:
    if kind == "none":
        return np.asarray(field, dtype=np.float64)
    retained = np.ones((SIDE, SIDE), dtype=bool)
    retained[[0, -1], :] = False
    retained[:, [0, -1]] = False
    values = np.asarray(field, dtype=np.float64).copy()
    interior = values[retained]
    basis = _design(kind)
    gram = basis.T @ basis
    coefficients = np.linalg.solve(gram, basis.T @ interior)
    values[retained] = interior - basis @ coefficients
    values[~retained] = 0.0
    return values


def _corner_whitener(noise: np.ndarray) -> DICSpectralWhitener:
    calibration = image_flow_to_canonical(
        np.asarray(noise[:512, :512]), pixel_size_mm=PIXEL_SIZE_MM
    )
    return DICSpectralWhitener.from_stationary_noise_field(
        calibration,
        target_shape=(SIDE, SIDE),
        sample_count=SAMPLE_COUNT,
        seed=SEED,
        remove_spatial_mean=False,
        support_mask=_support(),
    )


def _transform_jacobian(
    jacobian: np.ndarray,
    transfer: DICSpectralTransfer,
    whitener: DICSpectralWhitener,
    kind: str,
) -> np.ndarray:
    expected_rows = STATES * SIDE * SIDE * COMPONENTS
    if jacobian.shape[0] != expected_rows:
        raise ValueError(f"unexpected Jacobian shape {jacobian.shape}")
    fields = jacobian.reshape(STATES, SIDE, SIDE, COMPONENTS, jacobian.shape[1])
    output = np.empty_like(fields)
    for state in range(STATES):
        for parameter in range(jacobian.shape[1]):
            transformed = transfer.apply_without_wrap(fields[state, ..., parameter])
            transformed = _project(transformed, kind)
            output[state, ..., parameter] = whitener.apply(transformed)
    return output.reshape(jacobian.shape)


def _svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, singular, right_transposed = np.linalg.svd(matrix, full_matrices=False)
    return left, singular, right_transposed.T


def _modal_stats(
    noise: np.ndarray,
    origins: list[tuple[int, int]],
    whitener: DICSpectralWhitener,
    left: np.ndarray,
    kind: str,
) -> dict[str, object]:
    rng = np.random.default_rng(PROJECTION_SEED)
    projections = np.empty((SAMPLE_COUNT, left.shape[1]), dtype=np.float64)
    for draw in range(SAMPLE_COUNT):
        selected = rng.choice(len(origins), STATES, replace=False)
        stacked = np.stack(
            [whitener.apply(_project(_canonical(noise, origins[int(i)]), kind)) for i in selected],
            axis=0,
        )
        projections[draw] = left.T @ stacked.reshape(-1)
    return {
        "draws": SAMPLE_COUNT,
        "mode_projection_statistics": {
            str(mode + 1): {
                "mean": float(np.mean(projections[:, mode])),
                "std": float(np.std(projections[:, mode], ddof=1)),
                "median_abs": float(np.median(np.abs(projections[:, mode]))),
                "q95_abs": float(np.quantile(np.abs(projections[:, mode]), 0.95)),
            }
            for mode in range(left.shape[1])
        },
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    noise = np.load(NOISE_PATH, mmap_mode="r", allow_pickle=False)
    if noise.shape != (3600, 3100, 2):
        raise ValueError(f"unexpected noise shape {noise.shape}")
    transfer = DICSpectralTransfer.from_sinusoidal_csv(TRANSFER_PATH)
    whitener = _corner_whitener(noise)
    with np.load(FIELDS_PATH, allow_pickle=False) as archive:
        jacobian = np.asarray(archive["final_jacobian"], dtype=np.float64)
    origins = {name: _origins(bounds) for name, bounds in ZONES.items()}
    results: dict[str, object] = {}
    variants = {"N0": "none", "N1": "translation", "N2": "affine"}
    reference_left: np.ndarray | None = None
    for label, kind in variants.items():
        transformed = _transform_jacobian(jacobian, transfer, whitener, kind)
        left, singular, right = _svd(transformed)
        if reference_left is None:
            reference_left = right
        results[label] = {
            "projection": kind,
            "basis_support": "interior 19x19 per displacement component; border zeroed",
            "singular_values": singular.tolist(),
            "normalised": (singular / singular[0]).tolist(),
            "condition_number": float(singular[0] / singular[-1]),
            "right_singular_vectors": right.tolist(),
            "angles_to_N0_deg": {
                str(rank): np.degrees(
                    subspace_angles(reference_left[:, :rank], right[:, :rank])
                ).tolist()
                for rank in (1, 2, 3)
            },
            "modal_projection": {
                zone: _modal_stats(noise, zone_origins, whitener, left, kind)
                for zone, zone_origins in origins.items()
            },
        }
    report = {
        "schema_version": 1,
        "method": (
            "N0/N1/N2: wrap-free DIC transfer -> nuisance projection -> registered corner whitening"
        ),
        "no_forward_or_finite_difference": True,
        "noise": {
            "source": str(NOISE_PATH.relative_to(ROOT)),
            "shape": list(noise.shape),
            "dtype": str(noise.dtype),
            "sha256": hashlib.sha256(NOISE_PATH.read_bytes()).hexdigest(),
            "pixel_size_mm": PIXEL_SIZE_MM,
        },
        "jacobian": {
            "source": str(FIELDS_PATH.relative_to(ROOT)),
            "key": "final_jacobian",
            "shape": list(jacobian.shape),
            "parameter_order": ["tau0_mpa", "R_mpa", "Q_mpa", "b"],
            "state_count": STATES,
        },
        "registered_whitener": {
            "calibration_bounds_axis0_axis1": [0, 512, 0, 512],
            "sample_count": SAMPLE_COUNT,
            "seed": SEED,
            "remove_spatial_mean": False,
            "support": "one-node border removed",
        },
        "zones": {name: list(bounds) for name, bounds in ZONES.items()},
        "variants": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
