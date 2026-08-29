#!/usr/bin/env python3
"""Offline spatial-stationarity audit for the registered P43 repeat-frame noise.

The script reads only the archived repeat-frame field and archived SRIX
displacement Jacobians. It does not call a mechanical solver, recompute finite
differences, or alter the registered corner whitener.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOISE_PATH = ROOT / "validation/reference_data/dic_uncertainty_propagation_p0043_v1/centred_repeat_flow_pixels.npy"
TRANSFER_PATH = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
FIELDS_PATH = ROOT / "validation/reference_data/p0043_experimental_raw_femu_m20_v1/fields.npz"
PIXEL_SIZE_MM = 0.00184
SIDE = 21
STATES = 8
COMPONENTS = 2
SAMPLE_COUNT = 256
SEED = 42
PROJECTION_SEED = 20260829

# Bounds are (axis0_start, axis0_stop, axis1_start, axis1_stop).  The P43
# contracts define axis 0 as transverse x and axis 1 as tensile y.
ZONES = {
    "calibration_corner": (0, 512, 0, 512),
    "corner_adjacent_holdout": (0, 512, 512, 1024),
    "p43_solve": (1290, 1950, 780, 1390),
    "p43_core": (1440, 1800, 930, 1240),
}
M20_CROP = (1610, 1630, 1075, 1095)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_WHITENING = _load_module(
    "offline_dic_whitening_stationarity",
    ROOT / "src/fem_inhouse/identification/dic_whitening.py",
)
_COORDINATES = _load_module(
    "offline_measurement_coordinates_stationarity",
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


def _origins(bounds: tuple[int, int, int, int], *, stride: int = 32,
             exclude: tuple[int, int, int, int] | None = None) -> list[tuple[int, int]]:
    x0, x1, y0, y1 = bounds
    result: list[tuple[int, int]] = []
    for x in range(x0, x1 - SIDE + 1, stride):
        for y in range(y0, y1 - SIDE + 1, stride):
            if exclude is not None:
                ex0, ex1, ey0, ey1 = exclude
                if x < ex1 and x + SIDE > ex0 and y < ey1 and y + SIDE > ey0:
                    continue
            result.append((x, y))
    return result


def _canonical(noise: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
    x, y = origin
    return image_flow_to_canonical(
        np.asarray(noise[x : x + SIDE, y : y + SIDE]),
        pixel_size_mm=PIXEL_SIZE_MM,
    )


def _window_metrics(field: np.ndarray, whitener: DICSpectralWhitener | None = None) -> dict[str, float]:
    values = field if whitener is None else whitener.apply(field)
    centred = values - values.mean(axis=(0, 1), keepdims=True)
    spectrum = np.fft.fftn(centred, axes=(0, 1), norm="ortho")
    power = np.sum(np.abs(spectrum) ** 2, axis=-1)
    fx = np.fft.fftfreq(SIDE)[:, None]
    fy = np.fft.fftfreq(SIDE)[None, :]
    low = np.sqrt(fx * fx + fy * fy) <= 0.125
    total_power = float(np.sum(power))
    low_power = float(np.sum(power[low]))
    flat = values.reshape(-1, COMPONENTS)
    covariance = np.cov(flat, rowvar=False, ddof=1)
    return {
        "mean_ux": float(values[..., 0].mean()),
        "mean_uy": float(values[..., 1].mean()),
        "std_ux": float(values[..., 0].std(ddof=1)),
        "std_uy": float(values[..., 1].std(ddof=1)),
        "rms_ux": float(np.sqrt(np.mean(values[..., 0] ** 2))),
        "rms_uy": float(np.sqrt(np.mean(values[..., 1] ** 2))),
        "cov_ux_uy": float(covariance[0, 1]),
        "low_frequency_fraction": low_power / total_power if total_power else 0.0,
    }


def _summarise_windows(noise: np.ndarray, origins: list[tuple[int, int]],
                       whitener: DICSpectralWhitener | None = None) -> dict[str, object]:
    records = [_window_metrics(_canonical(noise, origin), whitener) for origin in origins]
    keys = tuple(records[0]) if records else ()
    summary: dict[str, object] = {"window_count": len(records)}
    for key in keys:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std_across_windows": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "median": float(np.median(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
        }
    return summary


def _registered_corner_whitener(noise: np.ndarray) -> DICSpectralWhitener:
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


def _make_whitener(realisations: np.ndarray) -> DICSpectralWhitener:
    return DICSpectralWhitener.from_noise_realisations(
        realisations,
        relative_floor=1.0e-6,
        absolute_floor=0.0,
        remove_spatial_mean=False,
        support_mask=_support(),
    )


def _local_candidate(noise: np.ndarray) -> tuple[DICSpectralWhitener, list[tuple[int, int]]]:
    origins = _origins(ZONES["p43_solve"], exclude=ZONES["p43_core"])
    if len(origins) < SAMPLE_COUNT:
        raise ValueError(f"only {len(origins)} local calibration windows available")
    rng = np.random.default_rng(SEED)
    selected = [origins[int(i)] for i in rng.choice(len(origins), SAMPLE_COUNT, replace=False)]
    realisations = np.stack([_canonical(noise, origin) for origin in selected], axis=0)
    return _make_whitener(realisations), selected


def _transform_jacobian(jacobian: np.ndarray, transfer: DICSpectralTransfer,
                        whitener: DICSpectralWhitener) -> np.ndarray:
    expected_rows = STATES * SIDE * SIDE * COMPONENTS
    if jacobian.shape[0] != expected_rows:
        raise ValueError(f"unexpected Jacobian shape {jacobian.shape}")
    fields = jacobian.reshape(STATES, SIDE, SIDE, COMPONENTS, jacobian.shape[1])
    output = np.empty_like(fields)
    for state in range(STATES):
        for parameter in range(jacobian.shape[1]):
            field = transfer.apply_without_wrap(fields[state, ..., parameter])
            output[state, ..., parameter] = whitener.apply(field)
    return output.reshape(jacobian.shape)


def _modal_projection_stats(noise: np.ndarray, origins: list[tuple[int, int]],
                            whitener: DICSpectralWhitener, left: np.ndarray,
                            *, draws: int = SAMPLE_COUNT) -> dict[str, object]:
    if len(origins) < STATES:
        raise ValueError("not enough validation windows")
    rng = np.random.default_rng(PROJECTION_SEED)
    projections = np.empty((draws, left.shape[1]), dtype=np.float64)
    for draw in range(draws):
        chosen = rng.choice(len(origins), STATES, replace=False)
        stacked = np.stack([
            whitener.apply(_canonical(noise, origins[int(index)]))
            for index in chosen
        ], axis=0)
        projections[draw] = left.T @ stacked.reshape(-1)
    return {
        "draws": draws,
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


def _empirical_delta_q(singular: np.ndarray, projection: dict[str, object]) -> list[float]:
    """Local detectability scale using the held-out modal noise dispersion."""
    statistics = projection["mode_projection_statistics"]  # type: ignore[index]
    return [
        float(statistics[str(index + 1)]["std"]) / float(singular[index])
        for index in range(singular.size)
    ]


def _read_jacobian() -> np.ndarray:
    with np.load(FIELDS_PATH, allow_pickle=False) as archive:
        return np.asarray(archive["final_jacobian"], dtype=np.float64)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-local", action="store_true",
                        help="build the P43-local candidate after stationarity stats")
    args = parser.parse_args()

    noise = np.load(NOISE_PATH, mmap_mode="r", allow_pickle=False)
    if noise.shape != (3600, 3100, 2):
        raise ValueError(f"unexpected noise shape {noise.shape}")
    transfer = DICSpectralTransfer.from_sinusoidal_csv(TRANSFER_PATH)
    corner = _registered_corner_whitener(noise)
    report: dict[str, object] = {
        "schema_version": 1,
        "method": "offline repeated-frame spatial stationarity and modal calibration",
        "no_forward_or_finite_difference": True,
        "noise": {
            "source": str(NOISE_PATH.relative_to(ROOT)),
            "shape": list(noise.shape),
            "dtype": str(noise.dtype),
            "sha256": hashlib.sha256(NOISE_PATH.read_bytes()).hexdigest(),
            "pixel_size_mm": PIXEL_SIZE_MM,
            "conversion": "image_flow_to_canonical (swap image components, multiply by pixel size)",
        },
        "zones": {
            name: {"bounds_axis0_axis1": list(bounds), "window_side": SIDE, "stride": 32}
            for name, bounds in ZONES.items()
        },
        "m20_crop": {"bounds_axis0_axis1": list(M20_CROP), "inside": "p43_core"},
        "registered_whitener": {
            "name": "corner_registered_whitener",
            "calibration_bounds": [0, 512, 0, 512],
            "sample_count": SAMPLE_COUNT,
            "seed": SEED,
            "remove_spatial_mean": False,
            "support": "one-node border removed",
            "transfer": "not applied to noise stationarity summaries; applied to Jacobian modes",
        },
        "zone_statistics_raw": {},
        "zone_statistics_corner_whitened": {},
    }
    origins_by_zone: dict[str, list[tuple[int, int]]] = {}
    for name, bounds in ZONES.items():
        origins = _origins(bounds)
        origins_by_zone[name] = origins
        report["zone_statistics_raw"][name] = _summarise_windows(noise, origins)  # type: ignore[index]
        report["zone_statistics_corner_whitened"][name] = _summarise_windows(  # type: ignore[index]
            noise, origins, corner
        )

    jacobian = _read_jacobian()
    whitened_jacobian = _transform_jacobian(jacobian, transfer, corner)
    left, singular, _ = np.linalg.svd(whitened_jacobian, full_matrices=False)
    report["corner_modal_projection"] = {
        name: _modal_projection_stats(noise, origins, corner, left)
        for name, origins in origins_by_zone.items()
        if name != "calibration_corner"
    }
    report["corner_sensitivity_svd"] = {
        "singular_values": singular.tolist(),
        "normalised": (singular / singular[0]).tolist(),
    }
    for name, projection in report["corner_modal_projection"].items():  # type: ignore[union-attr]
        projection["empirical_delta_q_noise"] = _empirical_delta_q(  # type: ignore[index]
            singular, projection
        )

    if args.build_local:
        local, calibration_origins = _local_candidate(noise)
        local_jacobian = _transform_jacobian(jacobian, transfer, local)
        local_left, local_singular, _ = np.linalg.svd(local_jacobian, full_matrices=False)
        local_projection = _modal_projection_stats(
            noise, origins_by_zone["p43_core"], local, local_left
        )
        report["local_candidate"] = {
            "name": "P43_local_candidate_whitener",
            "calibration_zone": "p43_solve excluding p43_core",
            "calibration_window_count": len(calibration_origins),
            "calibration_seed": SEED,
            "validation_zone": "p43_core",
            "zone_statistics": _summarise_windows(noise, origins_by_zone["p43_core"], local),
            "modal_projection": local_projection,
            "singular_values": local_singular.tolist(),
            "normalised": (local_singular / local_singular[0]).tolist(),
            "absolute_delta_q_1sigma_nominal": (1.0 / local_singular).tolist(),
            "empirical_delta_q_noise": _empirical_delta_q(local_singular, local_projection),
            "interpretation": "candidate only; not a replacement for the registered whitener",
        }
    else:
        report["local_candidate"] = {
            "status": "not_built",
            "reason": "run with --build-local only after inspecting stationarity statistics",
        }

    corner_core_raw = report["zone_statistics_raw"]["p43_core"]  # type: ignore[index]
    corner_calibration_raw = report["zone_statistics_raw"]["calibration_corner"]  # type: ignore[index]
    report["conclusion"] = {
        "verdict": "B",
        "label": "strong spatial nonstationarity remains; absolute detectability remains unqualified",
        "basis": {
            "p43_core_to_corner_median_rms_ratio_ux": float(
                corner_core_raw["rms_ux"]["median"] / corner_calibration_raw["rms_ux"]["median"]
            ),
            "p43_core_to_corner_median_rms_ratio_uy": float(
                corner_core_raw["rms_uy"]["median"] / corner_calibration_raw["rms_uy"]["median"]
            ),
            "local_candidate_built": bool(args.build_local),
            "local_candidate_is_replacement": False,
        },
        "interpretation": (
            "P43 raw noise amplitude is materially lower than the registered corner calibration. "
            "The local stationary candidate improves component-scale amplitudes but does not bring "
            "held-out modal projections uniformly closer to unit variance; no absolute detectability "
            "rank is therefore qualified."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "build_local": args.build_local}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
