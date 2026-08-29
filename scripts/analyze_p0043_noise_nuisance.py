#!/usr/bin/env python3
"""Offline decomposition of structured components in the P43 repeat-frame noise.

This diagnostic uses only the hydrated repeated-frame field.  It estimates and
removes translation and affine displacement fields independently in each
region, then records energy and coarse spectral changes.  It does not call a
mechanical solver or alter the registered whitening artifacts.
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
NOISE_PATH = ROOT / (
    "validation/reference_data/dic_uncertainty_propagation_p0043_v1/centred_repeat_flow_pixels.npy"
)
PIXEL_SIZE_MM = 0.00184
SIDE = 21
ZONES = {
    "full_field": (0, 3600, 0, 3100),
    "calibration_corner": (0, 512, 0, 512),
    "p43_solve": (1290, 1950, 780, 1390),
    "p43_core": (1440, 1800, 930, 1240),
}


def _load_coordinates():
    path = ROOT / "src/fem_inhouse/measurement/coordinates.py"
    spec = importlib.util.spec_from_file_location("offline_coordinates_nuisance", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.image_flow_to_canonical


image_flow_to_canonical = _load_coordinates()


def _canonical_region(noise: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x0, x1, y0, y1 = bounds
    return image_flow_to_canonical(np.asarray(noise[x0:x1, y0:y1]), pixel_size_mm=PIXEL_SIZE_MM)


def _affine_fit(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = field.shape[:2]
    x = np.linspace(-1.0, 1.0, rows, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, columns, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    design = np.stack((np.ones_like(xx), xx, yy), axis=-1).reshape(-1, 3)
    values = field.reshape(-1, field.shape[-1]).astype(np.float64, copy=False)
    gram = design.T @ design
    coefficients = np.linalg.solve(gram, design.T @ values)
    affine = (design @ coefficients).reshape(field.shape)
    translation = np.broadcast_to(values.mean(axis=0), values.shape).reshape(field.shape)
    return translation, affine, coefficients


def _rms(field: np.ndarray) -> dict[str, float]:
    component = np.sqrt(np.mean(np.square(field), axis=(0, 1)))
    return {
        "ux": float(component[0]),
        "uy": float(component[1]),
        "vector": float(np.sqrt(np.mean(np.sum(np.square(field), axis=-1)))),
    }


def _spectrum_summary(field: np.ndarray) -> dict[str, object]:
    centered = field - field.mean(axis=(0, 1), keepdims=True)
    rows, columns = field.shape[:2]
    fx = np.fft.fftfreq(rows)[:, None]
    fy = np.fft.rfftfreq(columns)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    low_mask = radius <= 0.125
    by_component: dict[str, dict[str, float]] = {}
    for component, name in enumerate(("ux", "uy")):
        spectrum = np.fft.rfft2(centered[..., component], norm="ortho")
        power = np.square(np.abs(spectrum))
        total = float(power.sum())
        low = float(power[low_mask].sum())
        by_component[name] = {
            "low_frequency_fraction": low / total if total else 0.0,
            "high_frequency_fraction": 1.0 - low / total if total else 0.0,
        }
    return {"components": by_component}


def _region_report(noise: np.ndarray, bounds: tuple[int, int, int, int]) -> dict[str, object]:
    raw = _canonical_region(noise, bounds)
    translation, affine, coefficients = _affine_fit(raw)
    translation_removed = raw - translation
    affine_removed = raw - affine
    raw_energy = float(np.sum(np.square(raw)))
    translation_energy = float(np.sum(np.square(translation_removed)))
    affine_energy = float(np.sum(np.square(affine_removed)))
    return {
        "bounds_axis0_axis1": list(bounds),
        "shape": list(raw.shape),
        "affine_basis": "[1, x_normalized, y_normalized] per component",
        "affine_coefficients": coefficients.tolist(),
        "raw_rms": _rms(raw),
        "translation_removed_rms": _rms(translation_removed),
        "affine_removed_rms": _rms(affine_removed),
        "energy_fraction_removed_by_translation": 1.0 - translation_energy / raw_energy,
        "energy_fraction_removed_by_affine": 1.0 - affine_energy / raw_energy,
        "spectrum_raw": _spectrum_summary(raw),
        "spectrum_translation_removed": _spectrum_summary(translation_removed),
        "spectrum_affine_removed": _spectrum_summary(affine_removed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    noise = np.load(NOISE_PATH, mmap_mode="r", allow_pickle=False)
    if noise.shape != (3600, 3100, 2):
        raise ValueError(f"unexpected noise shape {noise.shape}")
    report = {
        "schema_version": 1,
        "method": "translation and affine nuisance decomposition of repeated-frame noise",
        "no_forward_or_finite_difference": True,
        "noise": {
            "source": str(NOISE_PATH.relative_to(ROOT)),
            "shape": list(noise.shape),
            "dtype": str(noise.dtype),
            "sha256": hashlib.sha256(NOISE_PATH.read_bytes()).hexdigest(),
            "pixel_size_mm": PIXEL_SIZE_MM,
            "conversion": "image_flow_to_canonical",
        },
        "regions": {name: _region_report(noise, bounds) for name, bounds in ZONES.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
