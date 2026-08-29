#!/usr/bin/env python3
"""Offline coarse spatial cross-validation of P43 repeat-frame noise."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_nuisance():
    path = ROOT / "scripts/analyze_p0043_noise_nuisance.py"
    spec = importlib.util.spec_from_file_location("offline_nuisance_cv", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    nuisance = _load_nuisance()
    noise = np.load(nuisance.NOISE_PATH, mmap_mode="r", allow_pickle=False)
    tile_x, tile_y = 600, 620
    tiles = {}
    for ix, x0 in enumerate(range(0, noise.shape[0], tile_x)):
        for iy, y0 in enumerate(range(0, noise.shape[1], tile_y)):
            x1, y1 = min(x0 + tile_x, noise.shape[0]), min(y0 + tile_y, noise.shape[1])
            bounds = (x0, x1, y0, y1)
            tiles[f"tile_{ix}_{iy}"] = nuisance._region_report(noise, bounds)
    report = {
        "schema_version": 1,
        "method": "non-overlapping coarse tile spatial cross-validation",
        "no_forward_or_finite_difference": True,
        "tile_shape_nominal_axis0_axis1": [tile_x, tile_y],
        "noise_source": str(nuisance.NOISE_PATH.relative_to(ROOT)),
        "tiles": tiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tile_count": len(tiles)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
