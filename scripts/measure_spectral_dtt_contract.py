#!/usr/bin/env python3
"""Measure and archive the full-Dirichlet DST-I numerical contract."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from fem_inhouse.spectral2d import FullDirichletDSTIPlan2D, StructuredGrid2D

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "validation"
    / "reference_data"
    / "spectral_mechanics_evidence_v1"
    / "dtt_contract.json"
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(nx: int, ny: int) -> dict[str, object]:
    grid = StructuredGrid2D(nx, ny, 2.0, 3.0)
    plan = FullDirichletDSTIPlan2D(grid)
    seed = nx * 100 + ny
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(*grid.interior_shape, 2))
    v = rng.normal(size=(*grid.interior_shape, 2))
    u_hat = plan.forward_displacement(u)
    v_hat = plan.forward_displacement(v)
    round_trip_error = float(
        np.linalg.norm(plan.inverse_displacement(u_hat) - u) / np.linalg.norm(u)
    )
    uv = np.vdot(u, v)
    transformed_uv = np.vdot(u_hat, v_hat)
    inner_product_error = float(
        abs(transformed_uv - uv)
        / max(abs(uv), np.linalg.norm(u) * np.linalg.norm(v), 1.0e-30)
    )
    return {
        "nx": nx,
        "ny": ny,
        "seed": seed,
        "round_trip_relative_error": round_trip_error,
        "inner_product_relative_error": inner_product_error,
    }


def main() -> None:
    grids = [measure(nx, ny) for nx, ny in ((4, 4), (5, 4), (4, 5), (7, 6), (12, 12))]
    maximum_round_trip_error = max(item["round_trip_relative_error"] for item in grids)
    maximum_inner_product_error = max(item["inner_product_relative_error"] for item in grids)
    report = {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "source_sha256": {
            "grid.py": file_sha256(ROOT / "src/fem_inhouse/spectral2d/grid.py"),
            "transforms.py": file_sha256(ROOT / "src/fem_inhouse/spectral2d/transforms.py"),
            "measure_spectral_dtt_contract.py": file_sha256(Path(__file__)),
        },
        "grids": grids,
        "maximum_round_trip_error": maximum_round_trip_error,
        "maximum_inner_product_error": maximum_inner_product_error,
        "passed": bool(
            maximum_round_trip_error < 1.0e-13
            and maximum_inner_product_error < 1.0e-13
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
