"""Record deterministic SciPy/FFTW transform and B0 contract measurements."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyfftw
import scipy

from fem_inhouse.spectral2d.green import B0Green2D, ReferenceOperatorSymbols
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.transform_factory import create_full_dirichlet_dsti_plan
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig


def _relative(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(float(np.linalg.norm(reference)), 1.0e-30)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records = []
    for mesh in (4, 8, 12, 24, 48):
        grid = StructuredGrid2D(mesh, mesh, 1.0, 1.0)
        field = np.random.default_rng(mesh).normal(size=(*grid.interior_shape, 2))
        scipy_plan = create_full_dirichlet_dsti_plan(
            grid, SpectralTransformConfig(backend="scipy")
        )
        fftw_plan = create_full_dirichlet_dsti_plan(
            grid,
            SpectralTransformConfig(
                backend="fftw", fftw_planner_effort="estimate", fftw_use_wisdom=False
            ),
        )
        scipy_forward = scipy_plan.forward_displacement(field)
        fftw_forward = fftw_plan.forward_displacement(field)
        scipy_inverse = scipy_plan.inverse_displacement(scipy_forward)
        fftw_inverse = fftw_plan.inverse_displacement(fftw_forward)
        other = np.random.default_rng(mesh + 1000).normal(size=field.shape)
        scipy_other = scipy_plan.forward_displacement(other)
        fftw_other = fftw_plan.forward_displacement(other)
        scipy_green = B0Green2D(
            ReferenceOperatorSymbols(
                np.ones(grid.interior_shape),
                np.ones(grid.interior_shape),
                np.ones(grid.interior_shape),
            ),
            lambda_0=2.0,
            mu_0=3.0,
        )
        scipy_preconditioned = scipy_plan.inverse_displacement(scipy_green.apply(scipy_forward))
        fftw_preconditioned = fftw_plan.inverse_displacement(scipy_green.apply(fftw_forward))
        records.append(
            {
                "mesh": mesh,
                "roundtrip_scipy": _relative(scipy_inverse, field),
                "roundtrip_fftw": _relative(fftw_inverse, field),
                "forward_backend": _relative(fftw_forward, scipy_forward),
                "inverse_backend": _relative(
                    fftw_inverse, scipy_inverse
                ),
                "inner_product_backend": abs(
                    np.vdot(fftw_forward, fftw_other) - np.vdot(scipy_forward, scipy_other)
                )
                / max(float(np.linalg.norm(field) * np.linalg.norm(other)), 1.0e-30),
                "preconditioner_backend": _relative(fftw_preconditioned, scipy_preconditioned),
                "scipy_diagnostics": asdict(scipy_plan.diagnostics),
                "fftw_diagnostics": asdict(fftw_plan.diagnostics),
            }
        )
    report = {
        "status": "completed_fftw_contract_measurement",
        "grids": [4, 8, 12, 24, 48],
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyfftw": pyfftw.__version__,
        },
        "records": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
