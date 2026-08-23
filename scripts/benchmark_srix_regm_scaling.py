#!/usr/bin/env python3
"""Benchmark one SRIX-REGM evaluation on M20 and M100-like grids."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fem_inhouse.identification.srix_equilibrium_gap import SrixEquilibriumGapProblem
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from scripts.qualify_srix_regm_twin import (
    PIXEL_SIZE_MM,
    SUBSTEPS_PER_SEGMENT,
    _boundary_history,
    _expanded_path,
    _material_factory,
    _operator,
    _orientation_map,
    _theta_from_preset,
)

ROOT = Path(__file__).resolve().parents[1]


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=(20, 100))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation/reference_data/srix_regm_scaling_v1/report.json",
    )
    arguments = parser.parse_args()
    output = arguments.output if arguments.output.is_absolute() else ROOT / arguments.output
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    git_sha = _git("rev-parse HEAD")
    git_dirty = bool(_git("status --porcelain"))
    library = os.environ.get(
        "MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so")
    )
    if not Path(library).is_file():
        raise SystemExit(f"missing MFront library: {library}")

    theta = _theta_from_preset()
    path = _expanded_path()
    rows = []
    for pixels in arguments.sizes:
        grid = StructuredGrid2D(
            pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels
        )
        orientations = _orientation_map(pixels)
        setup_started = time.perf_counter()
        operator = _operator(grid, orientations)
        operator_setup_seconds = time.perf_counter() - setup_started
        problem = SrixEquilibriumGapProblem(
            operator=operator,
            displacement_history=_boundary_history(grid),
            state_indices=tuple(range(1, len(path))),
            scored_states=set(
                range(SUBSTEPS_PER_SEGMENT, len(path), SUBSTEPS_PER_SEGMENT)
            ),
            material_factory=_material_factory(
                pixels=pixels,
                orientations=orientations,
                library=library,
                threads=arguments.threads,
            ),
            time_increments=np.full(len(path) - 1, 1.0 / (len(path) - 1)),
            debug=False,
        )
        evaluation = problem.evaluate(theta)
        rows.append(
            {
                "mesh": [pixels, pixels],
                "material_points": 2 * pixels * pixels,
                "states_replayed": len(path) - 1,
                "states_scored": 8,
                "operator_setup_seconds": operator_setup_seconds,
                "evaluation_timing": asdict(evaluation.timing),
                "backend_timing": dict(evaluation.backend_timing),
                "material_evaluations": evaluation.material_evaluations,
                "residual_rms_mm": evaluation.residual_rms,
            }
        )

    report = {
        "schema_version": 1,
        "method": "SRIX-REGM scaling benchmark on prescribed affine history",
        "scientific_use": "performance only; the affine history is not an equilibrated twin",
        "git_sha": git_sha,
        "dirty": git_dirty,
        "machine": platform.node(),
        "threads": arguments.threads,
        "parameter_set": theta.as_runtime_overrides(),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
