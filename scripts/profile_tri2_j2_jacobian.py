"""Profile the allocated and persistent TRI2 Python-J2 tangent actions."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from fem_inhouse.core.plane_stress_material import PythonJ2PlaneStressBatch
from fem_inhouse.spectral2d import StructuredGrid2D, TwoStateJacobianWorkspace
from fem_inhouse.spectral2d.kinematics import EBITwoTriangleKinematics2D
from fem_inhouse.spectral2d.newton_two_state import (
    TraditionalTwoStateTriangleBatch,
    pack_interior,
)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=int, default=100)
    parser.add_argument("--crop", nargs=4, type=int)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--kernel", choices=("einsum", "explicit"), default="einsum")
    parser.add_argument("--workspace", choices=("allocated", "persistent"), default="persistent")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.mesh < 2 or arguments.samples < 1:
        raise SystemExit("--mesh must be at least 2 and --samples must be positive")

    grid = StructuredGrid2D(arguments.mesh, arguments.mesh, 1.0, 1.0)
    point_count = 2 * arguments.mesh * arguments.mesh
    material = PythonJ2PlaneStressBatch(
        np.full(point_count, 250.0),
        np.full(point_count, 500.0),
        0.245,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    kinematics = EBITwoTriangleKinematics2D(grid)
    elements = TraditionalTwoStateTriangleBatch(material, grid.pixel_shape)
    rng = np.random.default_rng(20260805 + arguments.mesh)
    displacement = rng.normal(scale=1.0e-3, size=(*grid.node_shape, 2))
    trial = elements.evaluate_samples(kinematics.strain_samples(displacement), time_increment=0.1)
    vectors = rng.normal(
        size=(arguments.samples, 2 * (arguments.mesh - 1) * (arguments.mesh - 1))
    )
    fields = np.zeros((arguments.samples, *grid.node_shape, 2))
    fields[:, 1:-1, 1:-1, :] = vectors.reshape(
        arguments.samples, arguments.mesh - 1, arguments.mesh - 1, 2
    )
    persistent = TwoStateJacobianWorkspace.create(grid)
    for field in fields[: min(5, arguments.samples)]:
        if arguments.workspace == "persistent":
            persistent.nodal_increment[...] = field
            elements.tangent_action_into(
                kinematics=kinematics,
                trial=trial,
                workspace=persistent,
                kernel=arguments.kernel,
            )
        else:
            elements.tangent_action(field, kinematics=kinematics, trial=trial)

    samples: list[float] = []
    checksum = 0.0
    for field in fields:
        started = time.perf_counter()
        if arguments.workspace == "persistent":
            persistent.nodal_increment[...] = field
            result = elements.tangent_action_into(
                kinematics=kinematics,
                trial=trial,
                workspace=persistent,
                kernel=arguments.kernel,
            )
        else:
            result = pack_interior(
                elements.tangent_action(field, kinematics=kinematics, trial=trial)
            )
        samples.append(time.perf_counter() - started)
        checksum += float(np.sum(result))

    report = {
        "mesh": arguments.mesh,
        "crop": arguments.crop,
        "samples": arguments.samples,
        "kernel": arguments.kernel,
        "workspace": arguments.workspace,
        "result_checksum": checksum,
        "action_seconds": _summary(samples),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
