"""Command-line interface limited to the supported case study."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fem_inhouse.config import (
    CaseStudyConfig,
    MaterialConfig,
    MeshConfig,
    SolverConfig,
)
from fem_inhouse.examples import (
    reduced_biaxial_case,
    save_reduced_example,
    validate_reduced_case,
)
from fem_inhouse.partitioning import PartitionLayout
from fem_inhouse.solver import linear_solver_backend, require_pypardiso
from fem_inhouse.workflows import PartitionWorkflow

PARTITION_FIELDS = ("U", "S", "E", "PEEQ")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fem-inhouse",
        description="Kinematics-driven 316L case-study reconstruction tools.",
    )
    parser.add_argument("--verbose", action="store_true", help="enable progress logging")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("backend", help="verify and print the sparse solver backend")

    validate = commands.add_parser("validate", help="run the reduced analytical case")
    validate.add_argument("--nx", type=int, default=10)
    validate.add_argument("--ny", type=int, default=10)

    example = commands.add_parser("example", help="run and save the reduced example")
    example.add_argument("--output", type=Path, required=True)
    example.add_argument("--nx", type=int, default=10)
    example.add_argument("--ny", type=int, default=10)

    layout = commands.add_parser("layout", help="write an article partition manifest")
    layout.add_argument("--count", type=int, choices=(25, 100), required=True)
    layout.add_argument("--padding", type=int, default=150)
    layout.add_argument("--output", type=Path, required=True)

    partition = commands.add_parser(
        "partition",
        help="inspect, solve, or stitch the article partition workflow",
    )
    partition.add_argument("--input", type=Path, required=True)
    partition.add_argument("--output", type=Path, required=True)
    partition.add_argument("--count", type=int, choices=(25, 100), required=True)
    partition.add_argument("--padding", type=int, default=150)
    partition.add_argument("--base-pixel-mm", type=float, default=0.001)
    partition.add_argument("--scale-factor", type=float, default=1.84)
    partition.add_argument("--young-modulus-mpa", type=float, default=205_000.0)
    partition.add_argument("--poisson-ratio", type=float, default=0.3)
    partition.add_argument("--hardening-exponent", type=float, default=0.245)
    partition.add_argument("--increments", type=int, default=20)
    partition.add_argument("--max-newton-iterations", type=int, default=15)
    partition.add_argument("--residual-tolerance", type=float, default=1e-6)
    action = partition.add_mutually_exclusive_group(required=True)
    action.add_argument("--list-pending", action="store_true")
    action.add_argument("--partition-id", type=int)
    action.add_argument("--solve-pending", action="store_true")
    action.add_argument("--stitch", choices=PARTITION_FIELDS)
    partition.add_argument(
        "--field-output",
        type=Path,
        help="optional output path used with --stitch",
    )
    return parser


def _print_json(data) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _load_partition_field(directory: Path, name: str) -> np.ndarray:
    path = directory / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"missing partition input: {path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {values.shape}")
    return values


def _partition_workflow(args: argparse.Namespace) -> PartitionWorkflow:
    displacement_x = _load_partition_field(args.input, "displacement_x_mm")
    displacement_y = _load_partition_field(args.input, "displacement_y_mm")
    yield_stress = _load_partition_field(args.input, "yield_stress_mpa")
    hardening = _load_partition_field(args.input, "hardening_coefficient_mpa")
    if yield_stress.shape != hardening.shape:
        raise ValueError("material maps must have the same shape")
    nx, ny = yield_stress.shape
    side = 5 if args.count == 25 else 10
    config = CaseStudyConfig(
        mesh=MeshConfig(
            nx=nx,
            ny=ny,
            base_pixel_size_mm=args.base_pixel_mm,
            scale_factor=args.scale_factor,
        ),
        material=MaterialConfig(
            young_modulus_mpa=args.young_modulus_mpa,
            poisson_ratio=args.poisson_ratio,
            hardening_exponent=args.hardening_exponent,
        ),
        solver=SolverConfig(
            increments=args.increments,
            max_newton_iterations=args.max_newton_iterations,
            residual_tolerance=args.residual_tolerance,
        ),
    )
    return PartitionWorkflow(
        config=config,
        layout=PartitionLayout((nx, ny), (side, side), padding=args.padding),
        displacement_x_mm=displacement_x,
        displacement_y_mm=displacement_y,
        yield_stress_mpa=yield_stress,
        hardening_coefficient_mpa=hardening,
        output_directory=args.output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one supported command and return a process exit status."""

    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.command == "backend":
        require_pypardiso()
        print(linear_solver_backend())
        return 0
    if args.command == "layout":
        side = 5 if args.count == 25 else 10
        layout = PartitionLayout((3_600, 3_100), (side, side), padding=args.padding)
        layout.write_manifest(args.output)
        print(args.output)
        return 0
    if args.command == "validate":
        _result, report = validate_reduced_case(reduced_biaxial_case(nx=args.nx, ny=args.ny))
        _print_json(asdict(report))
        return 0 if report.passed else 1
    if args.command == "example":
        report = save_reduced_example(args.output, nx=args.nx, ny=args.ny)
        _print_json(asdict(report))
        return 0 if report.passed else 1
    if args.command == "partition":
        workflow = _partition_workflow(args)
        if args.list_pending:
            _print_json({"pending": workflow.pending_partition_ids()})
            return 0
        if args.partition_id is not None:
            _print_json(workflow.solve_partition(args.partition_id))
            return 0
        if args.solve_pending:
            solved = workflow.solve_pending()
            _print_json(
                {
                    "remaining": workflow.pending_partition_ids(),
                    "solved": solved,
                }
            )
            return 0
        output = workflow.stitch(args.stitch, output_path=args.field_output)
        print(output.filename)
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover
