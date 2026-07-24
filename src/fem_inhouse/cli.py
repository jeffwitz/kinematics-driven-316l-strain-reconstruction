"""Command-line interface limited to the supported case study."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from fem_inhouse.examples import (
    reduced_biaxial_case,
    save_reduced_example,
    validate_reduced_case,
)
from fem_inhouse.partitioning import PartitionLayout
from fem_inhouse.solver import linear_solver_backend, require_pypardiso


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
    return parser


def _print_json(data) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


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
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover
