#!/usr/bin/env python
"""Run the 13 missing points of the registered P43 (ell, alpha) matrix.

Protocol: `validation/p0043_small_parameter_matrix_preregistration.md`.

Full spatial resolution throughout -- correction C1 of the preregistration: the
symmetric observation operator is only defined when one element is one pixel,
so no reduced-mesh tier exists. Every setting other than `ell` and the coupling
modulus is fixed to the archived campaigns' values so the three reused points
stay comparable.

Sequential on purpose: `mfront_threads = 8` matches the archived manifests, and
changing it would leave a different configuration on disk for no gain.

Resumable. A point whose `status.json` reports `complete` is skipped, so the
script can be interrupted and restarted.

Progress is written to `progress.tsv` in the log directory after every
increment, and a one-line summary to `progress.txt`, so a long run can be
followed without reading the per-run logs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: alpha is a multiplier of the reference coupling modulus, which is how the
#: archived runs are parameterised.
HCHI_REFERENCE_MPA = 5168.147582748343

ALPHAS = (0.5, 1.0, 2.0, 4.0)
ELLS_MICROMETRES = (20.0, 40.0, 58.88, 90.0)

#: Already archived, and never recomputed.
ARCHIVED: dict[tuple[float, float], str] = {
    (1.0, 58.88): "results/constitutive-nonlocal-p0043-pad150-a100",
    (2.0, 58.88): "results/constitutive-nonlocal-p0043-pad150-a200",
    (4.0, 58.88): "results/constitutive-nonlocal-p0043-pad150-a400",
}

INCREMENTS = 20
PARTITION_ID = 43

#: Registered solver-reproducibility replicate: one matrix point recomputed at
#: 40 increments, whose spread becomes a floor on every heat map.
REPLICATE_POINT = (2.0, 40.0)
REPLICATE_INCREMENTS = 40


def _tag_alpha(alpha: float) -> str:
    return f"a{alpha:g}".replace(".", "p")


def _tag_ell(ell: float) -> str:
    return f"ell{ell:g}".replace(".", "p")


@dataclass(frozen=True, slots=True)
class Point:
    alpha: float
    ell_um: float
    increments: int
    output: Path

    @property
    def label(self) -> str:
        suffix = "" if self.increments == INCREMENTS else f"-inc{self.increments}"
        return f"{_tag_alpha(self.alpha)}-{_tag_ell(self.ell_um)}{suffix}"

    @property
    def coupling_modulus_mpa(self) -> float:
        return self.alpha * HCHI_REFERENCE_MPA


def planned_points() -> list[Point]:
    points: list[Point] = []
    for ell in ELLS_MICROMETRES:
        for alpha in ALPHAS:
            if (alpha, ell) in ARCHIVED:
                continue
            name = f"mm-id-p0043-{_tag_alpha(alpha)}-{_tag_ell(ell)}"
            points.append(Point(alpha, ell, INCREMENTS, ROOT / "results" / name))
    alpha, ell = REPLICATE_POINT
    points.append(
        Point(
            alpha,
            ell,
            REPLICATE_INCREMENTS,
            ROOT
            / "results"
            / f"mm-id-p0043-{_tag_alpha(alpha)}-{_tag_ell(ell)}-inc{REPLICATE_INCREMENTS}",
        )
    )
    return points


def is_complete(point: Point) -> bool:
    status = point.output / "partitions" / f"{PARTITION_ID:04d}" / "status.json"
    if not status.is_file():
        return False
    try:
        return bool(json.loads(status.read_text(encoding="utf-8")).get("complete"))
    except (OSError, json.JSONDecodeError):
        return False


def command(point: Point) -> list[str]:
    return [
        str(ROOT / ".venv/bin/fem-inhouse"),
        "--verbose",
        "partition",
        "--input",
        str(ROOT / "data/processed/case_study"),
        "--output",
        str(point.output),
        "--parts-x",
        "10",
        "--parts-y",
        "10",
        "--padding",
        "150",
        "--increments",
        str(point.increments),
        "--constitutive-backend",
        "mfront-native-plane-stress",
        "--nonlocal-plasticity",
        "--nonlocal-length-um",
        repr(point.ell_um),
        "--nonlocal-coupling-modulus-mpa",
        repr(point.coupling_modulus_mpa),
        "--nonlocal-relaxation",
        "0.5",
        "--nonlocal-tolerance",
        "1e-6",
        "--nonlocal-max-iterations",
        "15",
        "--partition-id",
        str(PARTITION_ID),
        "--mfront-threads",
        "8",
    ]


def solver_environment() -> dict[str, str]:
    """MFront needs its library, its bindings and its shared objects."""

    environment = dict(os.environ)
    # Insurance against the child block-buffering its own stream, which would
    # hide progress just as effectively as a buffered log file.
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MFRONT_BEHAVIOUR_LIBRARY"] = str(ROOT / "build/mfront/src/libBehaviour.so")
    local = Path.home() / ".local"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(local / "lib/python3.12/site-packages"), environment.get("PYTHONPATH")])
    )
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, [str(local / "lib"), environment.get("LD_LIBRARY_PATH")])
    )
    return environment


INCREMENT_PATTERN = re.compile(r"\bincrement=(\d+)")


def run_point(point: Point, *, index: int, total: int, logs: Path) -> tuple[bool, float]:
    """Run one point, streaming its log and updating the progress files."""

    log_path = logs / f"{point.label}.log"
    started = time.perf_counter()
    highest = 0
    with (
        # Line-buffered: a block-buffered log stays empty on disk for minutes,
        # which defeats the point of running with --verbose at all.
        log_path.open("w", encoding="utf-8", buffering=1) as log,
        subprocess.Popen(
            command(point),
            cwd=ROOT,
            env=solver_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process,
    ):
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            match = INCREMENT_PATTERN.search(line)
            if match is None:
                continue
            increment = int(match.group(1))
            if increment <= highest:
                continue
            highest = increment
            elapsed = time.perf_counter() - started
            per = elapsed / max(highest, 1)
            remaining = per * (point.increments - highest)
            _write_progress(
                logs,
                point=point,
                index=index,
                total=total,
                increment=highest,
                elapsed=elapsed,
                remaining=remaining,
            )
        code = process.wait()
    return code == 0, time.perf_counter() - started


def _write_progress(
    logs: Path,
    *,
    point: Point,
    index: int,
    total: int,
    increment: int,
    elapsed: float,
    remaining: float,
) -> None:
    row = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{index}/{total}\t{point.label}\t"
        f"{increment}/{point.increments}\t{elapsed / 60:.1f} min\t"
        f"ETA {remaining / 60:.1f} min\n"
    )
    with (logs / "progress.tsv").open("a", encoding="utf-8") as stream:
        stream.write(row)
    (logs / "progress.txt").write_text(
        f"point {index}/{total}  {point.label}  "
        f"increment {increment}/{point.increments}  "
        f"{elapsed / 60:.1f} min elapsed, about {remaining / 60:.1f} min left\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", default=str(ROOT / "results/mm-matrix-logs"))
    parser.add_argument("--dry-run", action="store_true", help="list the points and stop")
    arguments = parser.parse_args()

    logs = Path(arguments.logs)
    logs.mkdir(parents=True, exist_ok=True)
    points = planned_points()

    pending = [p for p in points if not is_complete(p)]
    print(f"{len(points)} points planned, {len(pending)} still to run", flush=True)
    for point in points:
        state = "done" if is_complete(point) else "pending"
        print(
            f"  {point.label:22s} alpha={point.alpha:<4g} ell={point.ell_um:<6g} "
            f"Hchi={point.coupling_modulus_mpa:12.4f} inc={point.increments:<3d} {state}",
            flush=True,
        )
    if arguments.dry_run:
        return 0

    campaign_started = time.perf_counter()
    failures: list[str] = []
    for index, point in enumerate(pending, start=1):
        print(
            f"\n=== [{index}/{len(pending)}] {point.label} -> {point.output.name}",
            flush=True,
        )
        succeeded, seconds = run_point(point, index=index, total=len(pending), logs=logs)
        status = "ok" if succeeded else "FAILED"
        print(f"    {status} in {seconds / 60:.1f} min", flush=True)
        if not succeeded:
            failures.append(point.label)
        done = time.perf_counter() - campaign_started
        left = (done / index) * (len(pending) - index)
        print(
            f"    campaign {done / 3600:.2f} h elapsed, about {left / 3600:.2f} h left",
            flush=True,
        )

    if failures:
        print(f"\nFAILED points: {', '.join(failures)}", flush=True)
        return 1
    print("\nall points complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
