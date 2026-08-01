#!/usr/bin/env python
"""Local runs on the candidate ROIs, for the qualification filter.

Local only: no `--nonlocal-plasticity`, so the coupling is absent rather than
zero-valued. Everything else matches the archived P43 local run, so the new
ROIs are comparable with it and with each other.

This is the gate the P43 campaign lacked. A ROI is only worth a coupled matrix
if the local model already places the bands roughly right and makes them
measurably too narrow; that cannot be known without the local field.

Resumable and verbose, same progress files as the matrix driver.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Top of the DIC band-morphology ranking. P43, rank seven, is already computed.
DEFAULT_PARTITIONS = (17, 58, 84)
INCREMENTS = 20


def output_for(partition_id: int) -> Path:
    return ROOT / "results" / f"constitutive-local-p{partition_id:04d}-pad150"


def is_complete(partition_id: int) -> bool:
    status = (
        output_for(partition_id) / "partitions" / f"{partition_id:04d}" / "status.json"
    )
    if not status.is_file():
        return False
    try:
        return bool(json.loads(status.read_text(encoding="utf-8")).get("complete"))
    except (OSError, json.JSONDecodeError):
        return False


def command(partition_id: int) -> list[str]:
    return [
        str(ROOT / ".venv/bin/fem-inhouse"),
        "--verbose",
        "partition",
        "--input", str(ROOT / "data/processed/case_study"),
        "--output", str(output_for(partition_id)),
        "--parts-x", "10",
        "--parts-y", "10",
        "--padding", "150",
        "--increments", str(INCREMENTS),
        "--constitutive-backend", "mfront-native-plane-stress",
        "--partition-id", str(partition_id),
        "--mfront-threads", "8",
    ]


def solver_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MFRONT_BEHAVIOUR_LIBRARY"] = str(
        ROOT / "build/mfront/src/libBehaviour.so"
    )
    local = Path.home() / ".local"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [str(local / "lib/python3.12/site-packages"), environment.get("PYTHONPATH")],
        )
    )
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, [str(local / "lib"), environment.get("LD_LIBRARY_PATH")])
    )
    return environment


INCREMENT_PATTERN = re.compile(r"\bincrement=(\d+)")


def run(partition_id: int, *, index: int, total: int, logs: Path) -> tuple[bool, float]:
    log_path = logs / f"local_p{partition_id:04d}.log"
    started = time.perf_counter()
    highest = 0
    with (
        log_path.open("w", encoding="utf-8", buffering=1) as log,
        subprocess.Popen(
            command(partition_id),
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
            remaining = (elapsed / highest) * (INCREMENTS - highest)
            (logs / "progress.txt").write_text(
                f"local {index}/{total}  p{partition_id:04d}  "
                f"increment {highest}/{INCREMENTS}  "
                f"{elapsed / 60:.1f} min elapsed, about {remaining / 60:.1f} min left\n",
                encoding="utf-8",
            )
        code = process.wait()
    return code == 0, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partitions", default=",".join(str(p) for p in DEFAULT_PARTITIONS)
    )
    parser.add_argument("--logs", default=str(ROOT / "results/local-roi-logs"))
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    logs = Path(arguments.logs)
    logs.mkdir(parents=True, exist_ok=True)
    wanted = [int(v) for v in arguments.partitions.split(",")]
    pending = [p for p in wanted if not is_complete(p)]

    for partition_id in wanted:
        state = "done" if is_complete(partition_id) else "pending"
        print(f"  p{partition_id:04d}  -> {output_for(partition_id).name}  {state}", flush=True)
    print(f"{len(pending)} local runs to do", flush=True)
    if arguments.dry_run:
        return 0

    started = time.perf_counter()
    failures: list[int] = []
    for index, partition_id in enumerate(pending, start=1):
        print(f"\n=== [{index}/{len(pending)}] local p{partition_id:04d}", flush=True)
        ok, seconds = run(partition_id, index=index, total=len(pending), logs=logs)
        print(f"    {'ok' if ok else 'FAILED'} in {seconds / 60:.1f} min", flush=True)
        if not ok:
            failures.append(partition_id)
        done = time.perf_counter() - started
        print(
            f"    {done / 60:.0f} min elapsed, about "
            f"{(done / index) * (len(pending) - index) / 60:.0f} min left",
            flush=True,
        )

    if failures:
        print(f"\nFAILED: {failures}", flush=True)
        return 1
    print("\nall local runs complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
