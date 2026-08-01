#!/usr/bin/env python
"""Observe every completed P43 matrix point through the symmetric operator.

Protocol: `validation/p0043_small_parameter_matrix_preregistration.md`,
section 4. No primary indicator compares the DIC to a raw FEM field, so every
matrix point has to be warped onto the reference image and re-observed through
DISFlow before it can be scored.

No mechanics is rerun: `replay_dic_observation` checks the source `U.npy`
against the immutable campaign status before observing it.

Resumable, and safe to run while the matrix is still computing -- a point whose
mechanics is not complete is skipped and picked up on the next pass.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from fem_inhouse.workflows.dic_observation_replay import replay_dic_observation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0043_parameter_matrix import (  # noqa: E402
    ARCHIVED,
    PARTITION_ID,
    planned_points,
)

PREPARED_CASE = ROOT / "data/processed/case_study"
REFERENCE_IMAGE = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/DIC_images/000294.tif")
OUTPUT_ROOT = ROOT / "validation/reference_data/p0043_matrix_observations_v1"

#: Primary by provenance, then the registered sensitivity profile.
PROFILES = ("legacy_script_2021", "declared_medium_v4")


def targets() -> list[tuple[str, Path]]:
    """Every matrix point that needs observing, archived ones excluded."""

    rows: list[tuple[str, Path]] = []
    for point in planned_points():
        if point.increments != 20:
            # The reproducibility replicate is scored like any other point.
            rows.append((point.label, point.output))
            continue
        rows.append((point.label, point.output))
    for (alpha, ell), path in ARCHIVED.items():
        rows.append((f"archived-a{alpha:g}-ell{ell:g}".replace(".", "p"), ROOT / path))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    pending: list[tuple[str, Path, str, Path]] = []
    for label, campaign in targets():
        status = campaign / "partitions" / f"{PARTITION_ID:04d}" / "status.json"
        if not status.is_file():
            print(f"  {label:26s} mechanics not started, skipped", flush=True)
            continue
        for profile in PROFILES:
            output = OUTPUT_ROOT / f"{label}_{profile}"
            if (output / "report.json").is_file():
                continue
            pending.append((label, campaign, profile, output))

    print(f"{len(pending)} observations pending", flush=True)
    for label, _campaign, profile, _output in pending:
        print(f"  {label:26s} {profile}", flush=True)
    if arguments.dry_run:
        return 0

    failures: list[str] = []
    for index, (label, campaign, profile, output) in enumerate(pending, start=1):
        started = time.perf_counter()
        print(f"=== [{index}/{len(pending)}] {label} {profile}", flush=True)
        try:
            replay_dic_observation(
                campaign=campaign,
                prepared_case=PREPARED_CASE,
                reference_image=REFERENCE_IMAGE,
                partition_id=PARTITION_ID,
                profile_name=profile,
                output_directory=output,
                overwrite=True,
            )
        except (OSError, ValueError, RuntimeError) as error:
            print(f"    FAILED {error}", flush=True)
            failures.append(f"{label}/{profile}")
            continue
        print(f"    ok in {time.perf_counter() - started:.0f} s", flush=True)

    if failures:
        print(f"FAILED: {', '.join(failures)}", flush=True)
        return 1
    print("all observations complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
