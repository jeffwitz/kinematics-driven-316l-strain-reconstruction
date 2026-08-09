"""Timing of the plane-stress tangent variants, on one case, interleaved.

Four ways of supplying the Newton matrix, everything else equal:

- `reference`  the condensed Python closure, the production backend;
- `gps`        the UMAT closure with the tangent the DSL returns;
- `gps_shadow` the UMAT closure with the reference Schur, evaluated at the
               GPS's own converged state -- one extra 3D integration per call;
- `gps_composite` the UMAT closure with the sub-stepped points' tangent
               rebuilt by finite differences on the composite trajectory.

The machine is noisy: on this hardware the reference alone has been measured
between `1.56` and `2.12 s` of material time on the same case. Two precautions
follow, and both matter more than the numbers themselves. Repetitions are
**interleaved** -- one pass of every variant, then the next pass -- so a
thermal drift hits all of them alike instead of the one that happened to run
during it. And the reported figure is the **median**, with the full spread
printed beside it, so a reader can see when a difference is smaller than the
noise it sits in.

Newton iteration counts are deterministic and are reported as they are.

Usage:
    MFRONT_BEHAVIOUR_LIBRARY=$PWD/build/mfront/src/libBehaviour.so \\
    .venv/bin/python scripts/benchmark_gps_tangent_variants.py --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

#: 20x20 window at the centre of the registered P43 crop.
CROP_20X20 = (1610, 1630, 1075, 1095)
EBSD_ORIENTATION_H5 = "/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5"
PAIRED_PARAMETER_SET = "316l_guilhem2013_nasri2018_meric_srix_rate_1e-3"

GPS = "mfront-native-generalised-plane-stress"
REFERENCE = "mfront-3d-condensed-plane-stress"

#: name -> (backend, extra flags)
VARIANTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "reference": (REFERENCE, ()),
    "gps": (GPS, ()),
    "gps_shadow": (GPS, ("--gps-shadow-tangent",)),
    "gps_composite": (GPS, ("--gps-composite-fd-tangent",)),
}


def _run(
    name: str,
    arguments: argparse.Namespace,
    output_directory: Path,
    repeat: int,
) -> dict[str, Any]:
    backend, extra = VARIANTS[name]
    report = output_directory / f"{name}_r{repeat}.json"
    log = output_directory / f"{name}_r{repeat}.log"
    command = [
        sys.executable,
        "scripts/qualify_crystal_tet2_p43.py",
        "--crop-nodes",
        *(str(value) for value in arguments.crop_nodes),
        "--increments",
        str(arguments.increments),
        "--paired-parameter-set",
        arguments.paired_parameter_set,
        "--ebsd-orientation-h5",
        str(arguments.ebsd_orientation_h5),
        "--material-backend",
        backend,
        "--mfront-threads",
        str(arguments.mfront_threads),
        "--maximum-newton-iterations",
        str(arguments.maximum_newton_iterations),
        "--local-transverse-predictor",
        "tangent",
        "--krylov-method",
        "lgmres",
        "--linear-mode",
        "eisenstat_walker",
        "--krylov-recycling",
        "--no-final-verification",
        "--output",
        str(report),
        *extra,
    ]
    # The material bridge owns its threads; leaving BLAS free to spawn its own
    # makes the timings depend on what else is running.
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    wall = time.perf_counter() - started
    if completed.returncode != 0 or not report.is_file():
        return {"variant": name, "repeat": repeat, "converged": False, "log": str(log)}
    payload = json.loads(report.read_text())

    def find(node: Any, key: str) -> Any:
        if isinstance(node, dict):
            if key in node:
                return node[key]
            for value in node.values():
                found = find(value, key)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = find(value, key)
                if found is not None:
                    return found
        return None

    return {
        "variant": name,
        "repeat": repeat,
        "converged": True,
        "wall_seconds": wall,
        "elapsed_seconds": find(payload, "elapsed_seconds"),
        "material_seconds": find(payload, "material_seconds"),
        "material_integration_seconds": find(payload, "material_integration_seconds"),
        "newton_iterations": find(payload, "newton_iterations"),
        "material_point_integrations": find(payload, "material_point_integrations"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-nodes", nargs=4, type=int, default=CROP_20X20)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mfront-threads", type=int, default=4)
    parser.add_argument("--maximum-newton-iterations", type=int, default=40)
    parser.add_argument("--paired-parameter-set", default=PAIRED_PARAMETER_SET)
    parser.add_argument("--ebsd-orientation-h5", type=Path, default=Path(EBSD_ORIENTATION_H5))
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/_generated/performance/gps_tangent_variants_m20.json"),
    )
    arguments = parser.parse_args()
    if os.environ.get("MFRONT_BEHAVIOUR_LIBRARY") is None:
        raise SystemExit("MFRONT_BEHAVIOUR_LIBRARY must be set")
    unknown = [name for name in arguments.variants if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {', '.join(unknown)}")

    directory = arguments.output.with_suffix("")
    directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for repeat in range(1, arguments.repeats + 1):
        for name in arguments.variants:
            record = _run(name, arguments, directory, repeat)
            records.append(record)
            if record["converged"]:
                print(
                    f"  pass {repeat}  {name:<14} "
                    f"{record['newton_iterations']:>3} Newton  "
                    f"material {record['material_seconds']:.2f}s  "
                    f"total {record['elapsed_seconds']:.2f}s",
                    flush=True,
                )
            else:
                print(f"  pass {repeat}  {name:<14} DID NOT CONVERGE", flush=True)

    summary: dict[str, Any] = {}
    for name in arguments.variants:
        runs = [r for r in records if r["variant"] == name and r["converged"]]
        if not runs:
            summary[name] = {"converged": False}
            continue
        material = [r["material_seconds"] for r in runs]
        summary[name] = {
            "converged": True,
            "runs": len(runs),
            "newton_iterations": sorted({r["newton_iterations"] for r in runs}),
            "material_median": statistics.median(material),
            "material_min": min(material),
            "material_max": max(material),
            "elapsed_median": statistics.median(r["elapsed_seconds"] for r in runs),
            "material_point_integrations": runs[0]["material_point_integrations"],
        }
    base = summary.get("reference", {})
    for item in summary.values():
        if item.get("converged") and base.get("converged"):
            item["material_speedup_vs_reference"] = (
                base["material_median"] / item["material_median"]
            )

    arguments.output.write_text(
        json.dumps({"records": records, "summary": summary}, indent=2, sort_keys=True) + "\n"
    )
    print()
    print(f"{'variante':<15}{'Newton':>8}{'materiau median':>18}{'etendue':>20}{'gain':>8}")
    for name in arguments.variants:
        item = summary[name]
        if not item.get("converged"):
            print(f"{name:<15}{'--':>8}{'ne converge pas':>18}")
            continue
        spread = f"{item['material_min']:.2f} - {item['material_max']:.2f}s"
        speedup = item.get("material_speedup_vs_reference")
        gain = f"{speedup:.2f}x" if speedup else "--"
        iterations = ",".join(str(value) for value in item["newton_iterations"])
        print(
            f"{name:<15}{iterations:>8}"
            f"{item['material_median']:>17.2f}s{spread:>20}{gain:>8}"
        )
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
