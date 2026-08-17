#!/usr/bin/env python3
"""Gate 6: kernel-component stability across runs.

Loads the trained runs' field reports (`--field-report` npz) and their result
JSONs, decomposes each recovered increment against the milestone-3 null
directions recomputed on the campaign grid (stored by the gate-4 twin), and
measures the two registered quantities:

* `std` across runs of `||P_ker Delta eps^p|| / ||Delta eps^p||  <= 5 %`
* `std` across runs of the holdout median `E`                        `<= 2 * margin(E)`

The decomposition uses the milestone-3 convention: least squares of the
assembled field in the 192-coefficient tensor family (degree zero, patches 8),
Euclidean coefficient coordinates, null directions = right singular vectors
with `sigma < sigma_1 * 1e-6`. A good `E` with an unstable kernel component
means the network has hidden the identifiability problem, not solved it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fem_inhouse.identification.tensor_local_inverse import TensorLocalBasis

ROOT = Path(__file__).resolve().parents[1]
PIXELS = 100
PATCHES = 8
SUBCELLS = 2
MAX_KERNEL_STD = 0.05
MARGIN_FACTOR = 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--twin",
        type=Path,
        default=ROOT / "validation/_generated/shared_tensor_generator/twin_gate4.npz",
    )
    parser.add_argument(
        "--margin",
        type=Path,
        default=ROOT / "validation/_generated/shared_tensor_generator/margin_frozen.json",
    )
    parser.add_argument("--fields", nargs="+", type=Path, required=True,
                        help="run field reports, paired with --reports in order")
    parser.add_argument("--reports", nargs="+", type=Path, required=True,
                        help="run result JSONs, paired with --fields in order")
    parser.add_argument("--rank", type=int, default=4,
                        help="which learned_r{rank} entry the reports hold")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation/_generated/shared_tensor_generator/gate6_stability.json",
    )
    arguments = parser.parse_args()
    if len(arguments.fields) != len(arguments.reports):
        raise SystemExit("--fields and --reports must be paired one to one")

    twin = np.load(arguments.twin, allow_pickle=False)
    null_vectors = twin["null_vectors"]
    margin = json.loads(arguments.margin.read_text(encoding="utf-8"))
    frozen_margin = float(margin["margin_frozen"])

    basis = TensorLocalBasis.build(PIXELS, PIXELS, PATCHES)
    count = basis.coefficient_count
    columns = np.empty((PIXELS * PIXELS * 3, count))
    unit = np.zeros(count)
    for index in range(count):
        unit[:] = 0.0
        unit[index] = 1.0
        columns[:, index] = basis.assemble(unit.reshape(basis.coefficient_shape)).ravel()
    gram = columns.T @ columns
    ridge = 1e-12 * max(float(np.trace(gram)) / count, 1e-300)

    def coefficient_fit(field_points: np.ndarray) -> np.ndarray:
        pixel = field_points.reshape(PIXELS, PIXELS, SUBCELLS, 3).mean(axis=2)
        right = basis.assemble_transpose(pixel).ravel()
        return np.linalg.solve(gram + ridge * np.eye(count), right)

    def kernel_share(field_points: np.ndarray) -> float:
        coefficients = coefficient_fit(field_points)
        projected = null_vectors @ coefficients
        return float(np.linalg.norm(projected) / max(np.linalg.norm(coefficients), 1e-300))

    rows = []
    for fields_path, report_path in zip(arguments.fields, arguments.reports, strict=True):
        fields = np.load(fields_path, allow_pickle=False)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        increments = fields["increments"]
        holdout = {int(s) for s in report["holdout_states"]}
        states = [int(s) for s in report["states"]]
        learned = report["results"][f"learned_r{arguments.rank}"]
        scores = {int(s): float(v) for s, v in learned["per_state"].items()}
        holdout_median = float(np.median([scores[s] for s in sorted(holdout)]))
        per_state_shares = {
            states[i]: kernel_share(increments[i]) for i in range(len(states))
        }
        rows.append(
            {
                "fields": str(fields_path),
                "report": str(report_path),
                "final_state_share": per_state_shares[states[-1]],
                "holdout_median_E": holdout_median,
                "per_state_share": per_state_shares,
            }
        )

    final_shares = np.asarray([r["final_state_share"] for r in rows])
    holdout_medians = np.asarray([r["holdout_median_E"] for r in rows])
    kernel_std = float(final_shares.std())
    e_std = float(holdout_medians.std())
    result = {
        "schema_version": 1,
        "runs": rows,
        "final_state_share_mean": float(final_shares.mean()),
        "final_state_share_std": kernel_std,
        "kernel_std_threshold": MAX_KERNEL_STD,
        "holdout_median_E_mean": float(holdout_medians.mean()),
        "holdout_median_E_std": e_std,
        "e_std_threshold": MARGIN_FACTOR * frozen_margin,
        "frozen_margin": frozen_margin,
        "passed_kernel_stability": kernel_std <= MAX_KERNEL_STD,
        "passed_E_stability": e_std <= MARGIN_FACTOR * frozen_margin,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
