#!/usr/bin/env python
"""Is (alpha, ell) identifiable at all on P43? Two tests, no new mechanics.

Operations 3 and 4 of the 2026-08-01 review.

**Pairwise distances.** Every computed field is scored against every other with
the same four defects, in units of the combined noise floor: the DIC repetition
residual and the solver's own reproducibility, added in quadrature because they
are independent. Two parameterisations separated by less than one floor are not
experimentally distinguishable, whatever a selection rule reports.

**Pseudo-data recovery.** Each run is taken in turn as a synthetic truth, the
measured DIC residual is added to it, and the whole set is scored against it. If
the tool cannot recover a parameterisation it generated itself, it cannot
identify one from the experiment.

Reads observed fields only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from fem_inhouse.validation.selection_indicators import (
    DEFECT_NAMES,
    PRINCIPAL_SCALE_PIXELS,
    evaluate,
)
from fem_inhouse.workflows.compare_gradient_fluctuation_criteria import (
    gradient_on_core,
    observed_displacement,
)
from fem_inhouse.workflows.validate_selection_indicators import (
    REPETITION_REALISATIONS,
    REPETITION_SEED,
    correlated_repetition_residual,
)

ROOT = Path(__file__).resolve().parent.parent
OBS = ROOT / "validation/reference_data/p0043_matrix_observations_v1"
SYM = ROOT / "validation/reference_data/dic_symmetric_observation_p0043_v1"
OUT = ROOT / "validation/reference_data/p0043_small_parameter_matrix_v1"

ALPHAS = (0.5, 1.0, 2.0, 4.0)
ELLS = (20.0, 40.0, 58.88, 90.0)
ARCHIVED = {(1.0, 58.88), (2.0, 58.88), (4.0, 58.88)}


def load_fields(profile: str) -> dict[str, np.ndarray]:
    """Every observed displacement, with the local model as `a0`."""

    fields: dict[str, np.ndarray] = {}
    local = SYM / f"local_{profile}" / "observed_flow_pixels.npy"
    if local.is_file():
        fields["a0-local"] = observed_displacement(local)
    for alpha in ALPHAS:
        for ell in ELLS:
            label = f"a{alpha:g}-ell{ell:g}".replace(".", "p")
            name = f"archived-{label}" if (alpha, ell) in ARCHIVED else label
            path = OBS / f"{name}_{profile}" / "observed_flow_pixels.npy"
            if path.is_file():
                fields[label] = observed_displacement(path)
    return fields


def combined_floor() -> dict[str, float]:
    """Measurement and solver floors in quadrature, per indicator."""

    payload = json.loads((OUT / "floors.json").read_text(encoding="utf-8"))
    measurement = payload["measurement_floor"]
    solver = payload["solver_floor"]
    return {name: float(np.hypot(measurement[name], solver[name])) for name in DEFECT_NAMES}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="legacy_script_2021")
    arguments = parser.parse_args()
    profile = arguments.profile

    floor = combined_floor()
    print("combined floor (measurement and solver in quadrature)")
    for name, value in floor.items():
        print(f"  {name:16s} {value:.5g}")

    displacements = load_fields(profile)
    gradients = {label: gradient_on_core(u) for label, u in displacements.items()}
    labels = list(gradients)
    print(f"\n{len(labels)} fields: {labels}")

    # --- operation 3: pairwise distances in floor units -------------------
    worst = np.full((len(labels), len(labels)), np.nan)
    detail: dict[str, dict[str, dict[str, float]]] = {}
    for i, first in enumerate(labels):
        detail[first] = {}
        for j, second in enumerate(labels):
            if i == j:
                worst[i, j] = 0.0
                continue
            defects = evaluate(
                gradients[first],
                gradients[second],
                label=first,
                scale_pixels=PRINCIPAL_SCALE_PIXELS,
            ).as_dict()
            units = {n: abs(defects[n]) / floor[n] for n in DEFECT_NAMES}
            detail[first][second] = units
            worst[i, j] = max(units.values())

    print("\nworst defect between each pair, in floor units (>1 = distinguishable)")
    print(f"{'':14s}" + "".join(f"{lab.replace('-ell', '/'):>12s}" for lab in labels))
    for i, first in enumerate(labels):
        print(
            f"{first.replace('-ell', '/'):14s}"
            + "".join(
                f"{worst[i, j]:12.1f}" if np.isfinite(worst[i, j]) else f"{'':>12s}"
                for j in range(len(labels))
            )
        )

    off = worst[~np.eye(len(labels), dtype=bool)]
    print(f"\nsmallest non-diagonal distance: {np.nanmin(off):.1f} floors")
    close = [
        (labels[i], labels[j], worst[i, j])
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
        if worst[i, j] < 10.0
    ]
    print(f"pairs below 10 floors: {len(close)}")
    for a, b, v in sorted(close, key=lambda t: t[2])[:10]:
        print(f"   {a:16s} {b:16s} {v:6.1f}")

    # --- operation 4: pseudo-data recovery --------------------------------
    print("\n=== pseudo-data recovery: each run as its own synthetic truth")
    generator = np.random.default_rng(REPETITION_SEED)
    rows = []
    for truth in labels:
        recovered: list[str] = []
        for _ in range(REPETITION_REALISATIONS):
            perturbed = displacements[truth] + correlated_repetition_residual(
                displacements[truth].shape[:2], generator=generator
            )
            reference = gradient_on_core(perturbed)
            scores = {}
            for label in labels:
                defects = evaluate(
                    gradients[label],
                    reference,
                    label=label,
                    scale_pixels=PRINCIPAL_SCALE_PIXELS,
                ).as_dict()
                scores[label] = max(abs(defects[n]) / floor[n] for n in DEFECT_NAMES)
            recovered.append(min(scores, key=lambda k: scores[k]))
        hits = sum(1 for r in recovered if r == truth)
        rows.append((truth, hits, recovered[0]))
        print(
            f"  truth {truth:16s} recovered {hits}/{REPETITION_REALISATIONS}"
            f"   first pick {recovered[0]}"
        )
    total = sum(h for _, h, _ in rows)
    print(
        f"\nrecovery rate: {total}/{len(rows) * REPETITION_REALISATIONS}"
        f" = {100 * total / (len(rows) * REPETITION_REALISATIONS):.0f} %"
    )

    (OUT / f"identifiability_{profile}.json").write_text(
        json.dumps(
            {
                "profile": profile,
                "combined_floor": floor,
                "labels": labels,
                "worst_defect_in_floor_units": {
                    labels[i]: {labels[j]: worst[i, j] for j in range(len(labels))}
                    for i in range(len(labels))
                },
                "pairwise_detail": detail,
                "recovery": {t: {"hits": h, "of": REPETITION_REALISATIONS} for t, h, _ in rows},
                "recovery_rate": total / (len(rows) * REPETITION_REALISATIONS),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"written {OUT / f'identifiability_{profile}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
