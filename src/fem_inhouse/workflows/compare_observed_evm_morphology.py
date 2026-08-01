"""Criteria set v2, with morphology, on archived observed-EVM fields.

Executes `validation/observed_evm_morphology_criteria_preregistration.md`,
including both amendments of 2026-08-01. Archived fields only: no mechanics, no
material parameter selection, no archived result modified.

Why a second set: the section-based criteria of v1 could not reject a negative
control built by displacing the material maps. Morphology separates it
immediately. That set is registered here, together with the two gates that have
to pass before any candidate is scored, since the criteria themselves were
chosen knowing what they do to that control.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.validation.band_geometry import quantile_thresholds
from fem_inhouse.validation.falsification_cases import (
    PerturbedField,
    add_spurious_band,
    standard_cases,
)
from fem_inhouse.validation.fractions_skill_score import (
    DEFAULT_SCALES_PIXELS,
    minimum_skilful_scale,
    skill_curve,
)
from fem_inhouse.validation.otsu_morphology import (
    FieldMorphology,
    describe_morphology,
    morphology_distance,
    otsu_threshold,
)
from fem_inhouse.validation.pareto_decision import EliminationRule, Sense, decide
from fem_inhouse.validation.residual_structure import energy_partition, signed_residual
from fem_inhouse.validation.spatial_bootstrap import (
    BootstrapDesign,
    compare_pair,
    paired_band_bootstrap,
)
from fem_inhouse.workflows.compare_observed_evm_candidates import (
    BOOTSTRAP_BLOCK,
    BOOTSTRAP_BLOCKS_SENSITIVITY,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    FSS_LEVEL,
    MINIMUM_AREA_PIXELS,
    MINIMUM_DETECTED_FRACTION,
    _git_sha,
    _sha256,
    continuity_on_valid_sections,
    extract_bands,
    section_metrics,
)

FloatArray = NDArray[np.float64]

#: Registered severity of each falsification case, most severe first. Only
#: these five are placed by the preregistration, so only these enter the G1
#: statistic; the rest are computed and reported.
REGISTERED_SEVERITY: dict[str, int] = {
    "band_removed": 4,
    "band_spurious": 3,
    "shift_16px": 2,
    "width_0p80": 1,
    "width_1p20": 1,
    "amplitude_0p90": 0,
}

#: The v2 Pareto criteria and their senses. Morphology contributes three: a
#: candidate can get the object count right and the width wrong, or both right
#: and the shape wrong, so collapsing them would hide what v1 missed.
CRITERION_SENSES: dict[str, Sense] = {
    "object_count_error": Sense.LOWER_IS_BETTER,
    "log_minor_axis_ratio": Sense.LOWER_IS_BETTER,
    "eccentricity_error": Sense.LOWER_IS_BETTER,
    "worst_centreline_error": Sense.LOWER_IS_BETTER,
    "worst_mass_error": Sense.LOWER_IS_BETTER,
    "minimum_skilful_scale_q90": Sense.LOWER_IS_BETTER,
    "corridor_energy_fraction": Sense.LOWER_IS_BETTER,
}

#: Registered E3: a field with no object cannot be compared on morphology.
MINIMUM_OBJECT_COUNT = 1


@dataclass(frozen=True, slots=True)
class FrozenReference:
    """Everything derived from the DIC alone, built once and never rebuilt.

    Keeping it in one object is what guarantees that no candidate can move the
    geometry, the threshold or the sections it is judged against.
    """

    dic: FloatArray
    bands: dict[str, dict[str, Any]]
    sections: dict[str, dict[str, list[float]]]
    morphology: FieldMorphology
    threshold: float
    fss_thresholds: dict[float, float]
    corridor: NDArray[np.bool_]


def _kendall_tau(values: list[float], ranks: list[int]) -> float:
    """Kendall's tau between a criterion and the registered severity ranks.

    Pairs tied in severity carry no information and are skipped; pairs tied in
    the criterion count as neither concordant nor discordant, so a criterion
    that never moves lands on exactly zero and is reported as insensitive
    rather than removed.
    """

    concordant = 0
    discordant = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if ranks[i] == ranks[j]:
                continue
            if not (np.isfinite(values[i]) and np.isfinite(values[j])):
                continue
            severity = np.sign(ranks[i] - ranks[j])
            observed = np.sign(values[i] - values[j])
            if observed == 0:
                continue
            if severity == observed:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return float((concordant - discordant) / total) if total else 0.0


def _morphology_criteria(
    reference: FieldMorphology,
    candidate: FieldMorphology,
) -> dict[str, float]:
    """The three registered morphology criteria, from regionprops distances."""

    distance = morphology_distance(reference, candidate)
    ratio = distance.get("axis_minor_ratio", float("nan"))
    return {
        "object_count_error": abs(distance["object_count_difference"]),
        # In log, so twice too wide and half too wide are penalised equally.
        "log_minor_axis_ratio": (
            abs(float(np.log(ratio))) if np.isfinite(ratio) and ratio > 0.0 else float("nan")
        ),
        "eccentricity_error": distance.get("eccentricity_error", float("nan")),
        "object_count": distance["object_count_candidate"],
        "active_fraction": distance["active_fraction_candidate"],
        "orientation_error_degrees": distance.get("orientation_error_degrees", float("nan")),
    }


def evaluate_field(
    field: FloatArray,
    *,
    reference: FrozenReference,
    label: str,
) -> dict[str, Any]:
    """Score one field against the frozen DIC geometry on the v2 criteria."""

    dic = reference.dic
    bands = reference.bands
    morphology = describe_morphology(
        field,
        threshold=reference.threshold,
        label_name=label,
        minimum_area_pixels=MINIMUM_AREA_PIXELS,
    )
    criteria = _morphology_criteria(reference.morphology, morphology)

    residual = signed_residual(dic, field)
    criteria["corridor_energy_fraction"] = energy_partition(
        residual, corridor=reference.corridor
    ).corridor_fraction

    curve = skill_curve(
        dic,
        field,
        threshold_value=reference.fss_thresholds[0.90],
        threshold_quantile=0.90,
        scales_pixels=DEFAULT_SCALES_PIXELS,
    )
    criteria["minimum_skilful_scale_q90"] = minimum_skilful_scale(curve, level=FSS_LEVEL)

    per_band: dict[str, Any] = {}
    section_errors: dict[str, dict[str, list[float]]] = {}
    for name, band in bands.items():
        candidate = section_metrics(field, band)
        band_reference = reference.sections[name]
        # Amendment 2: paired against the DIC's own offset on the same section.
        # The absolute offset used in v1 is not zero for the DIC against itself,
        # so it carried a constant geometric bias unrelated to any candidate.
        centreline = [
            abs(c - r)
            for c, r in zip(
                candidate["centroid_offset"],
                band_reference["centroid_offset"],
                strict=True,
            )
        ]
        mass = [abs(c - r) for c, r in zip(candidate["mass"], band_reference["mass"], strict=True)]
        section_errors[name] = {"centreline_error": centreline, "mass_error": mass}
        valid = np.asarray(candidate["valid"], dtype=bool)
        per_band[name] = {
            "centreline_error_median": float(np.nanmedian(np.asarray(centreline))),
            "mass_error_median": float(np.nanmedian(np.asarray(mass))),
            "continuity": continuity_on_valid_sections(
                detected=np.asarray(candidate["detected"], dtype=bool), valid=valid
            ),
        }

    criteria["worst_centreline_error"] = max(
        per_band[name]["centreline_error_median"] for name in bands
    )
    criteria["worst_mass_error"] = max(per_band[name]["mass_error_median"] for name in bands)
    criteria["worst_detected_fraction"] = min(
        per_band[name]["continuity"]["detected_fraction"] for name in bands
    )
    return {
        "criteria": criteria,
        "bands": per_band,
        "section_errors": section_errors,
        "morphology": {
            "active_fraction": morphology.active_fraction,
            "object_count": morphology.object_count,
            "objects": [
                {
                    "rank": o.rank,
                    "area_pixels": o.area_pixels,
                    "eccentricity": o.eccentricity,
                    "axis_minor_pixels": o.axis_minor_pixels,
                    "axis_major_pixels": o.axis_major_pixels,
                    "orientation_degrees": o.orientation_degrees,
                }
                for o in morphology.objects
            ],
        },
    }


def run_gate_g1(reference: FrozenReference) -> dict[str, Any]:
    """Rank known defects and remove any criterion that orders them backwards."""

    dic = reference.dic
    band_region = np.zeros(dic.shape, dtype=bool)
    for band in reference.bands.values():
        band_region |= band["mask"]
    cases: list[PerturbedField] = list(standard_cases(dic, band_region=band_region))
    # add_spurious_band is not part of standard_cases, but the registered
    # severity order names a spurious band, so it is added explicitly.
    rows, columns = dic.shape
    cases.append(
        PerturbedField(
            "band_spurious",
            add_spurious_band(
                dic,
                amplitude=float(np.nanquantile(dic, 0.95)),
                centre=(rows / 2.0, columns / 2.0),
                orientation_degrees=0.0,
                half_width_pixels=8.0,
            ),
            "spurious_band",
            1.0,
        )
    )

    per_case: dict[str, dict[str, float]] = {}
    for case in cases:
        scored = evaluate_field(
            np.ascontiguousarray(case.field, dtype=np.float64),
            reference=reference,
            label=case.name,
        )
        per_case[case.name] = {name: float(scored["criteria"][name]) for name in CRITERION_SENSES}

    ranked = [name for name in REGISTERED_SEVERITY if name in per_case]
    verdicts: dict[str, Any] = {}
    for criterion in CRITERION_SENSES:
        values = [per_case[name][criterion] for name in ranked]
        ranks = [REGISTERED_SEVERITY[name] for name in ranked]
        tau = _kendall_tau(values, ranks)
        finite = [v for v in values if np.isfinite(v)]
        constant = len(set(np.round(finite, 12))) <= 1 if finite else True
        if tau < 0.0:
            verdict = "removed_orders_defects_backwards"
        elif tau == 0.0 or constant:
            verdict = "kept_insensitive"
        else:
            verdict = "kept"
        verdicts[criterion] = {"kendall_tau": tau, "verdict": verdict}
    return {
        "registered_severity": REGISTERED_SEVERITY,
        "cases": per_case,
        "criteria": verdicts,
        "removed": sorted(c for c, v in verdicts.items() if v["verdict"].startswith("removed")),
        "insensitive": sorted(c for c, v in verdicts.items() if v["verdict"] == "kept_insensitive"),
    }


def run_gate_g2(scored: dict[str, Any]) -> dict[str, Any]:
    """The DIC against itself must be perfect on every criterion."""

    criteria = scored["criteria"]
    expected: dict[str, float] = {
        "object_count_error": 0.0,
        "log_minor_axis_ratio": 0.0,
        "eccentricity_error": 0.0,
        "worst_centreline_error": 0.0,
        "worst_mass_error": 0.0,
        "minimum_skilful_scale_q90": 1.0,
        "worst_detected_fraction": 1.0,
    }
    checks: dict[str, Any] = {}
    for name, target in expected.items():
        value = float(criteria[name])
        passed = bool(np.isfinite(value) and abs(value - target) <= 1e-9)
        checks[name] = {"value": value, "expected": target, "passed": passed}
    # The residual of a field against itself is identically zero, so its energy
    # cannot be partitioned. Undefined here is correct, not a failure.
    checks["corridor_energy_fraction"] = {
        "value": float(criteria["corridor_energy_fraction"]),
        "expected": float("nan"),
        "passed": bool(not np.isfinite(criteria["corridor_energy_fraction"])),
        "note": "no residual to partition when a field is compared with itself",
    }
    return {
        "checks": checks,
        "failed": sorted(name for name, c in checks.items() if not c["passed"]),
        "passed": all(c["passed"] for c in checks.values()),
    }


def _bootstrap(
    section_errors: dict[str, dict[str, dict[str, list[float]]]],
    bands: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Paired block bootstrap on the two section-based criteria."""

    labels = sorted(section_errors)
    results: dict[str, Any] = {"design": {}, "pairs": []}
    for metric in ("mass_error", "centreline_error"):
        per_band = {
            name: {
                label: np.asarray(section_errors[label][name][metric], dtype=np.float64)
                for label in labels
            }
            for name in bands
        }
        for block in (BOOTSTRAP_BLOCK, *BOOTSTRAP_BLOCKS_SENSITIVITY):
            design = BootstrapDesign(block_length=block, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
            samples = paired_band_bootstrap(per_band, design=design)
            results["design"][f"block{block}"] = design.as_dict()
            for i, first in enumerate(labels):
                for second in labels[i + 1 :]:
                    outcome = compare_pair(
                        samples[first],
                        samples[second],
                        metric=metric,
                        first=first,
                        second=second,
                        lower_is_better=True,
                    )
                    results["pairs"].append(
                        {
                            "metric": metric,
                            "block_length": block,
                            "first": first,
                            "second": second,
                            "median_difference": outcome.median_difference,
                            "probability_first_better": outcome.probability_first_better,
                            "decision": outcome.decision,
                        }
                    )
    return results


def compare_observed_evm_morphology(
    *,
    dic_evm: str | Path,
    candidates: dict[str, str | Path],
    output_directory: str | Path,
    profile_name: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the registered v2 campaign on one DISFlow profile."""

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    dic_path = Path(dic_evm)
    dic = np.asarray(np.load(dic_path, allow_pickle=False), dtype=np.float64)
    # Otsu is recomputed from this profile's own DIC and frozen here. Carrying
    # the first profile's threshold over would judge these fields against a
    # boundary derived from other data.
    threshold = otsu_threshold(dic)
    bands = extract_bands(dic, threshold=threshold)
    if not bands:
        raise ValueError("no band survived the registered selection rule")

    fss_thresholds = quantile_thresholds(dic, quantiles=(0.90,))
    reference_morphology = describe_morphology(
        dic, threshold=threshold, label_name="dic", minimum_area_pixels=MINIMUM_AREA_PIXELS
    )
    dic_sections = {name: section_metrics(dic, band) for name, band in bands.items()}
    corridor = np.zeros(dic.shape, dtype=bool)
    for band in bands.values():
        corridor |= band["corridor"]

    reference = FrozenReference(
        dic=dic,
        bands=bands,
        sections=dic_sections,
        morphology=reference_morphology,
        threshold=threshold,
        fss_thresholds=fss_thresholds,
        corridor=corridor,
    )

    gate_g1 = run_gate_g1(reference)
    gate_g2 = run_gate_g2(evaluate_field(dic, reference=reference, label="dic_self"))

    scored: dict[str, Any] = {}
    for label, path in sorted(candidates.items()):
        field = np.asarray(np.load(Path(path), allow_pickle=False), dtype=np.float64)
        if field.shape != dic.shape:
            raise ValueError(f"{label} does not share the DIC support")
        scored[label] = evaluate_field(field, reference=reference, label=label)
        scored[label]["source"] = {
            "path": str(Path(path).resolve()),
            "sha256": _sha256(Path(path)),
        }

    active = {
        name: sense for name, sense in CRITERION_SENSES.items() if name not in gate_g1["removed"]
    }
    table = {
        label: {name: float(payload["criteria"][name]) for name in CRITERION_SENSES}
        | {
            "worst_detected_fraction": float(payload["criteria"]["worst_detected_fraction"]),
            "object_count": float(payload["criteria"]["object_count"]),
        }
        for label, payload in scored.items()
    }
    rules = (
        EliminationRule(
            "worst_detected_fraction",
            Sense.HIGHER_IS_BETTER,
            MINIMUM_DETECTED_FRACTION,
            "registered E2: each DIC band carried by at least half its valid sections",
        ),
        EliminationRule(
            "object_count",
            Sense.HIGHER_IS_BETTER,
            float(MINIMUM_OBJECT_COUNT),
            "registered E3: a field with no object cannot be compared",
        ),
    )
    outcome = decide(table, senses=active, rules=rules)

    survivors = set(outcome.survivors)
    accepted = "translated" not in survivors or "translated" in outcome.dominated
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed_observed_evm_morphology_criteria",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "preregistration": ("validation/observed_evm_morphology_criteria_preregistration.md"),
        "profile": profile_name,
        "mechanics_rerun": False,
        "segmentation": {
            "method": "otsu_on_dic_of_this_profile_frozen",
            "threshold": threshold,
            "bands": {
                name: {
                    "area_pixels": band["object"].area_pixels,
                    "section_count": len(band["centreline"]),
                }
                for name, band in bands.items()
            },
        },
        "dic_morphology": {
            "active_fraction": reference_morphology.active_fraction,
            "object_count": reference_morphology.object_count,
            "objects": [
                {
                    "rank": o.rank,
                    "area_pixels": o.area_pixels,
                    "eccentricity": o.eccentricity,
                    "axis_minor_pixels": o.axis_minor_pixels,
                    "orientation_degrees": o.orientation_degrees,
                }
                for o in reference_morphology.objects
            ],
        },
        "gate_g1": gate_g1,
        "gate_g2": gate_g2,
        "candidates": {
            label: {k: v for k, v in payload.items() if k != "section_errors"}
            for label, payload in scored.items()
        },
        "decision": {
            "active_criteria": sorted(active),
            "criteria_table": table,
            "eliminated": outcome.eliminated,
            "survivors": outcome.survivors,
            "non_dominated": outcome.non_dominated,
            "dominated": outcome.dominated,
            "conclusion": outcome.conclusion,
            "acceptance_test_translated_control_rejected": accepted,
        },
        "source": {"dic_evm": str(dic_path.resolve()), "dic_sha256": _sha256(dic_path)},
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }
    report["bootstrap"] = _bootstrap(
        {label: payload["section_errors"] for label, payload in scored.items()}, bands
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return report
