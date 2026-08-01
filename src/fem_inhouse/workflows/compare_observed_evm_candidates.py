"""Run the registered observed-EVM candidate comparison on archived results.

Executes the protocol of
`validation/observed_evm_candidate_comparison_preregistration.md`. It reads
archived fields only: no mechanics, no material parameter selection, no
modification of any archived result.

The geometry is built from the DIC alone and frozen, so no candidate can move
the objects it is judged against.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.validation.band_geometry import (
    band_corridor,
    label_band_objects,
    order_centreline,
    prune_skeleton_spurs,
    quantile_thresholds,
    resample_polyline,
    smooth_centreline,
    tangents_and_normals,
    zhang_suen_thinning,
)
from fem_inhouse.validation.band_profiles import (
    continuity_metrics,
    estimate_background,
    measure_amplitude,
    measure_position,
    measure_width,
    sample_normal_profile,
    summarise,
)
from fem_inhouse.validation.fractions_skill_score import (
    DEFAULT_SCALES_PIXELS,
    minimum_skilful_scale,
    skill_curve,
)
from fem_inhouse.validation.otsu_morphology import (
    describe_morphology,
    morphology_distance,
    otsu_threshold,
)
from fem_inhouse.validation.pareto_decision import EliminationRule, Sense, decide
from fem_inhouse.validation.residual_structure import (
    classify_residual,
    energy_partition,
    residual_associations,
    signed_residual,
)
from fem_inhouse.validation.spatial_bootstrap import (
    BootstrapDesign,
    compare_pair,
    paired_band_bootstrap,
)

FloatArray = NDArray[np.float64]

# Registered constants, all fixed in the preregistration before this ran.
MINIMUM_AREA_PIXELS = 256
MTF50_PIXELS = 49.0
PRUNE_PIXELS = 16
SMOOTH_WINDOW = 9
SECTION_SPACING_PIXELS = 4.0
SECTION_HALF_LENGTH_PIXELS = 40.0
CORRIDOR_HALF_WIDTH_PIXELS = 12.0
BORDER_MARGIN_PIXELS = 4.0
BOOTSTRAP_BLOCK = 8
BOOTSTRAP_BLOCKS_SENSITIVITY = (4, 16)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260731
FSS_LEVEL = 0.7

#: Spurious EVM RMS of the repeated final DIC pair, from the measurement-chain
#: campaign. A section counts as carrying a band only when its excess peak
#: exceeds three times this, so "detected" means detected above the chain's own
#: reproducibility rather than merely above the local background.
CHAIN_NOISE_EVM = 1.363e-4
DETECTION_FLOOR_EVM = 3.0 * CHAIN_NOISE_EVM

#: Registered E2: a band must be carried by at least half of its sections.
MINIMUM_DETECTED_FRACTION = 0.50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def extract_bands(
    dic: FloatArray,
    *,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Build the frozen band geometry from the DIC field alone.

    Selection is automatic: an object is a band when its centreline reaches the
    chain's MTF-50, since a region whose main axis is shorter than the
    resolution length cannot be asserted to be a band.
    """

    labels, objects = label_band_objects(
        dic,
        threshold_value=threshold,
        threshold_quantile=float("nan"),
        minimum_area_pixels=MINIMUM_AREA_PIXELS,
    )
    bands: dict[str, dict[str, Any]] = {}
    for item in objects:
        mask = labels == item.identifier
        skeleton = prune_skeleton_spurs(
            zhang_suen_thinning(mask), minimum_branch_pixels=PRUNE_PIXELS
        )
        try:
            raw = order_centreline(skeleton)
        except ValueError:
            continue
        length = float(np.sum(np.linalg.norm(np.diff(raw, axis=0), axis=1)))
        if length < MTF50_PIXELS:
            continue
        centreline = resample_polyline(
            smooth_centreline(raw, window=SMOOTH_WINDOW),
            spacing_pixels=SECTION_SPACING_PIXELS,
        )
        _, normals = tangents_and_normals(centreline)
        bands[f"band{item.identifier}"] = {
            "object": item,
            "mask": mask,
            "centreline": centreline,
            "normals": normals,
            "raw_length_pixels": length,
            "corridor": band_corridor(
                dic.shape, centreline, half_width_pixels=CORRIDOR_HALF_WIDTH_PIXELS
            ),
        }
    return bands


def section_metrics(
    field: FloatArray,
    band: dict[str, Any],
) -> dict[str, list[float]]:
    """Measure every registered section quantity of one field on one band."""

    peaks: list[float] = []
    widths: list[float] = []
    positions: list[float] = []
    masses: list[float] = []
    detected: list[bool] = []
    valid: list[bool] = []
    for index, (origin, normal) in enumerate(
        zip(band["centreline"], band["normals"], strict=True)
    ):
        profile = sample_normal_profile(
            field,
            origin=(float(origin[0]), float(origin[1])),
            normal=(float(normal[0]), float(normal[1])),
            half_length_pixels=SECTION_HALF_LENGTH_PIXELS,
            section_id=index,
            border_margin_pixels=BORDER_MARGIN_PIXELS,
        )
        if not profile.valid:
            peaks.append(np.nan)
            widths.append(np.nan)
            positions.append(np.nan)
            masses.append(np.nan)
            detected.append(False)
            valid.append(False)
            continue
        background = estimate_background(
            profile, corridor_half_width_pixels=CORRIDOR_HALF_WIDTH_PIXELS
        )
        width = measure_width(profile, background)
        position = measure_position(profile, background)
        amplitude = measure_amplitude(
            profile, background, corridor_half_width_pixels=CORRIDOR_HALF_WIDTH_PIXELS
        )
        peaks.append(amplitude["peak"])
        widths.append(width.integral_pixels)
        positions.append(position["centroid_offset"])
        masses.append(amplitude["mass"])
        # Not "the excess is positive", which is true of almost any profile
        # because the background is a median of the tails, but "the excess
        # rises above the chain's own reproducibility".
        detected.append(bool(amplitude["peak"] > DETECTION_FLOOR_EVM))
        valid.append(True)
    return {
        "peak": peaks,
        "integral_width": widths,
        "centroid_offset": positions,
        "mass": masses,
        "detected": [float(v) for v in detected],
        # Kept apart from detection on purpose. A section can be unusable
        # because the band runs near the support edge, which is geometry, or
        # because no band is there, which is the result. Merging them made the
        # DIC score 0.46 against its own bands.
        "valid": [float(v) for v in valid],
    }


def continuity_on_valid_sections(
    *,
    detected: NDArray[np.bool_],
    valid: NDArray[np.bool_],
) -> dict[str, float]:
    """Continuity over usable sections, with unusable ones reported apart.

    Counting a section that leaves the support as "band not detected" makes the
    measure report geometry rather than agreement: on this ROI the DIC scores
    0.46 against its own first band that way, because more than half of its
    sections reach the core edge.
    """

    usable = int(np.count_nonzero(valid))
    if usable:
        base = continuity_metrics(
            detected[valid], spacing_pixels=SECTION_SPACING_PIXELS
        )
    else:
        base = {
            "detected_fraction": float("nan"),
            "detected_length_pixels": 0.0,
            "gap_count": float("nan"),
            "longest_gap_pixels": float("nan"),
        }
    return base | {
        "section_count": float(valid.size),
        "valid_section_count": float(usable),
        "valid_fraction": float(usable / valid.size) if valid.size else float("nan"),
    }


def compare_observed_evm_candidates(
    *,
    dic_evm: str | Path,
    candidates: dict[str, str | Path],
    output_directory: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute the registered comparison and write its machine-readable result."""

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if len(candidates) < 2:
        raise ValueError("at least two candidates are required")

    dic_path = Path(dic_evm)
    dic = np.asarray(np.load(dic_path, allow_pickle=False), dtype=np.float64)
    threshold = otsu_threshold(dic)
    bands = extract_bands(dic, threshold=threshold)
    if not bands:
        raise ValueError("no band survived the registered selection rule")

    fss_thresholds = quantile_thresholds(dic, quantiles=(0.80, 0.90, 0.95))
    reference_morphology = describe_morphology(
        dic, threshold=threshold, label_name="dic", minimum_area_pixels=MINIMUM_AREA_PIXELS
    )
    dic_sections = {name: section_metrics(dic, band) for name, band in bands.items()}
    all_corridors = np.zeros(dic.shape, dtype=bool)
    for band in bands.values():
        all_corridors |= band["corridor"]

    per_candidate: dict[str, Any] = {}
    section_errors: dict[str, dict[str, dict[str, list[float]]]] = {}
    for label, path in candidates.items():
        field = np.asarray(np.load(Path(path), allow_pickle=False), dtype=np.float64)
        if field.shape != dic.shape:
            raise ValueError(f"{label} does not share the DIC support")

        morphology = describe_morphology(
            field,
            threshold=threshold,
            label_name=label,
            minimum_area_pixels=MINIMUM_AREA_PIXELS,
        )
        residual = signed_residual(dic, field)
        energy = energy_partition(residual, corridor=all_corridors)
        flanks = (
            band_corridor(
                dic.shape,
                np.concatenate([b["centreline"] for b in bands.values()]),
                half_width_pixels=3.0 * CORRIDOR_HALF_WIDTH_PIXELS,
            )
            & ~all_corridors
        )

        curves = {}
        for quantile, value in sorted(fss_thresholds.items()):
            curve = skill_curve(
                dic,
                field,
                threshold_value=value,
                threshold_quantile=quantile,
                scales_pixels=DEFAULT_SCALES_PIXELS,
            )
            curves[f"q{int(quantile * 100)}"] = {
                "values": [float(v) for v in curve.values],
                "minimum_skilful_scale_0p7": minimum_skilful_scale(curve, level=FSS_LEVEL),
            }

        errors: dict[str, dict[str, list[float]]] = {}
        for name, band in bands.items():
            candidate_sections = section_metrics(field, band)
            reference_sections = dic_sections[name]
            errors[name] = {
                "mass_error": [
                    abs(c - r) for c, r in zip(
                        candidate_sections["mass"], reference_sections["mass"], strict=True
                    )
                ],
                "centreline_error": [abs(v) for v in candidate_sections["centroid_offset"]],
                "width_error": [
                    abs(c - r) for c, r in zip(
                        candidate_sections["integral_width"],
                        reference_sections["integral_width"],
                        strict=True,
                    )
                ],
                "detected": candidate_sections["detected"],
                "valid": candidate_sections["valid"],
            }
        section_errors[label] = errors

        per_candidate[label] = {
            "morphology": {
                "active_fraction": morphology.active_fraction,
                "object_count": morphology.object_count,
                "objects": [
                    {
                        "rank": o.rank,
                        "area_pixels": o.area_pixels,
                        "eccentricity": o.eccentricity,
                        "solidity": o.solidity,
                        "axis_major_pixels": o.axis_major_pixels,
                        "axis_minor_pixels": o.axis_minor_pixels,
                        "orientation_degrees": o.orientation_degrees,
                        "euler_number": o.euler_number,
                    }
                    for o in morphology.objects
                ],
            },
            "morphology_distance": morphology_distance(reference_morphology, morphology),
            "residual": {
                "corridor_energy_fraction": energy.corridor_fraction,
                "associations": residual_associations(residual, dic),
                "typology": classify_residual(
                    residual, corridor=all_corridors, flanks=flanks
                ),
            },
            "fss": curves,
            "bands": {
                name: {
                    key: summarise(np.asarray(values))
                    for key, values in errors[name].items()
                    if key not in {"detected", "valid"}
                }
                | {
                    "continuity": continuity_on_valid_sections(
                        detected=np.asarray(errors[name]["detected"], dtype=bool),
                        valid=np.asarray(errors[name]["valid"], dtype=bool),
                    )
                }
                for name in bands
            },
            "source": {"path": str(Path(path).resolve()), "sha256": _sha256(Path(path))},
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed_observed_evm_candidate_comparison",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "preregistration": (
            "validation/observed_evm_candidate_comparison_preregistration.md"
        ),
        "mechanics_rerun": False,
        "segmentation": {
            "method": "otsu_on_dic_frozen",
            "threshold": threshold,
            "minimum_area_pixels": MINIMUM_AREA_PIXELS,
            "selection_rule": f"centreline >= MTF-50 = {MTF50_PIXELS} px",
            "bands": {
                name: {
                    "area_pixels": band["object"].area_pixels,
                    "centreline_length_pixels": band["raw_length_pixels"],
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
        "candidates": per_candidate,
        "source": {"dic_evm": str(dic_path.resolve()), "dic_sha256": _sha256(dic_path)},
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }

    report["bootstrap"] = _bootstrap_block(section_errors, bands)
    report["decision"] = _decide_block(per_candidate, bands)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return report


def _bootstrap_block(
    section_errors: dict[str, dict[str, dict[str, list[float]]]],
    bands: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Paired block bootstrap and every ordered pairwise comparison."""

    labels = sorted(section_errors)
    results: dict[str, Any] = {"design": {}, "pairs": []}
    for metric in ("mass_error", "centreline_error", "width_error"):
        per_band = {
            name: {
                label: np.asarray(section_errors[label][name][metric], dtype=np.float64)
                for label in labels
            }
            for name in bands
        }
        for block in (BOOTSTRAP_BLOCK, *BOOTSTRAP_BLOCKS_SENSITIVITY):
            design = BootstrapDesign(
                block_length=block, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
            )
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
                            "lower_95": outcome.lower_95,
                            "upper_95": outcome.upper_95,
                            "probability_first_better": outcome.probability_first_better,
                            "decision": outcome.decision,
                        }
                    )
    return results


def _decide_block(
    per_candidate: dict[str, Any],
    bands: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Reduce to the registered Pareto criteria, all on the worst band."""

    senses = {
        "worst_mass_error": Sense.LOWER_IS_BETTER,
        "worst_centreline_error": Sense.LOWER_IS_BETTER,
        "worst_width_error": Sense.LOWER_IS_BETTER,
        "minimum_skilful_scale_q90": Sense.LOWER_IS_BETTER,
        "corridor_energy_fraction": Sense.LOWER_IS_BETTER,
        "worst_detected_fraction": Sense.HIGHER_IS_BETTER,
    }
    table: dict[str, dict[str, float]] = {}
    for label, payload in per_candidate.items():
        per_band = payload["bands"]
        table[label] = {
            "worst_mass_error": max(
                per_band[b]["mass_error"]["median"] for b in bands
            ),
            "worst_centreline_error": max(
                per_band[b]["centreline_error"]["median"] for b in bands
            ),
            "worst_width_error": max(
                per_band[b]["width_error"]["median"] for b in bands
            ),
            "minimum_skilful_scale_q90": payload["fss"]["q90"][
                "minimum_skilful_scale_0p7"
            ],
            "corridor_energy_fraction": payload["residual"]["corridor_energy_fraction"],
            "worst_detected_fraction": min(
                per_band[b]["continuity"]["detected_fraction"] for b in bands
            ),
        }
    rules = (
        EliminationRule(
            "worst_detected_fraction",
            Sense.HIGHER_IS_BETTER,
            MINIMUM_DETECTED_FRACTION,
            "registered E2: each DIC band carried by at least half its sections",
        ),
    )
    report = decide(table, senses=senses, rules=rules)
    return {
        "criteria_table": table,
        "eliminated": report.eliminated,
        "survivors": report.survivors,
        "non_dominated": report.non_dominated,
        "dominated": report.dominated,
        "conclusion": report.conclusion,
        "note": (
            "E2 is applied as registered; E5 was withdrawn to reported-only by "
            "amendment 1 and is reported in the morphology block"
        ),
    }
