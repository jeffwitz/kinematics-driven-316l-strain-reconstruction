"""Score the P43 (ell, alpha) matrix and apply the registered selection rule.

Protocol: `validation/p0043_small_parameter_matrix_preregistration.md`,
including corrections C1 to C4 and amendments A1 to A3.

Reads observed displacement fields only. No mechanics is rerun here.

The order matters and is enforced by the caller: section 9 validates the
indicators, then this scores the matrix. The Pareto front is built on the
**raw** defects, where the control-anchored normalisation cannot act; only the
minimax tie-break of section 10.3 uses `Z`.
"""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.validation.pareto_decision import Sense, pareto_front
from fem_inhouse.validation.selection_indicators import (
    DEFECT_NAMES,
    PRINCIPAL_SCALE_PIXELS,
    SENSITIVITY_SCALES_PIXELS,
    energy_ratio,
    evaluate,
    fluctuation_magnitude,
    minimax,
    normalise,
)
from fem_inhouse.validation.tile_bootstrap import (
    TileDesign,
    bootstrap_defects,
    prepare_fields,
)
from fem_inhouse.workflows.compare_gradient_fluctuation_criteria import (
    dic_displacement,
    gradient_on_core,
    observed_displacement,
)
from fem_inhouse.workflows.compare_observed_evm_candidates import _git_sha, _sha256

FloatArray = NDArray[np.float64]

#: Registered resampling design, amendment A2.
BOOTSTRAP_TILE_PIXELS = 49
BOOTSTRAP_TILE_SENSITIVITY = (32, 96)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260801

#: Registered stability thresholds of section 10.4.
ROBUSTLY_PREFERRED = 0.95
PREFERRED = 0.80

#: The two negative controls. `D_null` is the best of them, per indicator.
CONTROL_LABELS = ("homogeneous", "translated")

SENSES = dict.fromkeys(DEFECT_NAMES, Sense.LOWER_IS_BETTER)


@dataclass(frozen=True, slots=True)
class MatrixPoint:
    """One (ell, alpha) point and where its observation lives."""

    label: str
    alpha: float
    ell_um: float
    flow_path: Path
    converged: bool
    campaign: Path | None = None

    @property
    def achi(self) -> float:
        """``alpha * ell^2``, the direction the objective may be degenerate in."""

        return self.alpha * self.ell_um**2


#: Section 8 secondary indicators taken from the solver, reported and never
#: used for selection.
SOLVER_DIAGNOSTICS = (
    "cutbacks",
    "total_newton_iterations",
    "maximum_newton_iterations",
    "total_nonlocal_iterations",
    "mean_nonlocal_iterations",
    "maximum_nonlocal_iterations",
    "nonlocal_coupling_failures",
    "elapsed_seconds",
)


def solver_diagnostics(campaign: Path, *, partition_id: int = 43) -> dict[str, Any]:
    """Convergence diagnostics of one point, or why they are unavailable.

    A non-converged point writes no status, so its absence is the diagnostic.
    """

    status = campaign / "partitions" / f"{partition_id:04d}" / "status.json"
    if not status.is_file():
        return {"available": False, "reason": "no status written, the solve did not complete"}
    try:
        payload = json.loads(status.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"available": False, "reason": f"unreadable status: {error}"}
    diagnostics = payload.get("diagnostics", {})
    return {"available": True, "complete": bool(payload.get("complete"))} | {
        name: diagnostics.get(name) for name in SOLVER_DIAGNOSTICS
    }


def _defects_for(
    flow_path: Path,
    reference_gradient: FloatArray,
    *,
    label: str,
    scales: tuple[int, ...],
) -> tuple[dict[str, dict[str, float]], FloatArray, float]:
    gradient = gradient_on_core(observed_displacement(flow_path))
    per_scale = {
        str(scale): evaluate(
            gradient, reference_gradient, label=label, scale_pixels=scale
        ).as_dict()
        for scale in scales
    }
    magnitude = fluctuation_magnitude(gradient, scale_pixels=PRINCIPAL_SCALE_PIXELS)
    ratio = energy_ratio(
        magnitude,
        fluctuation_magnitude(reference_gradient, scale_pixels=PRINCIPAL_SCALE_PIXELS),
    )
    return per_scale, magnitude, ratio


def _null_defects(
    defects: dict[str, dict[str, dict[str, float]]],
    *,
    scale: int,
) -> tuple[dict[str, float], dict[str, str]]:
    """`D_null` per indicator: the **best** score any negative control reaches.

    Declared per indicator, as the specification requires, because the two
    controls fail in different ways and neither is uniformly the harder bar.
    """

    values: dict[str, float] = {}
    which: dict[str, str] = {}
    for name in DEFECT_NAMES:
        candidates = {
            label: defects[label][str(scale)][name]
            for label in CONTROL_LABELS
            if label in defects and np.isfinite(defects[label][str(scale)][name])
        }
        if not candidates:
            values[name] = float("nan")
            which[name] = "none"
            continue
        best = min(candidates, key=lambda label: candidates[label])
        values[name] = candidates[best]
        which[name] = best
    return values, which


def _solver_floor(
    replicate: tuple[str, Path] | None,
    reference: FloatArray,
    defects: dict[str, dict[str, dict[str, float]]],
    *,
    scales: tuple[int, ...],
) -> dict[str, Any]:
    """The solver's own reproducibility, as a floor on every heat map.

    One matrix point recomputed at 40 increments instead of 20. The spatial
    bootstrap says nothing about this: it resamples the observation, not the
    solve. Two neighbouring grid points differing by less than this floor are
    declared indistinguishable whatever the bootstrap says, because the
    difference is within what recomputing the same physics produces.
    """

    if replicate is None:
        return {"available": False, "reason": "replicate not computed"}
    twin, flow_path = replicate
    if twin not in defects or not flow_path.is_file():
        return {"available": False, "reason": f"no observation for the {twin} replicate"}
    per_scale, _magnitude, ratio = _defects_for(
        flow_path, reference, label=f"{twin}-replicate", scales=scales
    )
    principal = str(PRINCIPAL_SCALE_PIXELS)
    floor = {
        name: abs(per_scale[principal][name] - defects[twin][principal][name])
        for name in DEFECT_NAMES
    }
    return {
        "available": True,
        "twin": twin,
        "increments": "20 against 40",
        "replicate_defects": per_scale,
        "energy_ratio_49": ratio,
        "floor": floor,
    }


def _bootstrap_selection(
    magnitudes: dict[str, FloatArray],
    reference_magnitude: FloatArray,
    *,
    shape: tuple[int, int],
    self_defects: dict[str, float],
    null_defects: dict[str, float],
    design: TileDesign,
) -> dict[str, Any]:
    """How often each candidate minimises the worst normalised defect."""

    fields = {
        label: prepare_fields(
            label, magnitude, reference_magnitude, scale_pixels=PRINCIPAL_SCALE_PIXELS
        )
        for label, magnitude in magnitudes.items()
    }
    samples = bootstrap_defects(fields, shape=shape, design=design)
    labels = sorted(samples)
    scores = np.full((len(labels), design.draws), np.nan)
    for row, label in enumerate(labels):
        for draw in range(design.draws):
            raw = {name: samples[label][name][draw] for name in DEFECT_NAMES}
            scores[row, draw] = minimax(
                normalise(raw, self_defects=self_defects, null_defects=null_defects)
            )
    usable = np.isfinite(scores).all(axis=0)
    winners = np.argmin(scores[:, usable], axis=0)
    counts = np.bincount(winners, minlength=len(labels))
    total = int(usable.sum())
    frequency = {
        label: float(counts[row] / total) if total else float("nan")
        for row, label in enumerate(labels)
    }
    best = max(frequency, key=lambda label: frequency[label]) if total else None
    share = frequency.get(best, float("nan")) if best is not None else float("nan")
    if share >= ROBUSTLY_PREFERRED:
        verdict = "robustly_preferred"
    elif share >= PREFERRED:
        verdict = "preferred"
    else:
        verdict = "indistinguishable_zone"
    zone = _zone(scores, labels, usable=usable)
    return {
        "design": design.as_dict(),
        "usable_draws": total,
        "zone": zone,
        "win_frequency": frequency,
        "most_frequent": best,
        "most_frequent_share": share,
        "verdict": verdict,
        "minimax_quantiles": {
            label: {
                "q05": float(np.nanquantile(scores[row], 0.05)),
                "median": float(np.nanmedian(scores[row])),
                "q95": float(np.nanquantile(scores[row], 0.95)),
            }
            for row, label in enumerate(labels)
        },
    }


def _zone(
    scores: FloatArray,
    labels: list[str],
    *,
    usable: NDArray[np.bool_],
) -> dict[str, Any]:
    """The indistinguishable zone, from **paired** differences.

    Amendment A4. Comparing marginal quantiles is the interval-overlap fallacy:
    two 5-95 % bands can overlap while the paired difference is clearly non-zero,
    because the draws share their resampled tiles and the difference has far less
    spread than either score alone. The draws here are paired by construction, so
    the difference is what carries the information.

    A candidate joins the zone when the bootstrap interval of
    ``J_inf(candidate) - J_inf(best)`` contains zero. This is stricter than
    overlapping bands, so it tends to make the zone smaller and the campaign more
    decisive; that direction is why it is recorded rather than quietly applied.

    The zone stays an explicit point list, never a range of `ell` crossed with a
    range of `alpha`: a non-dominated set on a grid need not be a rectangle.
    """

    if not usable.any():
        return {"members": [], "reference": None, "differences": {}}
    medians = np.nanmedian(scores[:, usable], axis=1)
    best = int(np.nanargmin(medians))
    differences: dict[str, dict[str, float]] = {}
    members: list[str] = []
    for row, label in enumerate(labels):
        paired = scores[row, usable] - scores[best, usable]
        low = float(np.quantile(paired, 0.05))
        high = float(np.quantile(paired, 0.95))
        differences[label] = {
            "median": float(np.median(paired)),
            "q05": low,
            "q95": high,
        }
        if low <= 0.0 <= high:
            members.append(label)
    return {
        "members": sorted(members),
        "reference": labels[best],
        "differences": differences,
    }


def select_p0043_parameters(
    *,
    prepared_case: str | Path,
    points: list[MatrixPoint],
    controls: dict[str, str | Path],
    self_defects: dict[str, float],
    replicate: tuple[str, Path] | None = None,
    output_directory: str | Path,
    profile: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Score the matrix, build the front, apply the minimax, test stability."""

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    reference = gradient_on_core(dic_displacement(Path(prepared_case)))
    reference_magnitude = fluctuation_magnitude(reference, scale_pixels=PRINCIPAL_SCALE_PIXELS)
    scales = (PRINCIPAL_SCALE_PIXELS, *SENSITIVITY_SCALES_PIXELS)

    defects: dict[str, dict[str, dict[str, float]]] = {}
    magnitudes: dict[str, FloatArray] = {}
    ratios: dict[str, float] = {}
    sources: dict[str, dict[str, str]] = {}

    converged = [point for point in points if point.converged]
    for point in converged:
        per_scale, magnitude, ratio = _defects_for(
            point.flow_path, reference, label=point.label, scales=scales
        )
        defects[point.label] = per_scale
        magnitudes[point.label] = magnitude
        ratios[point.label] = ratio
        sources[point.label] = {
            "path": str(point.flow_path.resolve()),
            "sha256": _sha256(point.flow_path),
        }
    for label, path in controls.items():
        per_scale, magnitude, ratio = _defects_for(
            Path(path), reference, label=label, scales=scales
        )
        defects[label] = per_scale
        magnitudes[label] = magnitude
        ratios[label] = ratio
        sources[label] = {"path": str(Path(path).resolve()), "sha256": _sha256(Path(path))}

    null_defects, null_source = _null_defects(defects, scale=PRINCIPAL_SCALE_PIXELS)
    solver_floor = _solver_floor(replicate, reference, defects, scales=scales)

    selectable = [point.label for point in converged]
    raw = {label: dict(defects[label][str(PRINCIPAL_SCALE_PIXELS)]) for label in selectable}
    normalised = {
        label: normalise(values, self_defects=self_defects, null_defects=null_defects)
        for label, values in raw.items()
    }
    scores = {label: minimax(values) for label, values in normalised.items()}

    # Registered C3: the front is built on the raw defects, where the
    # control-anchored normalisation cannot act. Domination is invariant under
    # any per-indicator monotone rescaling, so the front is the same either way;
    # computing it on raw values makes that invariance visible.
    front, dominated = pareto_front(raw, senses=SENSES)

    bootstrap = _bootstrap_selection(
        {label: magnitudes[label] for label in selectable},
        reference_magnitude,
        shape=reference_magnitude.shape,
        self_defects=self_defects,
        null_defects=null_defects,
        design=TileDesign(BOOTSTRAP_TILE_PIXELS, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
    )
    sensitivity = {
        str(tile): _bootstrap_selection(
            {label: magnitudes[label] for label in selectable},
            reference_magnitude,
            shape=reference_magnitude.shape,
            self_defects=self_defects,
            null_defects=null_defects,
            design=TileDesign(tile, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
        )
        for tile in BOOTSTRAP_TILE_SENSITIVITY
    }

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed_p0043_parameter_selection",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "preregistration": "validation/p0043_small_parameter_matrix_preregistration.md",
        "profile": profile,
        "mechanics_rerun": False,
        "principal_scale_pixels": PRINCIPAL_SCALE_PIXELS,
        "points": [
            {
                "label": point.label,
                "alpha": point.alpha,
                "ell_um": point.ell_um,
                "achi": point.achi,
                "converged": point.converged,
            }
            for point in points
        ],
        "non_converged": [point.label for point in points if not point.converged],
        "solver_diagnostics": {
            point.label: solver_diagnostics(point.campaign)
            for point in points
            if point.campaign is not None
        },
        "self_defects": self_defects,
        "null_defects": null_defects,
        "null_defect_source": null_source,
        "solver_reproducibility": solver_floor,
        "defects": defects,
        "energy_ratio_49": ratios,
        "raw_table": raw,
        "normalised_table": normalised,
        "minimax": scores,
        "pareto_front": front,
        "pareto_dominated": dominated,
        "bootstrap": bootstrap,
        "bootstrap_tile_sensitivity": sensitivity,
        "zone": bootstrap["zone"]["members"],
        "zone_detail": bootstrap["zone"],
        "iso_achi_pairs": _iso_achi(points, scores),
        "sources": sources,
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }
    report["conclusion"] = _conclusion(report)
    _write_outputs(output, report, points=points)
    _figures(output, report, points=points)
    (output / "selection_report.md").write_text(_markdown(report, points=points), encoding="utf-8")
    (output / "selection_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return report


def _iso_achi(points: list[MatrixPoint], scores: dict[str, float]) -> list[dict[str, Any]]:
    """Pairs sharing `Achi = alpha * ell^2`, and whether they separate."""

    groups: dict[float, list[MatrixPoint]] = {}
    for point in points:
        groups.setdefault(round(point.achi, 6), []).append(point)
    pairs = []
    for achi, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        usable = [m for m in members if m.converged and np.isfinite(scores.get(m.label, np.nan))]
        pairs.append(
            {
                "achi": achi,
                "members": [m.label for m in members],
                "comparable": len(usable) == len(members),
                "minimax": {m.label: scores.get(m.label) for m in members},
                "separation": (
                    abs(scores[usable[0].label] - scores[usable[1].label])
                    if len(usable) == 2
                    else None
                ),
            }
        )
    return pairs


def _heatmap_grid(
    points: list[MatrixPoint],
    values: dict[str, float],
) -> tuple[FloatArray, list[float], list[float], list[tuple[int, int]]]:
    """Lay the points out on the (ell, alpha) grid, and say which are missing."""

    ells = sorted({point.ell_um for point in points})
    alphas = sorted({point.alpha for point in points})
    grid = np.full((len(alphas), len(ells)), np.nan)
    absent: list[tuple[int, int]] = []
    for point in points:
        row = alphas.index(point.alpha)
        column = ells.index(point.ell_um)
        value = values.get(point.label, float("nan"))
        if point.converged and np.isfinite(value):
            grid[row, column] = value
        else:
            absent.append((row, column))
    return grid, alphas, ells, absent


def _figures(output: Path, report: dict[str, Any], *, points: list[MatrixPoint]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    named = {
        "shape_indicator_heatmap.png": ("D_shape", "shape defect, 1 - Pearson"),
        "amplitude_indicator_heatmap.png": ("D_amplitude", "amplitude defect, |log q95 ratio|"),
        "localisation_indicator_heatmap.png": ("D_localisation", "localisation defect, 1 - FSS"),
        "presence_indicator_heatmap.png": ("D_presence", "presence defect, |log energy ratio|"),
    }
    floor = report["solver_reproducibility"].get("floor", {})
    for filename, (field, title) in named.items():
        values = {label: row[field] for label, row in report["raw_table"].items()}
        _draw_heatmap(
            plt,
            output / filename,
            points,
            values,
            title=title,
            floor=floor.get(field),
        )

    _draw_heatmap(
        plt,
        output / "minimax_score_heatmap.png",
        points,
        report["minimax"],
        title="worst normalised defect, minimax J_inf, lower is better",
    )

    front = set(report["pareto_front"])
    fig, ax = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    for label, row in sorted(report["raw_table"].items()):
        on_front = label in front
        ax.scatter(
            row["D_shape"],
            row["D_presence"],
            s=90 if on_front else 40,
            facecolor="crimson" if on_front else "lightgrey",
            edgecolor="black",
            zorder=3 if on_front else 2,
        )
        ax.annotate(
            label.replace("-ell", "/"),
            (row["D_shape"], row["D_presence"]),
            fontsize=6.5,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set_xlabel("D_shape")
    ax.set_ylabel("D_presence")
    ax.set_title(
        "Pareto front on the raw defects, red.\n"
        "Two of four coordinates shown; domination is judged on all four.",
        fontsize=9,
    )
    fig.savefig(output / "pareto_front.png", dpi=130)
    plt.close(fig)

    quantiles = report["bootstrap"]["minimax_quantiles"]
    frequency = report["bootstrap"]["win_frequency"]
    order = sorted(quantiles, key=lambda label: quantiles[label]["median"])
    zone = set(report["zone"])
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    positions = np.arange(len(order))
    medians = [quantiles[label]["median"] for label in order]
    lower = [quantiles[label]["median"] - quantiles[label]["q05"] for label in order]
    upper = [quantiles[label]["q95"] - quantiles[label]["median"] for label in order]
    ax.errorbar(
        positions,
        medians,
        yerr=[lower, upper],
        fmt="o",
        capsize=3,
        color="black",
        ecolor="grey",
    )
    for position, label in zip(positions, order, strict=True):
        if label in zone:
            ax.axvspan(position - 0.4, position + 0.4, color="gold", alpha=0.25, zorder=0)
        ax.annotate(
            f"{frequency.get(label, 0.0):.0%}",
            (position, quantiles[label]["q95"]),
            fontsize=7,
            ha="center",
            va="bottom",
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [label.replace("-ell", "/") for label in order], rotation=45, ha="right", fontsize=7
    )
    ax.set_ylabel("minimax J_inf")
    ax.set_title(
        "Selection stability: median and 5-95 % bootstrap band, "
        "percentage = share of draws won.\nShaded = the indistinguishable zone.",
        fontsize=9,
    )
    fig.savefig(output / "selection_stability.png", dpi=130)
    plt.close(fig)


def _draw_heatmap(
    plt: Any,
    path: Path,
    points: list[MatrixPoint],
    values: dict[str, float],
    *,
    title: str,
    floor: float | None = None,
) -> None:
    grid, alphas, ells, absent = _heatmap_grid(points, values)
    fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    image = ax.imshow(grid, cmap="viridis_r", origin="lower", aspect="auto")
    ax.set_xticks(range(len(ells)))
    ax.set_xticklabels([f"{value:g}" for value in ells])
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f"{value:g}" for value in alphas])
    ax.set_xlabel("ell (um)")
    ax.set_ylabel("alpha")
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            if np.isfinite(grid[row, column]):
                ax.text(
                    column,
                    row,
                    f"{grid[row, column]:.3g}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white",
                )
    for row, column in absent:
        ax.text(column, row, "no\nconv.", ha="center", va="center", fontsize=7, color="crimson")
    fig.colorbar(image, ax=ax, shrink=0.85)
    if floor is not None and np.isfinite(floor):
        # Neighbouring points closer than this are within what recomputing the
        # same physics produces, so the map must not be read below it.
        title = f"{title}\nsolver reproducibility floor: {floor:.3g}"
    ax.set_title(title, fontsize=9)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _write_outputs(output: Path, report: dict[str, Any], *, points: list[MatrixPoint]) -> None:
    geometry = {point.label: point for point in points}

    with (output / "parameter_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "alpha", "ell_um", "achi", "converged"])
        for point in points:
            writer.writerow(
                [point.label, point.alpha, point.ell_um, f"{point.achi:.6g}", point.converged]
            )

    for name, table in (
        ("indicator_matrix.csv", report["raw_table"]),
        ("normalised_indicator_matrix.csv", report["normalised_table"]),
    ):
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["label", "alpha", "ell_um", *DEFECT_NAMES, "minimax"])
            for label in sorted(table):
                # Controls carry no (ell, alpha), so their geometry columns
                # stay empty rather than being invented.
                located = geometry.get(label)
                writer.writerow(
                    [
                        label,
                        located.alpha if located is not None else "",
                        located.ell_um if located is not None else "",
                        *(f"{table[label][field]:.6g}" for field in DEFECT_NAMES),
                        f"{report['minimax'].get(label, float('nan')):.6g}",
                    ]
                )

    with (output / "bootstrap_selection.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "win_frequency", "minimax_q05", "minimax_median", "minimax_q95"])
        quantiles = report["bootstrap"]["minimax_quantiles"]
        for label in sorted(report["bootstrap"]["win_frequency"]):
            row = quantiles[label]
            writer.writerow(
                [
                    label,
                    f"{report['bootstrap']['win_frequency'][label]:.4f}",
                    f"{row['q05']:.6g}",
                    f"{row['median']:.6g}",
                    f"{row['q95']:.6g}",
                ]
            )

    (output / "pareto_front.json").write_text(
        json.dumps(
            {
                "front": report["pareto_front"],
                "dominated": report["pareto_dominated"],
                "senses": {name: "lower_is_better" for name in DEFECT_NAMES},
                "computed_on": "raw defects, per correction C3",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _conclusion(report: dict[str, Any]) -> dict[str, Any]:
    """One of the three registered outcomes of section 12, and nothing else.

    Case A needs a robustly preferred candidate that also rejects both negative
    controls; a candidate no better than a control is not a parameterisation.
    """

    bootstrap = report["bootstrap"]
    zone = report["zone"]
    best = bootstrap["most_frequent"]
    scores = report["minimax"]
    controls = [
        report["defects"][label][str(PRINCIPAL_SCALE_PIXELS)]
        for label in CONTROL_LABELS
        if label in report["defects"]
    ]

    def beats_controls(label: str | None) -> bool:
        if label is None or not np.isfinite(scores.get(label, np.nan)):
            return False
        candidate = report["raw_table"][label]
        return all(
            any(candidate[name] < control[name] for name in DEFECT_NAMES) for control in controls
        )

    rejects = beats_controls(best)
    if bootstrap["verdict"] == "robustly_preferred" and rejects:
        case = "A_robust_optimum"
        statement = (
            f"{best} stably minimises the worst normalised defect "
            f"({bootstrap['most_frequent_share']:.1%} of draws) and is better than "
            "both negative controls on at least one indicator each. It becomes the "
            "provisional P43 parameterisation."
        )
    elif len(zone) > 1 or bootstrap["verdict"] == "preferred":
        case = "B_robust_zone"
        statement = (
            "Several configurations are indistinguishable and form a compromise "
            f"zone of {len(zone)} points, reported as an explicit list rather than "
            "a rectangle. No point is chosen."
        )
    else:
        case = "C_indicators_not_selective"
        statement = (
            "The indicators do not distinguish the configurations or do not "
            "reject the negative controls. No parameterisation is selected and no "
            "new criterion development is opened."
        )
    return {
        "case": case,
        "statement": statement,
        "most_frequent": best,
        "most_frequent_share": bootstrap["most_frequent_share"],
        "beats_both_controls": rejects,
        "zone": zone,
    }


def _markdown(report: dict[str, Any], *, points: list[MatrixPoint]) -> str:
    """A factual summary. Interpretation belongs in the validation document."""

    geometry = {point.label: point for point in points}
    lines = [
        "# P43 (ell, alpha) selection — machine summary",
        "",
        f"Profile `{report['profile']}`, principal scale "
        f"`{report['principal_scale_pixels']} px`, generated "
        f"{report['created_at_utc']}.",
        "",
        f"**Registered outcome: {report['conclusion']['case']}.** "
        f"{report['conclusion']['statement']}",
        "",
        "## Defects at the principal scale",
        "",
        "| point | alpha | ell | " + " | ".join(DEFECT_NAMES) + " | minimax |",
        "|---|---:|---:|" + "---:|" * (len(DEFECT_NAMES) + 1),
    ]
    for label in sorted(report["raw_table"]):
        row = report["raw_table"][label]
        located = geometry.get(label)
        lines.append(
            f"| {label} | "
            + (f"{located.alpha:g}" if located else "")
            + " | "
            + (f"{located.ell_um:g}" if located else "")
            + " | "
            + " | ".join(f"{row[name]:.4g}" for name in DEFECT_NAMES)
            + f" | {report['minimax'].get(label, float('nan')):.4g} |"
        )

    lines += [
        "",
        f"Non-converged, excluded: {report['non_converged'] or 'none'}.",
        "",
        "## Front, stability and zone",
        "",
        f"- Pareto front on the raw defects: {report['pareto_front']}",
        f"- most frequent minimax winner: {report['bootstrap']['most_frequent']} "
        f"at {report['bootstrap']['most_frequent_share']:.1%} of "
        f"{report['bootstrap']['usable_draws']} usable draws",
        f"- bootstrap verdict: {report['bootstrap']['verdict']}",
        f"- indistinguishable zone: {report['zone']}",
        "",
        "## Iso-Achi pairs",
        "",
    ]
    for pair in report["iso_achi_pairs"]:
        separation = pair["separation"]
        lines.append(
            f"- `Achi = {pair['achi']:g}`: {pair['members']}, "
            + (
                f"minimax separation {separation:.4g}"
                if separation is not None
                else "not comparable, a member did not converge"
            )
        )
    lines += [
        "",
        "## What this does not license",
        "",
        "P43 only. A provisional reconstruction parameterisation, not an internal "
        "length of 316L, not transferability, and neither a validation nor a "
        "refutation of the nonlocal formulation.",
        "",
    ]
    return "\n".join(lines)
