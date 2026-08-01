"""Exploratory diagnostic: gradient-based fluctuation criteria on archived fields.

Implements the short specification of 2026-08-01. This is a **diagnostic, not a
decision procedure**: it selects no micromorphic parameter, touches neither the
v1 nor the v2 criteria set, and enters no Pareto front. Whether any of these
criteria deserves a place in an identification campaign is a question for a
separate preregistration.

Archived fields only. No mechanics is rerun.

Interpretation limit carried into every output: the archived high-fidelity
solutions use one fixed spatial range, ell = 58.88 um, with alpha in {0,1,2,4}.
The results describe one slice of the (ell, alpha) space and nothing more.
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
from scipy import ndimage

from fem_inhouse.measurement.coordinates import image_flow_to_canonical
from fem_inhouse.validation.gradient_fluctuation import (
    GAUSSIAN_SIGMA_FACTOR,
    SCALES_PIXELS,
    displacement_gradient,
    frobenius_norm,
    gradient_criteria,
    highpass_energy_ratio,
    multiscale_fluctuation,
    symmetric_part,
)
from fem_inhouse.validation.otsu_morphology import otsu_threshold
from fem_inhouse.workflows.compare_observed_evm_candidates import (
    _git_sha,
    _sha256,
    extract_bands,
)

FloatArray = NDArray[np.float64]

PIXEL_SIZE_MM = 0.00184
PARTITION_ID = 43
SOLVE_BOUNDS = (1290, 1950, 780, 1390)
CORE_BOUNDS = (1440, 1800, 930, 1240)
POISSON_RATIO = 0.3

#: The one spatial range behind every archived high-fidelity solution here.
SPATIAL_RANGE_MICROMETRES = 58.88

INTERPRETATION_LIMIT = (
    "The parameterisations tested at fixed spatial range do not necessarily "
    "reproduce the global amplitude and the spatial fluctuations "
    "simultaneously. Other combinations of spatial range and feedback "
    "intensity remain possible."
)


@dataclass(frozen=True, slots=True)
class Case:
    """One archived displacement field to score."""

    label: str
    flow_path: Path
    kind: str


def core_slice() -> tuple[slice, slice]:
    cx0, cx1, cy0, cy1 = CORE_BOUNDS
    sx0, _, sy0, _ = SOLVE_BOUNDS
    # One cell shorter on each side than the nodal core, because the gradient
    # is averaged to element centres exactly as the historical EVM operator is.
    return slice(cx0 - sx0, cx1 - sx0), slice(cy0 - sy0, cy1 - sy0)


def dic_displacement(prepared_case: Path) -> FloatArray:
    """The measured DIC displacement on the solve grid, in mm."""

    sx0, sx1, sy0, sy1 = SOLVE_BOUNDS
    ux = np.load(prepared_case / "displacement_x_mm.npy", mmap_mode="r", allow_pickle=False)
    uy = np.load(prepared_case / "displacement_y_mm.npy", mmap_mode="r", allow_pickle=False)
    return np.ascontiguousarray(
        np.stack(
            (
                np.asarray(ux[sx0 : sx1 + 1, sy0 : sy1 + 1], dtype=np.float64),
                np.asarray(uy[sx0 : sx1 + 1, sy0 : sy1 + 1], dtype=np.float64),
            ),
            axis=-1,
        )
    )


def observed_displacement(flow_path: Path) -> FloatArray:
    """The FEM displacement as the image chain and DISFlow actually see it."""

    flow = np.load(flow_path, allow_pickle=False)
    return image_flow_to_canonical(flow, pixel_size_mm=PIXEL_SIZE_MM)


def gradient_on_core(displacement: FloatArray) -> FloatArray:
    """Differentiate on the solve grid, then crop, so no edge touches the core."""

    gradient = displacement_gradient(
        displacement, spacing_x_mm=PIXEL_SIZE_MM, spacing_y_mm=PIXEL_SIZE_MM
    )
    rows, columns = core_slice()
    return np.ascontiguousarray(gradient[rows, columns])


def _synthetic_cases(
    displacement: FloatArray,
    *,
    band_region: NDArray[np.bool_],
) -> dict[str, FloatArray]:
    """The registered synthetic checks of section 6, on the DIC displacement.

    Each is a displacement-level operation, so every one of them goes through
    exactly the same differentiation as a real candidate.
    """

    shape = displacement.shape[:2]
    x = np.arange(shape[0], dtype=np.float64)[:, None] * PIXEL_SIZE_MM
    y = np.arange(shape[1], dtype=np.float64)[None, :] * PIXEL_SIZE_MM
    x = np.broadcast_to(x, shape)
    y = np.broadcast_to(y, shape)

    def affine(a: NDArray[np.float64]) -> FloatArray:
        return np.ascontiguousarray(
            np.stack((a[0, 0] * x + a[0, 1] * y, a[1, 0] * x + a[1, 1] * y), axis=-1)
        )

    mean = displacement.mean(axis=(0, 1))
    fluctuation = displacement - mean

    def smoothed(sigma: float) -> FloatArray:
        return np.ascontiguousarray(
            mean + ndimage.gaussian_filter(fluctuation, sigma=sigma, axes=(0, 1), mode="nearest")
        )

    angle = 5.0e-4
    without_band = fluctuation.copy()
    without_band[band_region] = 0.0

    return {
        "uniform_displacement": displacement + np.array([3.7e-3, -1.2e-3]),
        "affine_strain": displacement + affine(np.array([[1.0e-3, 0.0], [0.0, -3.0e-4]])),
        "rigid_rotation": displacement + affine(np.array([[0.0, -angle], [angle, 0.0]])),
        "band_translation_16px": np.ascontiguousarray(np.roll(displacement, 16, axis=0)),
        "amplitude_1p20": np.ascontiguousarray(mean + 1.20 * fluctuation),
        "width_change_sigma8": smoothed(8.0),
        "band_merge_sigma24": smoothed(24.0),
        "band_removal": np.ascontiguousarray(mean + without_band),
    }


def _figures(
    output: Path,
    *,
    reference_gradient: FloatArray,
    gradients: dict[str, FloatArray],
    multiscale: dict[str, dict[int, float]],
    energy: dict[str, dict[int, float]],
    summary: dict[str, dict[str, float]],
    existing: dict[str, dict[str, float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [label for label in gradients]
    reference_strain = symmetric_part(reference_gradient)

    def field_grid(name: str, maps: dict[str, FloatArray], title: str) -> None:
        columns = len(maps)
        fig, axes = plt.subplots(1, columns, figsize=(3.1 * columns, 3.6), constrained_layout=True)
        finite = np.concatenate([m[np.isfinite(m)].ravel() for m in maps.values()])
        vmax = float(np.quantile(finite, 0.99))
        for ax, (label, values) in zip(np.atleast_1d(axes), maps.items(), strict=True):
            image = ax.imshow(values, cmap="magma", vmin=0.0, vmax=vmax)
            ax.set_title(label, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(image, ax=np.atleast_1d(axes), shrink=0.8)
        fig.suptitle(title, fontsize=10)
        fig.savefig(output / name, dpi=120)
        plt.close(fig)

    field_grid(
        "frobenius_norm_fields.png",
        {"DIC": frobenius_norm(reference_gradient)}
        | {label: frobenius_norm(gradients[label]) for label in order},
        "Frobenius norm of the displacement gradient",
    )
    field_grid(
        "symmetric_strain_norm_fields.png",
        {"DIC": frobenius_norm(reference_strain)}
        | {label: frobenius_norm(symmetric_part(gradients[label])) for label in order},
        "Frobenius norm of the symmetric part (strain)",
    )
    field_grid(
        "gradient_residuals.png",
        {
            label: frobenius_norm(symmetric_part(gradients[label]) - reference_strain)
            for label in order
        },
        "Pointwise strain residual against the DIC",
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 4.6), constrained_layout=True)
    for label in order:
        style = "--" if label in {"homogeneous", "translated"} else "-"
        curve = multiscale[label]
        left.plot(list(curve), list(curve.values()), marker="o", label=label, linestyle=style)
        ratio = energy[label]
        right.plot(list(ratio), list(ratio.values()), marker="o", label=label, linestyle=style)
    for ax in (left, right):
        ax.set_xscale("log")
        ax.set_xticks(list(SCALES_PIXELS))
        ax.set_xticklabels([str(s) for s in SCALES_PIXELS])
        ax.set_xlabel("high-pass scale (pixels)")
        ax.axvline(49, color="grey", linewidth=0.8)
    left.axhline(1.0, color="black", linewidth=0.8)
    left.set_ylabel(r"$J_{\mathrm{fluct}}(s)$ on the strain")
    left.set_title(
        "Fluctuation distance. The line at 1 is what a field with no\n"
        "content at that scale scores, so below it is the only skill.",
        fontsize=9,
    )
    right.axhline(1.0, color="black", linewidth=0.8)
    right.set_yscale("log")
    right.set_ylabel(r"$\|H_s(\varepsilon_{\mathrm{FEM}})\| / \|H_s(\varepsilon_{\mathrm{DIC}})\|$")
    right.set_title(
        "High-pass strain energy carried, relative to the DIC.\n"
        "Every candidate is smooth where the DIC is not.",
        fontsize=9,
    )
    left.legend(fontsize=8)
    fig.suptitle("Multiscale fluctuation, dashed = negative control", fontsize=10)
    fig.savefig(output / "multiscale_fluctuation_scores.png", dpi=120)
    plt.close(fig)

    new_names = ["J_gradient", "J_strain", "J_norm_map", "J_fluctuation"]
    old_names = ["evm_relative_l2", "evm_pearson", "evm_iou_q90"]
    # Not every metric improves downwards: Pearson and IoU are higher-is-better,
    # and ranking them as if lower were better would put the homogeneous control
    # first on an IoU of exactly zero.
    higher_is_better = {"evm_pearson", "evm_iou_q90"}
    short = {
        "local": "loc",
        "alpha1": "a1",
        "alpha2": "a2",
        "alpha4": "a4",
        "homogeneous": "HOM",
        "translated": "TRA",
    }
    candidates = [label for label in order if label in existing]
    fig, ax = plt.subplots(figsize=(9.6, 4.8), constrained_layout=True)
    all_names = new_names + old_names
    for index, name in enumerate(all_names):
        source = summary if name in new_names else existing
        values = np.array([source[label].get(name, np.nan) for label in candidates], dtype=float)
        if name in higher_is_better:
            values = -values
        finite = np.isfinite(values)
        ranks = np.full(values.size, np.nan)
        ranks[finite] = np.argsort(np.argsort(values[finite])) + 1
        ax.plot(
            [index] * len(candidates),
            ranks,
            "o",
            color="lightgrey",
            markersize=17,
            zorder=1,
        )
        for position, label in enumerate(candidates):
            if np.isfinite(ranks[position]):
                ax.text(
                    index,
                    ranks[position],
                    short[label],
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold" if label in {"homogeneous", "translated"} else "normal",
                    color="crimson" if label in {"homogeneous", "translated"} else "black",
                    zorder=2,
                )
    ax.axvline(len(new_names) - 0.5, color="grey", linewidth=0.8)
    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels(all_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("rank, 1 = best")
    ax.invert_yaxis()
    ax.set_title(
        "Rank under the new gradient criteria (left) and the existing EVM "
        "metrics (right).\nNegative controls in red: a criterion that puts one "
        "of them near the top is not measuring what it claims.\n"
        "No selection is made from this figure.",
        fontsize=9,
    )
    fig.savefig(output / "criteria_ranking_comparison.png", dpi=120)
    plt.close(fig)


def compare_gradient_fluctuation_criteria(
    *,
    prepared_case: str | Path,
    cases: dict[str, str | Path],
    evm_fields: dict[str, str | Path],
    dic_evm: str | Path,
    output_directory: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Score every archived displacement field on the gradient criteria."""

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    measured = dic_displacement(Path(prepared_case))
    reference = gradient_on_core(measured)

    dic_evm_field = np.asarray(np.load(Path(dic_evm), allow_pickle=False), dtype=np.float64)
    bands = extract_bands(dic_evm_field, threshold=otsu_threshold(dic_evm_field))
    corridor_core = np.zeros(dic_evm_field.shape, dtype=bool)
    for band in bands.values():
        corridor_core |= band["corridor"]
    # Lift the core-sized corridor back onto the solve grid the synthetic
    # perturbations act on.
    band_region = np.zeros(measured.shape[:2], dtype=bool)
    rows, columns = core_slice()
    band_region[rows, columns] = corridor_core

    gradients: dict[str, FloatArray] = {}
    sources: dict[str, dict[str, str]] = {}
    for label, path in cases.items():
        gradients[label] = gradient_on_core(observed_displacement(Path(path)))
        sources[label] = {"path": str(Path(path).resolve()), "sha256": _sha256(Path(path))}

    synthetic = {
        label: gradient_on_core(field)
        for label, field in _synthetic_cases(measured, band_region=band_region).items()
    }

    summary: dict[str, dict[str, float]] = {}
    multiscale: dict[str, dict[int, float]] = {}
    energy: dict[str, dict[int, float]] = {}
    for label, gradient in (gradients | synthetic).items():
        summary[label] = gradient_criteria(gradient, reference)
        multiscale[label] = multiscale_fluctuation(gradient, reference)
        energy[label] = highpass_energy_ratio(gradient, reference)
    summary["dic_self"] = gradient_criteria(reference, reference)
    multiscale["dic_self"] = multiscale_fluctuation(reference, reference)
    energy["dic_self"] = highpass_energy_ratio(reference, reference)

    existing = _existing_metrics(dic_evm_field, evm_fields)

    _write_csv(
        output / "gradient_criteria_summary.csv",
        summary,
        [
            "J_gradient",
            "J_strain",
            "J_norm_map",
            "J_fluctuation",
            "J_fluctuation_symmetric",
            "norm_map_pearson",
            "norm_map_spearman",
            "norm_map_mean_bias",
            "norm_map_quantile_ratio_q90",
            "norm_map_quantile_ratio_q95",
        ],
    )
    _write_csv(
        output / "gradient_criteria_multiscale.csv",
        {
            label: {str(k): v for k, v in curve.items()}
            | {f"energy_ratio_{k}": v for k, v in energy[label].items()}
            for label, curve in multiscale.items()
        },
        [str(scale) for scale in SCALES_PIXELS]
        + [f"energy_ratio_{scale}" for scale in SCALES_PIXELS],
    )

    _figures(
        output,
        reference_gradient=reference,
        gradients=gradients,
        multiscale=multiscale,
        energy=energy,
        summary=summary,
        existing=existing,
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed_gradient_fluctuation_diagnostic",
        "study_type": "exploratory_diagnostic_no_selection",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "mechanics_rerun": False,
        "interpretation_limit": INTERPRETATION_LIMIT,
        "spatial_range_micrometres": SPATIAL_RANGE_MICROMETRES,
        "contract": {
            "partition_id": PARTITION_ID,
            "solve_bounds": list(SOLVE_BOUNDS),
            "core_bounds": list(CORE_BOUNDS),
            "pixel_size_mm": PIXEL_SIZE_MM,
            "differentiation": "np.gradient, array axis 0 = canonical x, then cell_average",
            "edge_handling": "differentiated on the solve grid, then cropped to the core",
            "mask": "declared_all_valid",
            "highpass": f"f - G_s * f, sigma = {GAUSSIAN_SIGMA_FACTOR} * s",
            "scales_pixels": list(SCALES_PIXELS),
        },
        "criteria": summary,
        "multiscale": {label: {str(k): v for k, v in c.items()} for label, c in multiscale.items()},
        "highpass_energy_ratio": {
            label: {str(k): v for k, v in c.items()} for label, c in energy.items()
        },
        "existing_metrics": existing,
        "sources": sources,
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }
    (output / "gradient_criteria_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return report


def _existing_metrics(
    dic_evm: FloatArray,
    evm_fields: dict[str, str | Path],
) -> dict[str, dict[str, float]]:
    """The archived EVM metrics, recomputed here on the same support."""

    threshold = float(np.quantile(dic_evm, 0.90))
    reference_active = dic_evm >= threshold
    result: dict[str, dict[str, float]] = {}
    for label, path in evm_fields.items():
        field = np.asarray(np.load(Path(path), allow_pickle=False), dtype=np.float64)
        difference = field - dic_evm
        denominator = float(np.sqrt(np.sum(dic_evm**2)))
        x = field.ravel() - field.mean()
        y = dic_evm.ravel() - dic_evm.mean()
        active = field >= threshold
        union = int(np.count_nonzero(active | reference_active))
        result[label] = {
            "evm_relative_l2": float(np.sqrt(np.sum(difference**2))) / denominator,
            "evm_pearson": float(np.sum(x * y) / np.sqrt(np.sum(x**2) * np.sum(y**2))),
            "evm_iou_q90": (
                float(np.count_nonzero(active & reference_active) / union)
                if union
                else float("nan")
            ),
        }
    return result


def _write_csv(
    path: Path,
    rows: dict[str, dict[str, float]],
    columns: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", *columns])
        for label in sorted(rows):
            writer.writerow(
                [label] + [f"{rows[label].get(name, float('nan')):.6g}" for name in columns]
            )
