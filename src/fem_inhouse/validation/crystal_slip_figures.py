"""Publication figures for the paired crystal-slip comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _imshow(axis: Any, field: np.ndarray, *, cmap: str, vmin: float, vmax: float) -> Any:
    image = axis.imshow(
        field.T,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    axis.set_xlabel("pixel x")
    axis.set_ylabel("pixel y")
    return image


def _system_label(index: int, labels: list[str]) -> str:
    return labels[index].split("  ", 1)[0]


def generate_comparison_figures(
    meric_equivalent: np.ndarray,
    srix_equivalent: np.ndarray,
    *,
    summary: dict[str, Any],
    labels: list[str],
    output_dir: Path,
) -> list[Path]:
    """Generate the final-field comparison figures and return their paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    meric_total = np.sum(meric_equivalent, axis=0)
    srix_total = np.sum(srix_equivalent, axis=0)
    absolute_max = float(max(np.max(meric_total), np.max(srix_total), np.finfo(float).eps))
    difference = meric_total - srix_total
    difference_bound = float(max(np.max(np.abs(difference)), np.finfo(float).eps))
    meric_norm = meric_total / max(float(np.sum(meric_total)), np.finfo(float).eps)
    srix_norm = srix_total / max(float(np.sum(srix_total)), np.finfo(float).eps)
    normalized_difference = meric_norm - srix_norm
    normalized_bound = float(max(np.max(np.abs(normalized_difference)), np.finfo(float).eps))
    paths: list[Path] = []

    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    image = _imshow(axes[0, 0], meric_total, cmap="viridis", vmin=0.0, vmax=absolute_max)
    axes[0, 0].set_title("Méric — total accumulated slip")
    figure.colorbar(image, ax=axes[0, 0], label="dimensionless slip")
    image = _imshow(axes[0, 1], srix_total, cmap="viridis", vmin=0.0, vmax=absolute_max)
    axes[0, 1].set_title("SRIX — total accumulated slip")
    figure.colorbar(image, ax=axes[0, 1], label="dimensionless slip")
    image = _imshow(
        axes[1, 0], difference, cmap="coolwarm", vmin=-difference_bound, vmax=difference_bound
    )
    axes[1, 0].set_title("Meric - SRIX")
    figure.colorbar(image, ax=axes[1, 0], label="dimensionless slip")
    image = _imshow(
        axes[1, 1],
        normalized_difference,
        cmap="coolwarm",
        vmin=-normalized_bound,
        vmax=normalized_bound,
    )
    axes[1, 1].set_title("Difference after independent normalization")
    figure.colorbar(image, ax=axes[1, 1], label="normalized difference")
    figure.suptitle("P43 100x100 - total accumulated system slip")
    path = output_dir / "total_accumulated_system_slip_absolute_and_normalized.png"
    _save(figure, path)
    paths.append(path)

    fractions = summary["system_distribution"]
    meric_values = np.array([item["meric_fraction"] for item in summary["systems"]])
    srix_values = np.array([item["srix_fraction"] for item in summary["systems"]])
    x = np.arange(12)
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    width = 0.38
    axis.bar(x - width / 2, meric_values, width, label="Méric", color="#4c78a8")
    axis.bar(x + width / 2, srix_values, width, label="SRIX", color="#f58518")
    axis.set_xticks(x, [_system_label(int(index), labels) for index in x])
    axis.set_ylabel("fraction of total accumulated slip")
    axis.set_xlabel("FCC slip system")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.set_title(
        "Per-system contributions — "
        f"S95 Jaccard={fractions['s95_jaccard']:.3f}, "
        f"TV={fractions['total_variation_distance']:.3f}, "
        f"rank rho={fractions['spearman_rank_correlation']!s}"
    )
    path = output_dir / "slip_system_global_fractions.svg"
    _save(figure, path)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for values, name, color in (
        (meric_values, "Méric", "#4c78a8"),
        (srix_values, "SRIX", "#f58518"),
    ):
        order = np.argsort(-values, kind="stable")
        axis.plot(np.arange(1, 13), np.cumsum(values[order]), marker="o", label=name, color=color)
        axis.scatter(np.arange(1, 13), values[order], s=12, color=color, alpha=0.35)
    axis.axhline(0.95, color="black", linestyle="--", linewidth=1, label="95%")
    axis.set_xticks(np.arange(1, 13))
    axis.set_xlabel("number of systems, sorted by contribution")
    axis.set_ylabel("cumulative contribution")
    axis.set_ylim(0.0, 1.04)
    axis.grid(alpha=0.25)
    axis.legend()
    path = output_dir / "slip_system_cumulative_contributions.svg"
    _save(figure, path)
    paths.append(path)

    for item in summary["systems"]:
        index = int(item["system"])
        if item["spatial"]["status"] != "significant":
            continue
        first, second = meric_equivalent[index], srix_equivalent[index]
        first_norm = first / max(float(np.sum(first)), np.finfo(float).eps)
        second_norm = second / max(float(np.sum(second)), np.finfo(float).eps)
        diff = first - second
        diff_norm = first_norm - second_norm
        bound = float(max(np.max(first), np.max(second), np.finfo(float).eps))
        diff_bound = float(max(np.max(np.abs(diff)), np.finfo(float).eps))
        norm_bound = float(max(np.max(np.abs(diff_norm)), np.finfo(float).eps))
        figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        image = _imshow(axes[0, 0], first, cmap="viridis", vmin=0.0, vmax=bound)
        axes[0, 0].set_title("Méric")
        figure.colorbar(image, ax=axes[0, 0])
        image = _imshow(axes[0, 1], second, cmap="viridis", vmin=0.0, vmax=bound)
        axes[0, 1].set_title("SRIX")
        figure.colorbar(image, ax=axes[0, 1])
        image = _imshow(axes[1, 0], diff, cmap="coolwarm", vmin=-diff_bound, vmax=diff_bound)
        axes[1, 0].set_title("absolute difference")
        figure.colorbar(image, ax=axes[1, 0])
        image = _imshow(axes[1, 1], diff_norm, cmap="coolwarm", vmin=-norm_bound, vmax=norm_bound)
        axes[1, 1].set_title("difference after normalization")
        figure.colorbar(image, ax=axes[1, 1])
        figure.suptitle(
            f"System {_system_label(index, labels)} — "
            f"fractions {item['meric_fraction']:.3f}/{item['srix_fraction']:.3f}"
        )
        path = output_dir / f"slip_system_{index + 1:02d}_spatial_comparison.png"
        _save(figure, path)
        paths.append(path)

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for item in summary["systems"]:
        spatial = item["spatial"]
        if spatial["status"] != "significant":
            continue
        normalized = spatial["normalized"]
        similarity = 1.0 - 0.5 * float(normalized["l1"])
        ratio = item["amplitude_ratio_meric_over_srix"]
        if ratio is None:
            continue
        size = 40.0 + 500.0 * 0.5 * (item["meric_fraction"] + item["srix_fraction"])
        axis.scatter(similarity, ratio, s=size, alpha=0.75, label=f"{int(item['system']) + 1:02d}")
        axis.annotate(
            f"{int(item['system']) + 1:02d}",
            (similarity, ratio),
            xytext=(4, 4),
            textcoords="offset points",
        )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("normalized spatial similarity (1 - L1/2)")
    axis.set_ylabel("Méric/SRIX integrated amplitude")
    axis.set_title("Mechanism similarity versus amplitude ratio")
    axis.grid(alpha=0.25)
    path = output_dir / "mechanism_amplitude_summary.svg"
    _save(figure, path)
    paths.append(path)
    return paths
