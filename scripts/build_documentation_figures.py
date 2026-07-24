#!/usr/bin/env python3
"""Generate the vector figures used by the Sphinx documentation."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "docs" / "_static"
NAVY = "#17324d"
BLUE = "#2980b9"
LIGHT_BLUE = "#d9edf7"
GREEN = "#2e8b57"
LIGHT_GREEN = "#dff0d8"
ORANGE = "#d97706"
LIGHT_ORANGE = "#fff0d4"
RED = "#b03a2e"
LIGHT_GREY = "#f5f7f9"
DARK_GREY = "#404040"


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "svg.fonttype": "none",
            "svg.hashsalt": "kinematics-driven-316l-docs",
        }
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


def _save_vector_pair(figure: Figure, stem: str) -> None:
    figure.savefig(
        OUTPUT_DIRECTORY / f"{stem}.svg",
        format="svg",
        bbox_inches="tight",
        transparent=True,
        metadata={"Date": None},
    )
    figure.savefig(
        OUTPUT_DIRECTORY / f"{stem}.pdf",
        format="pdf",
        bbox_inches="tight",
        transparent=True,
        metadata={"CreationDate": None, "ModDate": None},
    )


def _box(
    axis: plt.Axes,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    *,
    face_color: str,
    edge_color: str,
) -> None:
    rectangle = FancyBboxPatch(
        (center_x - width / 2, center_y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        linewidth=1.8,
        facecolor=face_color,
        edgecolor=edge_color,
    )
    axis.add_patch(rectangle)
    axis.text(
        center_x,
        center_y + 0.18 * height,
        title,
        ha="center",
        va="center",
        color=NAVY,
        fontweight="bold",
        fontsize=10.5,
    )
    axis.text(
        center_x,
        center_y - 0.18 * height,
        detail,
        ha="center",
        va="center",
        color=DARK_GREY,
        fontsize=8.5,
        linespacing=1.25,
    )


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.6,
            color=BLUE,
            connectionstyle="arc3,rad=0",
        )
    )


def workflow_figure() -> None:
    figure, axis = plt.subplots(figsize=(15, 4.3))
    axis.set_xlim(0, 15)
    axis.set_ylim(0, 4.3)
    axis.axis("off")

    centers = [1.2, 3.7, 6.2, 8.7, 11.2, 13.7]
    boxes = (
        (
            "Test + DIC",
            "Measured displacement\nat 1.84 µm/pixel",
            LIGHT_BLUE,
            BLUE,
        ),
        (
            "Preparation",
            "V → uₓ, U → uᵧ\npixels → millimetres",
            LIGHT_BLUE,
            BLUE,
        ),
        (
            "Local fields",
            "σᵧ(x,y), K(x,y)\neffective descriptors",
            LIGHT_ORANGE,
            ORANGE,
        ),
        (
            "CPS4 mesh",
            "1 pixel = 1 element\nDIC-driven boundaries",
            LIGHT_GREEN,
            GREEN,
        ),
        (
            "MFront + Newton",
            "Analytical J2/Ludwik\nglobal equilibrium",
            LIGHT_GREEN,
            GREEN,
        ),
        (
            "Reconstruction",
            "U, S, E, PE,\nPEEQ and RF",
            "#f3e5f5",
            "#7d3c98",
        ),
    )
    for center, (title, detail, face, edge) in zip(centers, boxes, strict=True):
        _box(
            axis,
            center,
            2.35,
            2.05,
            1.35,
            title,
            detail,
            face_color=face,
            edge_color=edge,
        )
    for left, right in pairwise(centers):
        _arrow(axis, (left + 1.05, 2.35), (right - 1.05, 2.35))

    axis.text(
        7.5,
        3.75,
        "From measured kinematics to a mechanically admissible field",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=NAVY,
    )
    axis.text(
        7.5,
        0.55,
        "DIC provides the observation; the FE solve enforces equilibrium. "
        "Local parameters reproduce spatial organization and are not "
        "intrinsic grain properties.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=DARK_GREY,
    )
    _save_vector_pair(figure, "workflow")
    plt.close(figure)


def hardening_figure() -> None:
    yield_stress = 124.0
    coefficient = 380.0
    exponent = 0.245
    first_positive_strain = 1e-6
    legacy_cap = 0.2
    plastic_strain = np.linspace(0.0, 0.35, 1200)

    regularized = np.where(
        plastic_strain <= first_positive_strain,
        plastic_strain * first_positive_strain ** (exponent - 1.0),
        plastic_strain**exponent,
    )
    mfront_stress = yield_stress + coefficient * regularized
    legacy_stress = yield_stress + coefficient * np.minimum(plastic_strain, legacy_cap) ** exponent

    figure, axis = plt.subplots(figsize=(8.6, 4.8))
    axis.plot(
        plastic_strain,
        mfront_stress,
        color=GREEN,
        linewidth=2.8,
        label="Analytical MFront, no upper cap",
    )
    axis.plot(
        plastic_strain,
        legacy_stress,
        color=RED,
        linewidth=2.1,
        linestyle="--",
        label="Historical table, plateau after PEEQ = 0.2",
    )
    axis.axvline(legacy_cap, color=RED, linewidth=1, alpha=0.6)
    axis.annotate(
        "Historical cap",
        xy=(legacy_cap, yield_stress + coefficient * legacy_cap**exponent),
        xytext=(0.235, 330),
        arrowprops={"arrowstyle": "->", "color": RED},
        color=RED,
    )
    axis.set(
        xlabel="Equivalent plastic strain PEEQ",
        ylabel="Yield-surface radius (MPa)",
        title="Ludwik–Hollomon hardening used by the repository",
        xlim=(0.0, 0.35),
    )
    axis.grid(True, color="#d5d8dc", linewidth=0.8)
    axis.legend(loc="lower right", frameon=True)
    axis.text(
        0.01,
        0.98,
        "σ = σᵧ + K·PEEQⁿ\nσᵧ = 124 MPa, K = 380 MPa, n = 0.245",
        transform=axis.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )
    _save_vector_pair(figure, "ludwik_hardening")
    plt.close(figure)


def partitioning_figure() -> None:
    figure, (global_axis, detail_axis) = plt.subplots(1, 2, figsize=(11.5, 5.0))

    global_axis.add_patch(Rectangle((0, 0), 10, 10, facecolor="white", edgecolor=NAVY, linewidth=2))
    for index in range(1, 10):
        global_axis.axvline(index, color="#aab7b8", linewidth=0.7)
        global_axis.axhline(index, color="#aab7b8", linewidth=0.7)
    global_axis.add_patch(
        Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor=NAVY, linewidth=1.5, alpha=0.85)
    )
    global_axis.text(0.5, 0.5, "p0", color="white", ha="center", va="center", fontweight="bold")
    global_axis.set(
        xlim=(-0.2, 10.2),
        ylim=(-0.2, 10.2),
        aspect="equal",
        title="Complete ROI: 10 × 10 cores",
        xlabel="x: 3600 elements",
        ylabel="y: 3100 elements",
    )
    global_axis.set_xticks([])
    global_axis.set_yticks([])

    detail_axis.add_patch(
        Rectangle(
            (0, 0),
            510,
            460,
            facecolor=LIGHT_BLUE,
            edgecolor=BLUE,
            linewidth=2.2,
            label="Solved region",
        )
    )
    detail_axis.add_patch(
        Rectangle(
            (0, 0),
            360,
            310,
            facecolor=LIGHT_GREEN,
            edgecolor=GREEN,
            linewidth=2.2,
            label="Retained core",
        )
    )
    detail_axis.annotate(
        "150-element padding",
        xy=(435, 310),
        xytext=(435, 390),
        ha="center",
        arrowprops={"arrowstyle": "<->", "color": ORANGE},
        color=ORANGE,
        fontweight="bold",
    )
    detail_axis.annotate(
        "150-element padding",
        xy=(360, 385),
        xytext=(450, 385),
        va="center",
        arrowprops={"arrowstyle": "<->", "color": ORANGE},
        color=ORANGE,
        fontweight="bold",
    )
    detail_axis.text(180, 155, "360 × 310\nelements", ha="center", va="center", color=NAVY)
    detail_axis.text(
        255,
        445,
        "510 × 460 solved elements",
        ha="center",
        va="center",
        color=NAVY,
        fontweight="bold",
    )
    detail_axis.set(
        xlim=(-15, 525),
        ylim=(-15, 475),
        aspect="equal",
        title="Corner partition p0",
        xlabel="axis 0 = x",
        ylabel="axis 1 = y",
    )
    detail_axis.legend(loc="upper right")
    detail_axis.grid(False)

    figure.suptitle(
        "Overlap used to reduce interface artefacts",
        fontsize=15,
        color=NAVY,
        fontweight="bold",
    )
    _save_vector_pair(figure, "partitioning")
    plt.close(figure)


def main() -> None:
    _configure()
    workflow_figure()
    hardening_figure()
    partitioning_figure()


if __name__ == "__main__":
    main()
