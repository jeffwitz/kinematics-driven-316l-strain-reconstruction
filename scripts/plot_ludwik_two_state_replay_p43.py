#!/usr/bin/env python3
"""Maps for the Ludwik replay: what the model reproduces, and where it fails.

A single ratio says how much of the elastic defect Ludwik closes; it cannot say
whether the remainder is spread over the crop or concentrated in a few bands.
That distinction decides the next inverse: a diffuse remainder points at the
hardening amplitude, a banded one at the flow direction and at crystallography.

Every panel is built from the `.npz` written by
`replay_ludwik_two_state_history_p43.py`, so the fields have already been
through the one shared observation operator. The two triangular subcells of a
pixel are averaged for display only -- all reported norms come from the report,
which uses both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

COMPONENTS = ("xx", "yy", "xy")


def _pixel_average(field: FloatArray) -> FloatArray:
    """Average the two subcells: `(nx, ny, 2, ...)` to `(nx, ny, ...)`."""

    return np.asarray(field).mean(axis=2)


def _show(axis: plt.Axes, values: FloatArray, title: str, **kwargs: object) -> object:
    image = axis.imshow(np.asarray(values).T, origin="lower", **kwargs)  # type: ignore[arg-type]
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    return image


def _equivalent_maps(data: dict[str, FloatArray], state: int, output: Path) -> None:
    dic = _pixel_average(data[f"dic_equivalent_{state}"])
    ludwik = _pixel_average(data[f"ludwik_equivalent_{state}"])
    elastic = _pixel_average(data[f"elastic_equivalent_{state}"])
    ceiling = float(np.percentile(dic, 99.5))
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.9), constrained_layout=True)
    for axis, values, title in zip(
        axes,
        (dic, ludwik, elastic),
        ("DIC", "Ludwik/J2", "elastic"),
        strict=True,
    ):
        image = _show(axis, values, title, vmin=0.0, vmax=ceiling, cmap="inferno")
    figure.colorbar(image, ax=axes, shrink=0.85, label="equivalent strain (incompressible)")
    figure.suptitle(
        f"State {state}: equivalent strain through the same observation operator",
        fontsize=10,
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _error_maps(data: dict[str, FloatArray], state: int, output: Path) -> None:
    dic = _pixel_average(data[f"dic_strain_{state}"])
    ludwik = _pixel_average(data[f"ludwik_strain_{state}"])
    elastic = _pixel_average(data[f"elastic_strain_{state}"])
    ludwik_error = np.linalg.norm(ludwik - dic, axis=-1)
    elastic_error = np.linalg.norm(elastic - dic, axis=-1)
    ceiling = float(np.percentile(elastic_error, 99.5))
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), constrained_layout=True)
    image = _show(
        axes[0], elastic_error, "elastic defect", vmin=0.0, vmax=ceiling, cmap="viridis"
    )
    _show(axes[1], ludwik_error, "Ludwik defect", vmin=0.0, vmax=ceiling, cmap="viridis")
    figure.colorbar(image, ax=axes[:2], shrink=0.85, label="Kelvin strain error")
    # Where Ludwik helps and where it hurts, on a symmetric scale so the sign
    # is legible: negative means the model moved the field away from the
    # measurement, which a single global ratio cannot show.
    improvement = elastic_error - ludwik_error
    span = float(np.percentile(np.abs(improvement), 99.5))
    gain = _show(
        axes[2],
        improvement,
        "elastic defect - Ludwik defect",
        vmin=-span,
        vmax=span,
        cmap="RdBu_r",
    )
    figure.colorbar(gain, ax=axes[2], shrink=0.85)
    figure.suptitle(f"State {state}: where the Ludwik correction acts", fontsize=10)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _component_maps(data: dict[str, FloatArray], state: int, output: Path) -> None:
    dic = _pixel_average(data[f"dic_strain_{state}"])
    ludwik = _pixel_average(data[f"ludwik_strain_{state}"])
    residual = dic - ludwik
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), constrained_layout=True)
    for index, (axis, name) in enumerate(zip(axes, COMPONENTS, strict=True)):
        values = residual[..., index]
        span = float(np.percentile(np.abs(values), 99.5))
        image = _show(
            axis,
            values,
            f"r_L {name}",
            vmin=-span,
            vmax=span,
            cmap="RdBu_r",
        )
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle(
        f"State {state}: Ludwik residual r_L = eps_DIC - eps_Ludwik, Kelvin components",
        fontsize=10,
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _summary(report: dict, states: list[int], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
    absolute = [report["per_state"][str(state)]["absolute"]["E_L"] for state in states]
    axes[0].plot(states, absolute, "o-", label="total tensor")
    for name in COMPONENTS:
        axes[0].plot(
            states,
            [
                report["per_state"][str(state)]["absolute"]["per_component"][name]["E_L"]
                for state in states
            ],
            "s--",
            alpha=0.7,
            label=name,
        )
    axes[0].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[0].set_xlabel("state")
    axes[0].set_ylabel("E_L")
    axes[0].set_title("Ludwik defect relative to the elastic defect", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].plot(
        states,
        [
            report["per_state"][str(state)]["absolute"]["elastic_relative_error"]
            for state in states
        ],
        "o-",
        label="elastic",
    )
    axes[1].plot(
        states,
        [
            report["per_state"][str(state)]["absolute"]["ludwik_relative_error"]
            for state in states
        ],
        "s-",
        label="Ludwik",
    )
    axes[1].set_xlabel("state")
    axes[1].set_ylabel("strain error relative to the DIC norm")
    axes[1].set_title("Absolute agreement with the measurement", fontsize=10)
    axes[1].legend(fontsize=8)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()

    report = json.loads(arguments.report.read_text(encoding="utf-8"))
    data = dict(np.load(report["field_file"]))
    states = [int(state) for state in report["states"]]
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    for state in states:
        _equivalent_maps(data, state, arguments.output_directory / f"equivalent_{state}.png")
        _error_maps(data, state, arguments.output_directory / f"defect_{state}.png")
        _component_maps(data, state, arguments.output_directory / f"residual_{state}.png")
    _summary(report, states, arguments.output_directory / "summary.png")
    print(f"wrote {1 + 3 * len(states)} figures to {arguments.output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
