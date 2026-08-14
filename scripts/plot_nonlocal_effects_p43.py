#!/usr/bin/env python3
"""Plot local/non-local plastic source fields from coupled P43 archives."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _field(path: Path, key: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    with np.load(path) as arrays:
        value = np.asarray(arrays[key], dtype=float)
    if value.ndim == 1 and shape is not None:
        value = value.reshape(shape)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{path}: {key} must be a square 2-D field")
    return value


def _plot_one(local_path: Path, nonlocal_path: Path, label: str, output: Path) -> dict[str, float]:
    local = _field(local_path, "monolithic_peeq")
    coupled_source = _field(nonlocal_path, "monolithic_peeq")
    chi = _field(nonlocal_path, "monolithic_chi", local.shape)

    source_delta = coupled_source - local
    filtered_delta = chi - coupled_source
    scale = max(float(np.max(np.abs(local))), float(np.max(np.abs(coupled_source))), 1e-30)
    delta_scale = max(float(np.max(np.abs(source_delta))), 1e-30)
    centre = local.shape[0] // 2

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = (
        (local, f"{label}: local source q"),
        (coupled_source, f"{label}: q with non-local coupling"),
        (chi, f"{label}: filtered χ"),
        (source_delta, "q(non-local) - q(local)"),
        (filtered_delta, "chi - q(non-local)"),
        (None, "central row profiles"),
    )
    for axis, (values, title) in zip(axes.flat, panels, strict=True):
        if values is None:
            x = np.arange(local.shape[1])
            axis.plot(x, local[centre], label="local q")
            axis.plot(x, coupled_source[centre], label="non-local q")
            axis.plot(x, chi[centre], label="χ")
            axis.set_xlabel("pixel index")
            axis.set_ylabel("source")
            axis.legend(fontsize=8)
            axis.grid(alpha=0.25)
            axis.set_title(title)
            continue
        is_difference = title.startswith(("q(", "chi -"))
        image = axis.imshow(
            values.T,
            origin="lower",
            aspect="equal",
            cmap="coolwarm" if is_difference else "viridis",
            vmin=-delta_scale if is_difference else 0.0,
            vmax=delta_scale if is_difference else scale,
        )
        axis.set_title(title)
        axis.set_xlabel("x pixel")
        axis.set_ylabel("y pixel")
        figure.colorbar(image, ax=axis, shrink=0.85)

    figure.suptitle(f"P43 M100 — effect of scalar non-local coupling — {label}")
    figure.savefig(output, dpi=220)
    plt.close(figure)
    summary = {
        "local_max": float(local.max()),
        "nonlocal_source_max": float(coupled_source.max()),
        "chi_max": float(chi.max()),
        "source_delta_l2_relative": float(
            np.linalg.norm(source_delta) / max(np.linalg.norm(local), 1e-30)
        ),
        "source_delta_abs_max": float(np.max(np.abs(source_delta))),
        "chi_source_l2_relative": float(
            np.linalg.norm(filtered_delta) / max(np.linalg.norm(coupled_source), 1e-30)
        ),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--j2-local", type=Path, required=True)
    parser.add_argument("--j2-nonlocal", type=Path, required=True)
    parser.add_argument("--meric-local", type=Path, required=True)
    parser.add_argument("--meric-nonlocal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "j2": _plot_one(
            args.j2_local,
            args.j2_nonlocal,
            "J2 / p",
            args.output_dir / "p43_m100_j2_nonlocal_effect.png",
        ),
        "meric": _plot_one(
            args.meric_local,
            args.meric_nonlocal,
            "Méric / Γ",
            args.output_dir / "p43_m100_meric_nonlocal_effect.png",
        ),
    }
    import json

    summary["parameters"] = {"length_scale_mm": 0.05888, "coupling_modulus_mpa": 5168.0}
    (args.output_dir / "p43_m100_nonlocal_effects.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
