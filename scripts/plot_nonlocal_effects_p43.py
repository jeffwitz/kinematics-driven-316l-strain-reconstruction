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


def _limits(*arrays: np.ndarray) -> tuple[float, float]:
    """Return readable 5--95% limits, ignoring the inactive zero background."""
    values = np.concatenate([np.abs(np.asarray(array, dtype=float).ravel()) for array in arrays])
    finite = values[np.isfinite(values)]
    positive = finite[finite > max(float(np.max(finite, initial=0.0)) * 1.0e-12, 1.0e-30)]
    if positive.size == 0:
        positive = finite
    low, high = np.percentile(positive, [5, 95])
    return float(low), float(max(high, low + 1.0e-30))


def _plot_one(
    local_path: Path, nonlocal_path: Path, label: str, output: Path, source_name: str
) -> dict[str, float]:
    local = _field(local_path, "monolithic_peeq")
    coupled_source = _field(nonlocal_path, "monolithic_peeq")
    chi = _field(nonlocal_path, "monolithic_chi", local.shape)

    source_delta = coupled_source - local
    filtered_delta = chi - coupled_source
    source_limits = _limits(local, coupled_source)
    chi_limits = _limits(chi)
    source_delta_limits = _limits(source_delta)
    filtered_delta_limits = _limits(filtered_delta)
    centre = local.shape[0] // 2

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = (
        (local, f"{label}: local source {source_name}"),
        (coupled_source, f"{label}: {source_name} with non-local coupling"),
        (chi, f"{label}: filtered χ"),
        (source_delta, f"{source_name}(non-local) - {source_name}(local)"),
        (filtered_delta, f"chi - {source_name}(non-local)"),
        (None, "central row profiles"),
    )
    for axis, (values, title) in zip(axes.flat, panels, strict=True):
        if values is None:
            x = np.arange(local.shape[1])
            axis.plot(x, local[centre], label=f"local {source_name}")
            axis.plot(x, coupled_source[centre], label=f"non-local {source_name}")
            axis.plot(x, chi[centre], label="χ")
            axis.set_xlabel("pixel index")
            axis.set_ylabel("source")
            axis.legend(fontsize=8)
            axis.grid(alpha=0.25)
            axis.set_title(title)
            continue
        is_difference = title.startswith((f"{source_name}(", "chi -"))
        if title == f"{label}: filtered χ":
            limits = chi_limits
        elif title.startswith(f"{source_name}(non-local)"):
            limits = source_delta_limits
        elif title.startswith("chi -"):
            limits = filtered_delta_limits
        else:
            limits = source_limits
        image = axis.imshow(
            values.T,
            origin="lower",
            aspect="equal",
            cmap="coolwarm" if is_difference else "viridis",
            vmin=-limits[1] if is_difference else limits[0],
            vmax=limits[1],
        )
        axis.set_title(title)
        axis.set_xlabel("x pixel")
        axis.set_ylabel("y pixel")
        figure.colorbar(image, ax=axis, shrink=0.85)

    figure.suptitle(
        f"P43 M100 — final DIC increment — scalar non-local coupling — {label}"
    )
    figure.savefig(output, dpi=220)
    plt.close(figure)
    summary = {
        "local_max": float(local.max()),
        "nonlocal_source_max": float(coupled_source.max()),
        "chi_max": float(chi.max()),
        "source_percentile_5": source_limits[0],
        "source_percentile_95": source_limits[1],
        "chi_percentile_5": chi_limits[0],
        "chi_percentile_95": chi_limits[1],
        "source_name": source_name,
        "source_active_fraction_local": float(np.count_nonzero(local > 1.0e-12) / local.size),
        "source_active_fraction_nonlocal": float(
            np.count_nonzero(coupled_source > 1.0e-12) / coupled_source.size
        ),
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
    parser.add_argument("--srix-local", type=Path, required=True)
    parser.add_argument("--srix-nonlocal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "j2": _plot_one(
            args.j2_local,
            args.j2_nonlocal,
            "J2 / p",
            args.output_dir / "p43_m100_j2_nonlocal_effect.png",
            "p",
        ),
        "meric": _plot_one(
            args.meric_local,
            args.meric_nonlocal,
            "Méric / Γ",
            args.output_dir / "p43_m100_meric_nonlocal_effect.png",
            "Gamma",
        ),
        "srix": _plot_one(
            args.srix_local,
            args.srix_nonlocal,
            "SRIX / Gamma",
            args.output_dir / "p43_m100_srix_nonlocal_effect.png",
            "Gamma",
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
