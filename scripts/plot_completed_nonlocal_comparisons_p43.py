#!/usr/bin/env python3
"""Plot physically comparable fields from completed P43 non-local runs.

The DIC panel contains an equivalent *total* strain reconstructed from the
measured displacement.  It is deliberately kept separate from J2 PEEQ, which
is a history variable and is not observable from one DIC image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from fem_inhouse.postprocessing.kinematics import (
    plane_stress_equivalent_strain,
    strain_from_displacement,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_ebi import unpack_interior

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/case_study"


def _archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as arrays:
        return {name: np.asarray(arrays[name]).copy() for name in arrays.files}


def _report(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile_limits(
    arrays: tuple[np.ndarray, ...], low: float = 5.0, high: float = 95.0
) -> tuple[float, float]:
    values = np.concatenate([np.asarray(array, dtype=float).ravel() for array in arrays])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cannot scale an empty field")
    vmin, vmax = np.percentile(finite, [low, high])
    if vmax <= vmin:
        vmax = vmin + max(abs(vmin), 1.0) * 1.0e-12
    return float(vmin), float(vmax)


def _difference_limit(array: np.ndarray) -> float:
    finite = np.abs(np.asarray(array, dtype=float)[np.isfinite(array)])
    return max(float(np.percentile(finite, 95.0)), np.finfo(float).tiny)


def _equivalent_strain(displacement: np.ndarray, spacing: float) -> np.ndarray:
    strain = strain_from_displacement(
        displacement[..., 0],
        displacement[..., 1],
        spacing_x=spacing,
        spacing_y=spacing,
    )
    return plane_stress_equivalent_strain(
        strain.epsilon_xx,
        strain.epsilon_yy,
        strain.gamma_xy,
        poisson_ratio=0.30,
        shear_convention="engineering",
    )


def _dic_displacement(crop: tuple[int, int, int, int]) -> np.ndarray:
    x0, x1, y0, y1 = crop
    ux = np.load(DATA / "displacement_x_mm.npy", mmap_mode="r")
    uy = np.load(DATA / "displacement_y_mm.npy", mmap_mode="r")
    return np.stack(
        (ux[x0 : x1 + 1, y0 : y1 + 1], uy[x0 : x1 + 1, y0 : y1 + 1]),
        axis=-1,
    )


def _full_displacement(mechanical: np.ndarray, boundary: np.ndarray, spacing: float) -> np.ndarray:
    nx, ny = boundary.shape[0] - 1, boundary.shape[1] - 1
    grid = StructuredGrid2D(nx, ny, nx * spacing, ny * spacing)
    full = boundary.copy()
    correction = unpack_interior(np.asarray(mechanical, dtype=float), grid)
    full[1:-1, 1:-1] += correction[1:-1, 1:-1]
    return full


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(candidate) - np.asarray(reference))
        / max(np.linalg.norm(reference), np.finfo(float).tiny)
    )


def _top_jaccard(left: np.ndarray, right: np.ndarray, percentile: float = 95.0) -> float:
    left_mask = left >= np.percentile(left, percentile)
    right_mask = right >= np.percentile(right, percentile)
    union = np.count_nonzero(left_mask | right_mask)
    return float(np.count_nonzero(left_mask & right_mask) / max(union, 1))


def _metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    return {
        "relative_l2": _relative_l2(right, left),
        "spearman": float(spearmanr(left.ravel(), right.ravel()).statistic),
        "top_5_percent_jaccard": _top_jaccard(left, right),
        "left_min": float(np.nanmin(left)),
        "left_max": float(np.nanmax(left)),
        "right_min": float(np.nanmin(right)),
        "right_max": float(np.nanmax(right)),
    }


def _image(
    axis: plt.Axes,
    values: np.ndarray,
    title: str,
    *,
    limits: tuple[float, float],
    difference: bool = False,
    label: str = "",
) -> None:
    image = axis.imshow(
        values.T,
        origin="lower",
        aspect="equal",
        cmap="coolwarm" if difference else "viridis",
        vmin=limits[0],
        vmax=limits[1],
    )
    axis.set_title(title)
    axis.set_xlabel("x pixel")
    axis.set_ylabel("y pixel")
    axis.figure.colorbar(image, ax=axis, shrink=0.82, label=label)


def _plot_srix(
    local: dict[str, np.ndarray], nonlocal_: dict[str, np.ndarray], output: Path
) -> dict[str, object]:
    gamma_local = np.asarray(local["monolithic_source"], dtype=float)
    gamma_nonlocal = np.asarray(nonlocal_["monolithic_source"], dtype=float)
    chi = np.asarray(nonlocal_["monolithic_chi"], dtype=float).reshape(gamma_local.shape)
    delta = gamma_nonlocal - gamma_local
    gamma_limits = _percentile_limits((gamma_local, gamma_nonlocal))
    chi_limits = _percentile_limits((chi,))
    delta_limit = _difference_limit(delta)

    figure, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    _image(axes[0, 0], gamma_local, "SRIX local: Γ", limits=gamma_limits, label="Γ")
    _image(
        axes[0, 1],
        gamma_nonlocal,
        "SRIX non-local: Γ",
        limits=gamma_limits,
        label="Γ",
    )
    _image(axes[1, 0], chi, "SRIX non-local: χ", limits=chi_limits, label="χ")
    _image(
        axes[1, 1],
        delta,
        "Γ(non-local) - Γ(local)",
        limits=(-delta_limit, delta_limit),
        difference=True,
        label="ΔΓ",
    )
    figure.suptitle(
        "P43 M100 EBSD — effet du couplage non local SRIX\néchelles tronquées aux percentiles 5-95"
    )
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return {
        "gamma_local_vs_nonlocal": _metrics(gamma_local, gamma_nonlocal),
        "chi_vs_nonlocal_gamma": _metrics(gamma_nonlocal, chi),
        "figure": str(output),
    }


def _plot_j2_dic(
    local: dict[str, np.ndarray],
    nonlocal_: dict[str, np.ndarray],
    report: dict[str, object],
    output: Path,
) -> dict[str, object]:
    crop = tuple(int(value) for value in report["crop_nodes"])
    spacing = float(report["pixel_size_mm"])
    boundary = _dic_displacement(crop)
    dic_evm = _equivalent_strain(boundary, spacing)
    local_evm = _equivalent_strain(
        _full_displacement(local["monolithic_mechanical"], boundary, spacing), spacing
    )
    nonlocal_evm = _equivalent_strain(
        _full_displacement(nonlocal_["monolithic_mechanical"], boundary, spacing), spacing
    )
    local_peeq = np.asarray(local["monolithic_peeq"], dtype=float)
    nonlocal_peeq = np.asarray(nonlocal_["monolithic_peeq"], dtype=float)
    peeq_delta = nonlocal_peeq - local_peeq
    evm_limits = _percentile_limits((dic_evm, local_evm, nonlocal_evm))
    peeq_limits = _percentile_limits((local_peeq, nonlocal_peeq))
    peeq_delta_limit = _difference_limit(peeq_delta)

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for axis, values, title in zip(
        axes[0],
        (dic_evm, local_evm, nonlocal_evm),
        (
            "DIC: déformation totale équivalente",
            "J2 local: déformation totale équivalente",
            "J2 non-local: déformation totale équivalente",
        ),
        strict=True,
    ):
        _image(axis, values, title, limits=evm_limits, label="EVM")
    _image(axes[1, 0], local_peeq, "J2 local: PEEQ", limits=peeq_limits, label="PEEQ")
    _image(axes[1, 1], nonlocal_peeq, "J2 non-local: PEEQ", limits=peeq_limits, label="PEEQ")
    _image(
        axes[1, 2],
        peeq_delta,
        "PEEQ(non-local) - PEEQ(local)",
        limits=(-peeq_delta_limit, peeq_delta_limit),
        difference=True,
        label="ΔPEEQ",
    )
    figure.suptitle(
        "P43 M100 — DIC vs J2 local/non-local\n"
        "DIC comparé à la déformation totale; PEEQ séparée — percentiles 5-95"
    )
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return {
        "total_strain_dic_vs_local": _metrics(dic_evm, local_evm),
        "total_strain_dic_vs_nonlocal": _metrics(dic_evm, nonlocal_evm),
        "peeq_local_vs_nonlocal": _metrics(local_peeq, nonlocal_peeq),
        "figure": str(output),
        "physical_note": (
            "DIC equivalent total strain is not PEEQ; the two quantities are shown "
            "on separate rows and are not subtracted from each other."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--srix-local", type=Path, required=True)
    parser.add_argument("--srix-nonlocal", type=Path, required=True)
    parser.add_argument("--j2-local", type=Path, required=True)
    parser.add_argument("--j2-nonlocal", type=Path, required=True)
    parser.add_argument("--j2-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "srix": _plot_srix(
            _archive(arguments.srix_local),
            _archive(arguments.srix_nonlocal),
            arguments.output_dir / "p43_m100_srix_gamma_local_vs_nonlocal.png",
        ),
        "j2": _plot_j2_dic(
            _archive(arguments.j2_local),
            _archive(arguments.j2_nonlocal),
            _report(arguments.j2_report),
            arguments.output_dir / "p43_m100_j2_local_nonlocal_dic.png",
        ),
        "slip_systems": {
            "status": "not_archived_in_completed_coupled_runs",
            "reason": (
                "The completed coupled SRIX archives contain Gamma but not the twelve "
                "per-system signed and accumulated slips; they cannot be reconstructed "
                "from Gamma alone."
            ),
        },
        "display": {"lower_percentile": 5.0, "upper_percentile": 95.0},
    }
    summary_path = arguments.output_dir / "p43_m100_completed_nonlocal_figures.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
