#!/usr/bin/env python3
"""Plot DIC, prior and constrained rank-7 P43 M20 EVM maps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET, get_parameter_set
from fem_inhouse.identification.srix_parameter_coordinates import SrixTheta9
from fem_inhouse.postprocessing.kinematics import (
    plane_stress_equivalent_strain,
    strain_from_displacement,
)
from scripts.qualify_srix_p0043_synthetic_smoke import (
    CROP,
    _forward,
    _load_inputs,
    _make_path,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "validation/reference_data/p0043_experimental_raw_svd7_provisional_v3/report.json"
DEFAULT_OUTPUT = ROOT / "validation/reference_data/p0043_experimental_raw_svd7_provisional_v3"
PIXEL_SIZE_MM = 0.00184


def _evm(displacement: np.ndarray) -> np.ndarray:
    """Plane-stress total equivalent strain on the displacement nodal grid."""
    strain = strain_from_displacement(
        displacement[..., 0],
        displacement[..., 1],
        spacing_x=PIXEL_SIZE_MM,
        spacing_y=PIXEL_SIZE_MM,
    )
    return plane_stress_equivalent_strain(
        strain.epsilon_xx,
        strain.epsilon_yy,
        strain.gamma_xy,
        poisson_ratio=0.30,
        shear_convention="engineering",
    )


def _plot_maps(
    output: Path,
    evm_dic: np.ndarray,
    evm_prior: np.ndarray,
    evm_final: np.ndarray,
    labels: list[str],
) -> None:
    # One common scale per row makes DIC/prior/final directly comparable.
    rows = len(labels)
    fig, axes = plt.subplots(rows, 5, figsize=(15, 2.25 * rows), squeeze=False)
    for row, label in enumerate(labels):
        values = [evm_dic[row], evm_prior[row], evm_final[row]]
        vmax = max(float(np.nanmax(v)) for v in values)
        vmax = max(vmax, 1.0e-12)
        err_prior = evm_prior[row] - evm_dic[row]
        err_final = evm_final[row] - evm_dic[row]
        err_limit = max(
            float(np.nanmax(np.abs(err_prior))),
            float(np.nanmax(np.abs(err_final))),
            1.0e-12,
        )
        panels = [
            (evm_dic[row], "DIC", "viridis", 0.0, vmax, "%"),
            (evm_prior[row], "Départ", "viridis", 0.0, vmax, "%"),
            (evm_final[row], "Arrivée", "viridis", 0.0, vmax, "%"),
            (err_prior, "Départ − DIC", "coolwarm", -err_limit, err_limit, "%"),
            (err_final, "Arrivée − DIC", "coolwarm", -err_limit, err_limit, "%"),
        ]
        for col, (data, title, cmap, low, high, unit) in enumerate(panels):
            ax = axes[row, col]
            image = ax.imshow(100.0 * data.T, origin="lower", cmap=cmap,
                              vmin=100.0 * low, vmax=100.0 * high, aspect="equal")
            ax.set_title(f"{title}\nétat {label}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label=unit)
    fig.suptitle("P43 M20 — EVM totale équivalente : DIC, départ et arrivée", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "p0043_raw_svd7_evm_maps.png", dpi=180)
    plt.close(fig)


def _plot_summary(output: Path, evm_dic: np.ndarray, evm_prior: np.ndarray, evm_final: np.ndarray,
                  labels: list[str]) -> None:
    rms = lambda x: np.sqrt(np.mean(np.square(x), axis=(1, 2)))
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    ax.plot(x, 100.0 * rms(evm_dic), "o-", label="DIC")
    ax.plot(x, 100.0 * rms(evm_prior), "o-", label="Départ")
    ax.plot(x, 100.0 * rms(evm_final), "o-", label="Arrivée")
    ax.set_xticks(x, labels)
    ax.set_ylabel("RMS EVM (%)")
    ax.set_xlabel("État DIC")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "p0043_raw_svd7_evm_rms_by_state.png", dpi=180)
    plt.close(fig)


def _plot_historical_style(output: Path, evm_dic: np.ndarray, evm_prior: np.ndarray,
                           evm_final: np.ndarray, labels: list[str]) -> None:
    """Write the former 2x3 comparison layout, with one common scale for all states."""
    common_max = max(float(np.nanmax(evm_dic)), float(np.nanmax(evm_prior)),
                     float(np.nanmax(evm_final)))
    diff_prior = evm_prior - evm_dic
    diff_final = evm_final - evm_dic
    diff_change = evm_final - evm_prior
    diff_max = max(float(np.nanmax(np.abs(diff_prior))), float(np.nanmax(np.abs(diff_final))),
                   float(np.nanmax(np.abs(diff_change))), 1.0e-12)
    for row, label in enumerate(labels):
        panels = (
            (evm_dic[row], "DIC EVM", "viridis", 0.0, common_max, "%"),
            (evm_prior[row], "EVM départ", "viridis", 0.0, common_max, "%"),
            (evm_final[row], "EVM arrivée", "viridis", 0.0, common_max, "%"),
            (diff_prior[row], "Départ − DIC", "coolwarm", -diff_max, diff_max, "points de %"),
            (diff_final[row], "Arrivée − DIC", "coolwarm", -diff_max, diff_max, "points de %"),
            (diff_change[row], "Arrivée − départ", "coolwarm", -diff_max, diff_max, "points de %"),
        )
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        for ax, (data, title, cmap, low, high, unit) in zip(axes.flat, panels, strict=True):
            image = ax.imshow(100.0 * data.T, origin="lower", cmap=cmap,
                              vmin=100.0 * low, vmax=100.0 * high, aspect="equal")
            ax.set_title(title)
            ax.set_xlabel("x node index")
            ax.set_ylabel("y node index")
            fig.colorbar(image, ax=ax, label=unit)
        fig.suptitle(f"P43 M20 — comparaison EVM historique — état {label}", fontsize=14)
        fig.savefig(output / f"p0043_raw_svd7_evm_state_{int(label):02d}.png", dpi=220)
        plt.close(fig)
    # A compact overview in the same six-panel convention, one row per state.
    fig, axes = plt.subplots(len(labels), 6, figsize=(18, 2.7 * len(labels)), squeeze=False)
    for row, label in enumerate(labels):
        panels = (evm_dic[row], evm_prior[row], evm_final[row], diff_prior[row], diff_final[row], diff_change[row])
        for col, data in enumerate(panels):
            difference = col >= 3
            ax = axes[row, col]
            image = ax.imshow(100.0 * data.T, origin="lower", aspect="equal",
                              cmap="coolwarm" if difference else "viridis",
                              vmin=-100.0 * diff_max if difference else 0.0,
                              vmax=100.0 * diff_max if difference else 100.0 * common_max)
            if row == 0:
                ax.set_title(("DIC", "Départ", "Arrivée", "Départ−DIC", "Arrivée−DIC", "Arrivée−départ")[col])
            ax.set_ylabel(f"état {label}")
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("P43 M20 — EVM historique, échelle commune", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "p0043_raw_svd7_evm_historical_overview.png", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT.read_text())
    measured_macro, angles, _ = _load_inputs(CROP)
    path = _make_path(measured_macro, 4)
    scored = tuple(4 * index for index in range(1, 9))
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so"))
    prior = SrixTheta9.from_parameter_set(get_parameter_set(DEFAULT_PARAMETER_SET))
    final = SrixTheta9.from_log_coordinates(np.asarray(report["final_eta"], dtype=float))
    fields_prior, timing_prior = _forward(prior, path, angles, library, args.threads)
    fields_final, timing_final = _forward(final, path, angles, library, args.threads)
    dic = np.stack([_evm(path[index - 1].boundary) for index in scored])
    prior_evm = np.stack([_evm(fields_prior[index - 1].displacement) for index in scored])
    final_evm = np.stack([_evm(fields_final[index - 1].displacement) for index in scored])
    labels = [str(index) for index in range(1, 9)]
    _plot_maps(output, dic, prior_evm, final_evm, labels)
    _plot_historical_style(output, dic, prior_evm, final_evm, labels)
    _plot_summary(output, dic, prior_evm, final_evm, labels)
    np.savez_compressed(output / "p0043_raw_svd7_evm_fields.npz", dic=dic, prior=prior_evm, final=final_evm)
    metrics = {
        "states": labels,
        "units": "fraction (plots in percent)",
        "prior_forward_timing": timing_prior,
        "final_forward_timing": timing_final,
        "dic_rms": np.sqrt(np.mean(dic**2, axis=(1, 2))).tolist(),
        "prior_rms": np.sqrt(np.mean(prior_evm**2, axis=(1, 2))).tolist(),
        "final_rms": np.sqrt(np.mean(final_evm**2, axis=(1, 2))).tolist(),
        "prior_minus_dic_rms": np.sqrt(np.mean((prior_evm - dic)**2, axis=(1, 2))).tolist(),
        "final_minus_dic_rms": np.sqrt(np.mean((final_evm - dic)**2, axis=(1, 2))).tolist(),
    }
    (output / "p0043_raw_svd7_evm_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"output": str(output), "prior_forward_s": timing_prior["seconds"],
                      "final_forward_s": timing_final["seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
