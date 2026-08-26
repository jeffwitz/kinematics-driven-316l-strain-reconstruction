#!/usr/bin/env python3
"""Plot shared P43 M20 outputs from the NumPy and MFront forwards."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/reference_data/p0043_m20_numpy_srix_forward_f_v1"
NUMPY = OUT / "fields_numpy.npz"
MFRONT = ROOT / "validation/reference_data/p0043_m20_c_f_forward_identified_v1/fields_F.npz"


def main() -> int:
    npy = np.load(NUMPY)
    mfront = np.load(MFRONT)
    labels = np.arange(1, npy["scored_displacement"].shape[0] + 1)
    dic = npy["dic_displacement"]
    numpy_disp = npy["scored_displacement"]
    mfront_disp = mfront["scored_displacement"]
    numpy_evm = npy["evm"]
    mfront_evm = mfront["evm"]
    dic_evm = npy["dic_evm"]

    def disp_rms(value: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean(value**2, axis=(1, 2, 3)))

    def evm_rms(value: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean(value**2, axis=(1, 2)))

    numpy_minus_dic = disp_rms(numpy_disp - dic)
    mfront_minus_dic = disp_rms(mfront_disp - dic)
    numpy_minus_mfront = disp_rms(numpy_disp - mfront_disp)
    metrics = {
        "states": labels.tolist(),
        "numpy_vs_dic_displacement_rms_mm": numpy_minus_dic.tolist(),
        "mfront_vs_dic_displacement_rms_mm": mfront_minus_dic.tolist(),
        "numpy_vs_mfront_displacement_rms_mm": numpy_minus_mfront.tolist(),
        "numpy_vs_mfront_displacement_max_abs_mm": float(np.max(np.abs(numpy_disp - mfront_disp))),
        "numpy_vs_mfront_displacement_relative_l2": float(
            np.linalg.norm(numpy_disp - mfront_disp) / max(np.linalg.norm(mfront_disp), 1e-30)
        ),
        "dic_evm_rms": evm_rms(dic_evm).tolist(),
        "numpy_evm_rms": evm_rms(numpy_evm).tolist(),
        "mfront_evm_rms": evm_rms(mfront_evm).tolist(),
        "numpy_vs_mfront_evm_rms": evm_rms(numpy_evm - mfront_evm).tolist(),
    }
    (OUT / "numpy_mfront_comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    axes[0].plot(labels, numpy_minus_dic * 1e6, "o-", label="NumPy - DIC")
    axes[0].plot(labels, mfront_minus_dic * 1e6, "s--", label="MFront - DIC")
    axes[0].set_ylabel("déplacement RMS (µm)")
    axes[0].set_title("Écart à la DIC")
    axes[1].plot(labels, numpy_minus_mfront * 1e6, "o-")
    axes[1].set_ylabel("NumPy - MFront RMS (µm)")
    axes[1].set_title("Accord des backends")
    axes[2].plot(labels, 100 * evm_rms(dic_evm), "o-", label="DIC")
    axes[2].plot(labels, 100 * evm_rms(numpy_evm), "o-", label="NumPy")
    axes[2].plot(labels, 100 * evm_rms(mfront_evm), "s--", label="MFront")
    axes[2].set_ylabel("EVM RMS (%)")
    axes[2].set_title("EVM")
    for ax in axes:
        ax.set_xlabel("état scoré")
        ax.grid(alpha=0.3)
        ax.set_xticks(labels)
    axes[0].legend()
    axes[2].legend()
    fig.savefig(OUT / "numpy_mfront_comparison_curves.png", dpi=180)
    plt.close(fig)

    state = -1
    panels = [
        (dic_evm[state], "DIC", "viridis"),
        (numpy_evm[state], "NumPy", "viridis"),
        (mfront_evm[state], "MFront", "viridis"),
        (numpy_evm[state] - dic_evm[state], "NumPy - DIC", "coolwarm"),
        (mfront_evm[state] - dic_evm[state], "MFront - DIC", "coolwarm"),
        (numpy_evm[state] - mfront_evm[state], "NumPy - MFront", "coolwarm"),
    ]
    common = max(
        float(np.max(dic_evm[state])),
        float(np.max(numpy_evm[state])),
        float(np.max(mfront_evm[state])),
    )
    diff = max(float(np.max(np.abs(panel[0]))) for panel in panels[3:])
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for index, (data, title, cmap) in enumerate(panels):
        ax = axes.flat[index]
        if index < 3:
            image = ax.imshow(100 * data.T, origin="lower", cmap=cmap, vmin=0, vmax=100 * common)
        else:
            image = ax.imshow(
                100 * data.T, origin="lower", cmap=cmap, vmin=-100 * diff, vmax=100 * diff
            )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, label="%")
    fig.suptitle("P43 M20 Order F — EVM état final, NumPy/MFront/DIC")
    fig.savefig(OUT / "numpy_mfront_evm_final.png", dpi=200)
    plt.close(fig)
    print(json.dumps({"output": str(OUT), "comparison": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
