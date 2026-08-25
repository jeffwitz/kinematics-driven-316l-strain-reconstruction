#!/usr/bin/env python3
"""Compare classical M20 forwards with C- and F-order EBSD mappings."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from fem_inhouse.core.nonlinear import run_fem
from fem_inhouse.identification.srix_parameter_coordinates import SrixTheta9
from fem_inhouse.postprocessing.kinematics import (
    plane_stress_equivalent_strain,
    strain_from_displacement,
)
from scripts.qualify_srix_p0043_synthetic_smoke import CROP, _load_inputs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "validation/reference_data/p0043_ebsd_mapping_audit_v1"
REPORT = ROOT / "validation/reference_data/p0043_experimental_raw_svd7_provisional_v3/report.json"
PIXEL_SIZE_MM = 0.00184


def _evm(displacement: np.ndarray) -> np.ndarray:
    strain = strain_from_displacement(
        displacement[..., 0], displacement[..., 1],
        spacing_x=PIXEL_SIZE_MM, spacing_y=PIXEL_SIZE_MM,
    )
    return plane_stress_equivalent_strain(
        strain.epsilon_xx, strain.epsilon_yy, strain.gamma_xy,
        poisson_ratio=0.30, shear_convention="engineering",
    )


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = candidate - reference
    return {
        "rms": float(np.sqrt(np.mean(np.square(delta)))),
        "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(reference)),
        "pearson": float(np.corrcoef(candidate.ravel(), reference.ravel())[0, 1]),
        "spearman": float(spearmanr(candidate.ravel(), reference.ravel()).statistic),
    }


def _plot(output: Path, dic: np.ndarray, c: np.ndarray, f: np.ndarray) -> None:
    c_minus = c - dic
    f_minus = f - dic
    f_c = f - c
    fields = (dic, c, f, c_minus, f_minus, f_c)
    titles = ("DIC EVM", "Mapping C", "Mapping F", "C − DIC", "F − DIC", "F − C")
    common_max = max(float(np.max(x)) for x in fields[:3])
    diff_max = max(float(np.max(np.abs(x))) for x in fields[3:])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, values, title in zip(axes.flat, fields, titles, strict=True):
        difference = title in {"C − DIC", "F − DIC", "F − C"}
        image = ax.imshow(
            100.0 * values.T, origin="lower", aspect="equal",
            cmap="coolwarm" if difference else "viridis",
            vmin=-100.0 * diff_max if difference else 0.0,
            vmax=100.0 * diff_max if difference else 100.0 * common_max,
        )
        ax.set_title(title)
        ax.set_xlabel("x node index")
        ax.set_ylabel("y node index")
        fig.colorbar(image, ax=ax, label="points de %" if difference else "%")
    fig.suptitle("P43 M20 — test de mapping EBSD C/F, paramètres optimisés", fontsize=14)
    fig.savefig(output / "old_vs_corrected_evm.png", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT.read_text())
    theta = SrixTheta9.from_log_coordinates(report["final_eta"])
    macro, angles, _ = _load_inputs(CROP)
    boundary = macro[-1]
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so"))
    base = dict(
        disp_x=boundary[..., 0], disp_y=boundary[..., 1],
        yield_map=np.ones((20, 20)), K_map=np.ones((20, 20)), n_exp=0.245,
        x_size=20 * PIXEL_SIZE_MM, y_size=20 * PIXEL_SIZE_MM,
        element_size=PIXEL_SIZE_MM, scale_factor=1.0,
        E_mod=205000.0, nu=0.30, N_inc=32,
        constitutive_backend="mfront", mfront_behaviour_id="fcc_forest_rubin_srix",
        mfront_library=library, mfront_threads=args.threads, verbose=False,
    )
    fields = {}
    timings = {}
    for order in ("C", "F"):
        started = time.perf_counter()
        fields[order] = run_fem(
            **base,
            constitutive_options={
                "parameters": theta.as_runtime_overrides(),
                "crystal_orientation": {
                    "mode": "ebsd", "euler_bunge_deg": angles, "element_order": order,
                },
            },
        )
        timings[order] = time.perf_counter() - started
    dic = _evm(boundary)
    evm_c = _evm(fields["C"]["U"])
    evm_f = _evm(fields["F"]["U"])
    _plot(output, dic, evm_c, evm_f)
    np.savez_compressed(output / "old_vs_corrected_fields.npz", dic=dic, mapping_c=evm_c, mapping_f=evm_f,
                        displacement_c=fields["C"]["U"], displacement_f=fields["F"]["U"])
    metrics = {"C_vs_DIC": _metrics(evm_c, dic), "F_vs_DIC": _metrics(evm_f, dic),
               "F_vs_C": _metrics(evm_f, evm_c), "timings_seconds": timings,
               "parameters": theta.as_runtime_overrides(), "boundary_state": "final DIC state",
               "element_orders": {"historical": "C", "corrected": "F"},
               "claims": {"mechanical_mapping_effect_measured": True,
                          "dic_ebsd_axes_proven": False,
                          "sample_frame_proven": False}}
    (output / "localization_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
