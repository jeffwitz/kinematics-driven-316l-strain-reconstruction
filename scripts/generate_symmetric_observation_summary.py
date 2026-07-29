"""Generate consolidated P43 V3 figures from immutable replay artefacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

CASES = (
    (0, "local", "results/constitutive-local-p0043-pad150"),
    (1, "a100", "results/constitutive-nonlocal-p0043-pad150-a100"),
    (2, "a200", "results/constitutive-nonlocal-p0043-pad150-a200"),
    (4, "a400", "results/constitutive-nonlocal-p0043-pad150-a400"),
)
PROFILES = ("legacy_script_2021", "declared_medium_v4")


def _report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_row(
    *,
    alpha: int,
    case: str,
    profile: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    raw = report["metrics"]["raw"]
    observed = report["metrics"]["observed"]
    return {
        "alpha": alpha,
        "case": case,
        "profile": profile,
        "status": report["status"],
        "evm_post_filter_applied": report["evm_post_filter_applied"],
        "raw": {
            "relative_l2": raw["errors"]["relative_l2_error"],
            "pearson": raw["errors"]["pearson_correlation"],
            "rmse": raw["errors"]["rmse"],
            "top10_iou": raw["top10"]["intersection_over_union"],
            "absolute_q90_iou": raw["absolute_q90"]["intersection_over_union"],
            "absolute_q90_active_fraction": raw["absolute_q90"][
                "prediction_active_fraction"
            ],
        },
        "observed": {
            "relative_l2": observed["errors"]["relative_l2_error"],
            "pearson": observed["errors"]["pearson_correlation"],
            "rmse": observed["errors"]["rmse"],
            "top10_iou": observed["top10"]["intersection_over_union"],
            "absolute_q90_iou": observed["absolute_q90"][
                "intersection_over_union"
            ],
            "absolute_q90_active_fraction": observed["absolute_q90"][
                "prediction_active_fraction"
            ],
        },
    }


def _write_summary(source: Path) -> None:
    rows = []
    for profile in PROFILES:
        for alpha, case, _ in CASES:
            report = _report(source / f"{case}_{profile}" / "report.json")
            rows.append(
                _summary_row(
                    alpha=alpha,
                    case=case,
                    profile=profile,
                    report=report,
                )
            )
    payload = {
        "schema_version": 1,
        "status": "completed_symmetric_image_observation",
        "partition_id": 43,
        "primary_profile": "legacy_script_2021",
        "sensitivity_profile": "declared_medium_v4",
        "mechanics_rerun": False,
        "rows": rows,
    }
    (source / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate(source: Path, figures: Path) -> None:
    _write_summary(source)
    figures.mkdir(parents=True, exist_ok=True)
    primary = {
        alpha: source / f"{label}_legacy_script_2021"
        for alpha, label, _ in CASES
    }
    dic = np.load(primary[0] / "dic_evm.npy")
    raw = {alpha: np.load(path / "fem_raw_evm.npy") for alpha, path in primary.items()}
    observed = {
        alpha: np.load(path / "fem_observed_evm.npy") for alpha, path in primary.items()
    }
    vmax = max(
        float(np.max(field))
        for field in [dic, *raw.values(), *observed.values()]
    )
    error_limit = max(
        float(np.max(np.abs(field - dic)))
        for field in [*raw.values(), *observed.values()]
    )
    figure, axes = plt.subplots(4, 4, figsize=(14, 14), constrained_layout=True)
    for row, (alpha, _, _) in enumerate(CASES):
        panels = (
            (raw[alpha], "raw FEM", "magma", 0.0, vmax),
            (observed[alpha], "DISFlow-observed FEM", "magma", 0.0, vmax),
            (raw[alpha] - dic, "raw FEM - DIC", "coolwarm", -error_limit, error_limit),
            (
                observed[alpha] - dic,
                "observed FEM - DIC",
                "coolwarm",
                -error_limit,
                error_limit,
            ),
        )
        for column, (field, title, cmap, vmin, vmax_panel) in enumerate(panels):
            axes[row, column].imshow(
                field.T,
                origin="upper",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax_panel,
            )
            axes[row, column].set_title(f"alpha={alpha}: {title}")
            axes[row, column].set_xlabel("x element")
            axes[row, column].set_ylabel("y element")
    figure.suptitle(
        "P43 raw and symmetrically observed EVM — legacy-script DISFlow profile"
    )
    for extension in ("png", "pdf"):
        figure.savefig(figures / f"p43_symmetric_observation_fields.{extension}", dpi=180)
    plt.close(figure)

    peeq: dict[int, np.ndarray] = {}
    for alpha, _, campaign in CASES:
        manifest = _report(Path(campaign) / "manifest.json")
        partition = next(
            item
            for item in manifest["layout"]["partitions"]
            if item["partition_id"] == 43
        )
        cx0, cx1, cy0, cy1 = partition["core_bounds"]
        sx0, _, sy0, _ = partition["solve_bounds"]
        core = (slice(cx0 - sx0, cx1 - sx0), slice(cy0 - sy0, cy1 - sy0))
        peeq[alpha] = np.asarray(
            np.load(Path(campaign) / "partitions/0043/PEEQ.npy", mmap_mode="r")[core]
        )
    peeq_vmax = max(float(np.max(field)) for field in peeq.values())
    figure, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for axis, (alpha, field) in zip(axes, peeq.items(), strict=True):
        image = axis.imshow(
            field.T,
            origin="upper",
            cmap="viridis",
            vmin=0.0,
            vmax=peeq_vmax,
        )
        axis.set_title(f"alpha={alpha}, max={np.max(field):.4f}")
        axis.set_xlabel("x element")
        axis.set_ylabel("y element")
    figure.colorbar(image, ax=axes, label="PEEQ (model output)")
    figure.suptitle("P43 local equivalent plastic strain — not a DIC observable")
    for extension in ("png", "pdf"):
        figure.savefig(figures / f"p43_peeq_separate.{extension}", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    metrics = (
        ("relative_l2_error", "Relative L2 error", "errors"),
        ("pearson_correlation", "Pearson correlation", "errors"),
        ("intersection_over_union", "Top-10% IoU", "top10"),
        ("intersection_over_union", "Absolute DIC-q90 IoU", "absolute_q90"),
    )
    for profile, linestyle in (
        ("legacy_script_2021", "-"),
        ("declared_medium_v4", "--"),
    ):
        for axis, (key, label, family) in zip(axes.flat, metrics, strict=True):
            raw_values = []
            observed_values = []
            for _alpha, case, _ in CASES:
                report = _report(source / f"{case}_{profile}" / "report.json")
                raw_values.append(report["metrics"]["raw"][family][key])
                observed_values.append(report["metrics"]["observed"][family][key])
            if profile == "legacy_script_2021":
                axis.plot([0, 1, 2, 4], raw_values, "o:", color="0.45", label="raw FEM")
            axis.plot(
                [0, 1, 2, 4],
                observed_values,
                "o" + linestyle,
                label=f"observed: {profile}",
            )
            axis.set_xlabel("normalized coupling alpha")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("P43 metric changes produced by the image observation operator")
    for extension in ("png", "pdf"):
        figure.savefig(figures / f"p43_symmetric_observation_metrics.{extension}", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.figures)


if __name__ == "__main__":
    main()
