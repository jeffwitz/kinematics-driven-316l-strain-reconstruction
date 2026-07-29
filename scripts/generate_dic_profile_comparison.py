"""Generate the frozen A5 DISFlow-profile comparison summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

CASES = (
    ("A", "case_a_declared_legacy_warp"),
    ("B", "case_b_declared_corrected_warp"),
    ("C", "case_c_legacy_corrected_warp"),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generate(source: Path, figures: Path) -> None:
    """Write one consolidated CSV and comparison figure."""

    figures.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    for label, directory in CASES:
        root = source / directory
        manifest = _load(root / "manifest.json")
        null = _load(root / "null_test_report.json")
        transfer = _load(root / "transfer_report.json")
        summary_rows.append(
            {
                "case": label,
                "profile": manifest["disflow_profile"],
                "warp": manifest["warp_mode"],
                "null_evm_rms": null["spurious_historical_evm"]["rms"],
                "null_to_dic_rms_ratio": null["spurious_to_step40_rms_ratio"],
                "null_autocorrelation_pixels": null[
                    "radial_autocorrelation_first_one_over_e_pixels"
                ],
                "mtf50_horizontal_pixels": transfer["mtf50_wavelength_pixels"]["horizontal"],
                "mtf50_vertical_pixels": transfer["mtf50_wavelength_pixels"]["vertical"],
            }
        )
        for row in transfer["band_rows"]:
            band_rows.append({"case": label, **row})

    with (source / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (source / "band_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(band_rows[0]))
        writer.writeheader()
        writer.writerows(band_rows)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colours = {"A": "#4c78a8", "B": "#f58518", "C": "#54a24b"}
    for case in ("A", "B", "C"):
        selected = [
            row
            for row in band_rows
            if row["case"] == case and row["orientation"] == "horizontal"
        ]
        widths = np.asarray([row["imposed_width_pixels"] for row in selected])
        axes[0, 0].plot(
            widths,
            [row["recovered_width_pixels"] for row in selected],
            "o-",
            color=colours[case],
            label=case,
        )
        axes[0, 1].plot(
            widths,
            [row["peak_gain"] for row in selected],
            "o-",
            color=colours[case],
        )
        axes[1, 0].plot(
            widths,
            [row["peak_shift_pixels"] for row in selected],
            "o-",
            color=colours[case],
        )
        axes[1, 1].plot(
            widths,
            [row["centroid_shift_pixels"] for row in selected],
            "o-",
            color=colours[case],
        )
    axes[0, 0].plot([4, 32], [4, 32], "k--", linewidth=1, label="identity")
    axes[0, 0].set_ylabel("Subpixel recovered FWHM (px)")
    axes[0, 1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_ylabel("Recovered / imposed peak")
    axes[1, 0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_ylabel("Peak shift (px)")
    axes[1, 1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_ylabel("Centroid shift (px)")
    for axis in axes.flat:
        axis.set_xlabel("Imposed FWHM (px)")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(
        title=(
            "A: V4 + legacy warp\n"
            "B: V4 + corrected warp\n"
            "C: legacy profile + corrected warp"
        ),
        fontsize=8,
    )
    figure.suptitle("DISFlow profile and forward-warp sensitivity — horizontal bands")
    for extension in ("png", "pdf"):
        figure.savefig(figures / f"dic_profile_and_warp_comparison.{extension}", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.figures)


if __name__ == "__main__":
    main()
