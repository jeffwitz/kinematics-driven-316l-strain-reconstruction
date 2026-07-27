"""Reproducible section-equilibrium diagnostics for saved campaigns."""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from fem_inhouse.data_preparation import fingerprint_file
from fem_inhouse.postprocessing.section_equilibrium import (
    SectionEquilibriumResult,
    integrated_section_equilibrium,
)
from fem_inhouse.workflows.campaign_access import (
    load_json_object,
    load_partition_status,
    load_verified_partition_field,
    partition_from_manifest,
)


def _metric_record(
    *,
    label: str,
    campaign: Path,
    region: str,
    result: SectionEquilibriumResult,
) -> dict[str, Any]:
    return {
        "label": label,
        "campaign": str(campaign),
        "region": region,
        "section_force_mean_n": result.section_force_mean_n,
        "section_force_relative_dispersion": result.section_force_relative_dispersion,
        "naive_force_increment_rms_n": result.naive_force_increment_rms_n,
        "balance_residual_rms_n": result.balance_residual_rms_n,
        "balance_residual_relative_l2": result.balance_residual_relative_l2,
        "balance_residual_relative_to_mean_force": (
            result.balance_residual_relative_to_mean_force
        ),
        "boundary_flux_closure_gain": result.boundary_flux_closure_gain,
    }


def _write_profile(
    path: Path,
    *,
    solve: SectionEquilibriumResult,
    core: SectionEquilibriumResult,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "region",
                "y_mm",
                "section_force_n",
                "lateral_shear_flux_n_per_mm",
                "interval_balance_residual_n",
            )
        )
        for region, result in (("solve", solve), ("core", core)):
            residual = np.append(result.interval_balance_residual_n, np.nan)
            for values in zip(
                result.y_mm,
                result.section_force_n,
                result.lateral_shear_flux_n_per_mm,
                residual,
                strict=True,
            ):
                writer.writerow((region, *values))


def _plot_case(
    path: Path,
    *,
    label: str,
    solve: SectionEquilibriumResult,
    core: SectionEquilibriumResult,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), sharex=False, constrained_layout=True)
    for region, result, style in (
        ("padded solve domain", solve, "-"),
        ("retained core", core, "--"),
    ):
        axes[0].plot(result.y_mm, result.section_force_n, style, label=region)
        axes[1].plot(result.y_mm, result.lateral_shear_flux_n_per_mm, style, label=region)
        axes[2].plot(
            0.5 * (result.y_mm[:-1] + result.y_mm[1:]),
            result.interval_balance_residual_n,
            style,
            label=region,
        )
    axes[0].set_ylabel(r"$N_y$ (N)")
    axes[1].set_ylabel(r"$t(\sigma_{12,R}-\sigma_{12,L})$ (N/mm)")
    axes[2].set_ylabel("interval residual (N)")
    axes[2].set_xlabel("y (mm)")
    axes[0].set_title(f"Generalized section-equilibrium diagnostic — {label}")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def diagnose_section_equilibrium_campaigns(
    campaigns: Sequence[tuple[str, str | Path]],
    *,
    partition_id: int,
    output_directory: str | Path,
    thickness_mm: float,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate and plot generalized section equilibrium for saved campaigns."""

    if not campaigns:
        raise ValueError("at least one labelled campaign is required")
    labels = [label for label, _path in campaigns]
    if any(not label.strip() for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("campaign labels must be non-empty and unique")
    if not np.isfinite(thickness_mm) or thickness_mm <= 0.0:
        raise ValueError("thickness_mm must be finite and strictly positive")

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    profiles = output / "profiles"
    figures.mkdir()
    profiles.mkdir()

    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    common_layout: dict[str, Any] | None = None
    for label, campaign_value in campaigns:
        campaign = Path(campaign_value)
        manifest_path = campaign / "manifest.json"
        manifest = load_json_object(manifest_path)
        _layout, partition = partition_from_manifest(manifest, partition_id)
        if common_layout is None:
            common_layout = manifest["layout"]
        elif manifest["layout"] != common_layout:
            raise ValueError("campaigns do not use the same partition layout")
        manifest_hash = fingerprint_file(manifest_path)
        status = load_partition_status(
            campaign,
            partition_id=partition_id,
            manifest_sha256=manifest_hash,
        )
        stress = load_verified_partition_field(
            campaign,
            partition_id=partition_id,
            status=status,
            name="S",
            mmap_mode="r",
        )
        if stress.shape[:2] != partition.solve_shape:
            raise ValueError(f"stress shape disagrees with partition metadata: {campaign}")

        mesh = manifest["config"]["mesh"]
        spacing = float(mesh["base_pixel_size_mm"]) * float(mesh["scale_factor"])
        _solve_x0, _solve_x1, solve_y0, _solve_y1 = partition.solve_bounds
        _core_x0, _core_x1, core_y0, _core_y1 = partition.core_bounds
        solve_result = integrated_section_equilibrium(
            stress,
            spacing_x_mm=spacing,
            spacing_y_mm=spacing,
            thickness_mm=thickness_mm,
            y_origin_mm=solve_y0 * spacing,
        )
        core_stress = stress[partition.core_element_slice_local]
        core_result = integrated_section_equilibrium(
            core_stress,
            spacing_x_mm=spacing,
            spacing_y_mm=spacing,
            thickness_mm=thickness_mm,
            y_origin_mm=core_y0 * spacing,
        )
        records.extend(
            (
                _metric_record(label=label, campaign=campaign, region="solve", result=solve_result),
                _metric_record(label=label, campaign=campaign, region="core", result=core_result),
            )
        )
        _write_profile(profiles / f"{label}.csv", solve=solve_result, core=core_result)
        _plot_case(figures / f"{label}.png", label=label, solve=solve_result, core=core_result)
        sources.append(
            {
                "label": label,
                "campaign": str(campaign),
                "manifest_sha256": manifest_hash,
                "stress_sha256": status["outputs"]["S"],
                "solve_bounds": list(partition.solve_bounds),
                "core_bounds": list(partition.core_bounds),
                "spacing_mm": spacing,
            }
        )

    metrics_path = output / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    report = {
        "schema_version": 1,
        "status": "completed_baseline_no_acceptance_threshold",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "partition_id": partition_id,
        "case_count": len(campaigns),
        "thickness_mm": float(thickness_mm),
        "stress_convention": ["sigma_11", "sigma_22", "sigma_12"],
        "array_axes": ["x", "y", "component"],
        "equilibrium_equation": (
            "Delta N_y + Delta y * t * "
            "mean[sigma_12(x_R)-sigma_12(x_L)] = residual"
        ),
        "interpretation_boundary": (
            "P43 is an interior Dirichlet partition. Section-force constancy alone is not "
            "an equilibrium requirement; lateral shear flux must be included. Boundary "
            "tractions are approximated from cell-centred stresses."
        ),
        "sources": sources,
        "metrics": records,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
