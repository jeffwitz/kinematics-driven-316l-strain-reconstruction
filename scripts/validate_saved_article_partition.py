#!/usr/bin/env python3
"""Validate and compare a preserved article-sized partition campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from fem_inhouse.postprocessing.kinematics import (
    plane_stress_equivalent_strain,
    strain_from_displacement,
)

FIELDS = ("U", "S", "E", "PE", "PEEQ", "RF")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _seconds(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 2:
        return 60.0 * parts[0] + parts[1]
    if len(parts) == 3:
        return 3600.0 * parts[0] + 60.0 * parts[1] + parts[2]
    raise ValueError(f"unsupported elapsed-time value: {value}")


def _resource_usage(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8")

    def match(label: str) -> str:
        found = re.search(rf"^\s*{re.escape(label)}:\s*(.+)$", text, re.MULTILINE)
        if found is None:
            raise ValueError(f"missing {label!r} in {path}")
        return found.group(1).strip()

    return {
        "process_wall_seconds": _seconds(
            match("Elapsed (wall clock) time (h:mm:ss or m:ss)")
        ),
        "user_seconds": float(match("User time (seconds)")),
        "system_seconds": float(match("System time (seconds)")),
        "average_cpu_percent": int(match("Percent of CPU this job got").removesuffix("%")),
        "maximum_resident_set_kib": int(match("Maximum resident set size (kbytes)")),
        "swaps": int(match("Swaps")),
        "exit_status": int(match("Exit status")),
    }


def _field_metrics(reference: NDArray[np.float64], value: NDArray[np.float64]) -> dict[str, float]:
    difference = value - reference
    tiny = np.finfo(float).tiny
    return {
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "relative_linf": float(np.max(np.abs(difference)) / max(np.max(np.abs(reference)), tiny)),
        "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(reference), tiny)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
    }


def _scalar_metrics(reference: NDArray[np.float64], value: NDArray[np.float64]) -> dict[str, float]:
    difference = value - reference
    return {
        "rmse_percentage_points": float(100.0 * np.sqrt(np.mean(np.square(difference)))),
        "mae_percentage_points": float(100.0 * np.mean(np.abs(difference))),
        "bias_percentage_points": float(100.0 * np.mean(difference)),
        "spatial_correlation": float(np.corrcoef(value.ravel(), reference.ravel())[0, 1]),
        "relative_l2_error": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
    }


def _atomic_save(path: Path, array: NDArray[np.float64]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array)
    temporary.replace(path)


def _write_preview(
    path: Path,
    dic_evm: NDArray[np.float64],
    fem_evm: NDArray[np.float64],
    difference: NDArray[np.float64],
    von_mises: NDArray[np.float64],
    core_shape: tuple[int, int],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    panels = (
        (dic_evm.T * 100.0, "DIC equivalent strain", "%", "viridis"),
        (fem_evm.T * 100.0, "FEM equivalent strain", "%", "viridis"),
        (difference.T * 100.0, "FEM - DIC equivalent strain", "percentage points", "coolwarm"),
        (von_mises.T, "FEM von Mises stress", "MPa", "magma"),
    )
    for axis, (field, title, unit, colour_map) in zip(axes.ravel(), panels, strict=True):
        image = axis.imshow(field, origin="lower", aspect="auto", cmap=colour_map)
        axis.axvline(core_shape[0] - 0.5, color="white", linestyle=":", linewidth=1)
        axis.axhline(core_shape[1] - 0.5, color="white", linestyle=":", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("x node/element index")
        axis.set_ylabel("y node/element index")
        figure.colorbar(image, ax=axis, label=unit)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary, dpi=180)
    plt.close(figure)
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--partition-id", type=int, default=0)
    parser.add_argument("--comparison-campaign", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest_path = args.campaign / "manifest.json"
    request_path = args.campaign / "run-request.json"
    status_path = args.campaign / "partitions" / f"{args.partition_id:04d}" / "status.json"
    partition_dir = status_path.parent
    manifest = _load_json(manifest_path)
    request = _load_json(request_path)
    status = _load_json(status_path)
    if not status.get("complete"):
        raise RuntimeError(f"partition is not marked complete: {status_path}")

    partitions = manifest["layout"]["partitions"]
    partition = next(item for item in partitions if item["partition_id"] == args.partition_id)
    solve_x0, solve_x1, solve_y0, solve_y1 = partition["solve_bounds"]
    core_x0, core_x1, core_y0, core_y1 = partition["core_bounds"]
    solve_shape = (solve_x1 - solve_x0, solve_y1 - solve_y0)
    core_shape = (core_x1 - core_x0, core_y1 - core_y0)

    arrays = {name: np.asarray(np.load(partition_dir / f"{name}.npy")) for name in FIELDS}
    expected_shapes = {
        "U": (*np.add(solve_shape, 1), 2),
        "RF": (*np.add(solve_shape, 1), 2),
        "S": (*solve_shape, 3),
        "E": (*solve_shape, 3),
        "PE": (*solve_shape, 3),
        "PEEQ": solve_shape,
    }
    fields: dict[str, Any] = {}
    hashes_match = True
    for name, array in arrays.items():
        path = partition_dir / f"{name}.npy"
        digest = _sha256(path)
        hashes_match &= digest == status["outputs"][name]
        if array.shape != expected_shapes[name]:
            raise ValueError(f"{name} has shape {array.shape}, expected {expected_shapes[name]}")
        fields[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": digest,
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }

    displacement_x = np.load(args.input / "displacement_x_mm.npy", mmap_mode="r")[
        solve_x0 : solve_x1 + 1, solve_y0 : solve_y1 + 1
    ]
    displacement_y = np.load(args.input / "displacement_y_mm.npy", mmap_mode="r")[
        solve_x0 : solve_x1 + 1, solve_y0 : solve_y1 + 1
    ]
    dic_displacement = np.stack((displacement_x, displacement_y), axis=-1)
    boundary = np.zeros(dic_displacement.shape[:2], dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    boundary_error = arrays["U"][boundary] - dic_displacement[boundary]

    spacing = (
        float(manifest["config"]["mesh"]["base_pixel_size_mm"])
        * float(manifest["config"]["mesh"]["scale_factor"])
    )
    poisson_ratio = float(manifest["config"]["material"]["poisson_ratio"])
    dic_strain = strain_from_displacement(
        displacement_x,
        displacement_y,
        spacing_x=spacing,
        spacing_y=spacing,
    )
    fem_strain = strain_from_displacement(
        arrays["U"][..., 0],
        arrays["U"][..., 1],
        spacing_x=spacing,
        spacing_y=spacing,
    )
    dic_evm = plane_stress_equivalent_strain(
        dic_strain.epsilon_xx,
        dic_strain.epsilon_yy,
        dic_strain.gamma_xy,
        poisson_ratio=poisson_ratio,
        shear_convention="engineering",
    )
    fem_evm = plane_stress_equivalent_strain(
        fem_strain.epsilon_xx,
        fem_strain.epsilon_yy,
        fem_strain.gamma_xy,
        poisson_ratio=poisson_ratio,
        shear_convention="engineering",
    )
    difference_evm = fem_evm - dic_evm
    s_xx, s_yy, tau_xy = np.moveaxis(arrays["S"], -1, 0)
    von_mises = np.sqrt(np.square(s_xx) - s_xx * s_yy + np.square(s_yy) + 3 * np.square(tau_xy))

    derived_dir = args.campaign / "derived"
    derived_dir.mkdir(exist_ok=True)
    derived_arrays = {
        "DIC_EVM": dic_evm,
        "FEM_EVM": fem_evm,
        "DIFF_EVM": difference_evm,
        "S_MISES": von_mises,
    }
    for name, array in derived_arrays.items():
        _atomic_save(derived_dir / f"{name}.npy", array)
    preview_path = args.campaign / "preview.png"
    _write_preview(preview_path, dic_evm, fem_evm, difference_evm, von_mises, core_shape)

    reactions = arrays["RF"].reshape(-1, 2)
    resultant = np.sum(reactions, axis=0)
    absolute_reaction = np.sum(np.abs(reactions), axis=0)
    core_node_slices = (
        slice(core_x0 - solve_x0, core_x1 - solve_x0 + 1),
        slice(core_y0 - solve_y0, core_y1 - solve_y0 + 1),
    )
    diagnostics = status["diagnostics"]
    resource = _resource_usage(args.campaign / "resource-usage.txt")
    performance = {
        **resource,
        "solver_elapsed_seconds": diagnostics["elapsed_seconds"],
        **{
            key: value
            for key, value in diagnostics.items()
            if key.endswith("_seconds") and key != "elapsed_seconds"
        },
    }

    comparison: dict[str, Any] | None = None
    if args.comparison_campaign is not None:
        comparison_partition = (
            args.comparison_campaign / "partitions" / f"{args.partition_id:04d}"
        )
        comparison_status = _load_json(comparison_partition / "status.json")
        comparison_resource = _resource_usage(args.comparison_campaign / "resource-usage.txt")
        comparison_metrics = {
            name: _field_metrics(
                np.asarray(np.load(comparison_partition / f"{name}.npy")),
                arrays[name],
            )
            for name in FIELDS
        }
        old_diagnostics = comparison_status["diagnostics"]
        old_wall = float(comparison_resource["process_wall_seconds"])
        old_solver = float(old_diagnostics["elapsed_seconds"])
        old_memory = int(comparison_resource["maximum_resident_set_kib"])
        comparison = {
            "reference": str(args.comparison_campaign),
            "reference_backend": old_diagnostics["backend"],
            "candidate_backend": diagnostics["backend"],
            "field_metrics": comparison_metrics,
            "performance": {
                "reference_wall_seconds": old_wall,
                "candidate_wall_seconds": resource["process_wall_seconds"],
                "wall_speedup": old_wall / float(resource["process_wall_seconds"]),
                "wall_time_reduction_fraction": (
                    old_wall - float(resource["process_wall_seconds"])
                )
                / old_wall,
                "reference_solver_seconds": old_solver,
                "candidate_solver_seconds": diagnostics["elapsed_seconds"],
                "solver_speedup": old_solver / float(diagnostics["elapsed_seconds"]),
                "reference_constitutive_seconds": old_diagnostics["constitutive_seconds"],
                "candidate_constitutive_seconds": diagnostics["constitutive_seconds"],
                "constitutive_speedup": old_diagnostics["constitutive_seconds"]
                / diagnostics["constitutive_seconds"],
                "reference_maximum_resident_set_kib": old_memory,
                "candidate_maximum_resident_set_kib": resource[
                    "maximum_resident_set_kib"
                ],
                "memory_change_fraction": (
                    int(resource["maximum_resident_set_kib"]) - old_memory
                )
                / old_memory,
            },
        }

    peeq = arrays["PEEQ"]
    all_fields_finite = all(np.isfinite(array).all() for array in arrays.values())
    report = {
        "schema_version": 1,
        "outcome": "converged",
        "scope": {
            "global_element_shape": manifest["layout"]["global_shape"],
            "partition_id": args.partition_id,
            "core_element_shape": list(core_shape),
            "padding_elements": manifest["layout"]["padding"],
            "solve_element_shape": list(solve_shape),
            "element_count": int(np.prod(solve_shape)),
        },
        "constitutive_model": {
            "backend": manifest["config"]["solver"]["constitutive_backend"],
            "hardening_mode": manifest["config"]["solver"]["hardening_mode"],
            "peeq_cap": request["solver"]["peeq_cap"],
            "mfront": request["mfront"],
        },
        "provenance": {
            "software_git_sha": request["software_git_sha"],
            "prepared_input_manifest_sha256": request["prepared_input_manifest"]["sha256"],
            "workflow_manifest_sha256": _sha256(manifest_path),
            "status_sha256": _sha256(status_path),
            "run_log_sha256": _sha256(args.campaign / "run.log"),
            "resource_usage_sha256": _sha256(args.campaign / "resource-usage.txt"),
        },
        "convergence": {
            key: diagnostics[key]
            for key in (
                "backend",
                "attempted_increments",
                "converged_increments",
                "cutbacks",
                "total_newton_iterations",
                "maximum_newton_iterations",
                "final_convergence_criterion",
                "final_residual_norm",
                "final_relative_residual",
            )
        },
        "performance": performance,
        "integrity": {
            "all_fields_finite": all_fields_finite,
            "all_hashes_match_status": hashes_match,
            "fields": fields,
        },
        "mechanical_checks": {
            "dic_boundary_node_count": int(np.count_nonzero(boundary)),
            "dic_boundary_max_abs_error_mm": float(np.max(np.abs(boundary_error))),
            "dic_boundary_rms_error_mm": float(np.sqrt(np.mean(np.square(boundary_error)))),
            "reaction_resultant": resultant.tolist(),
            "reaction_absolute_sum": absolute_reaction.tolist(),
            "reaction_balance_ratio": float(
                np.linalg.norm(resultant) / np.sum(absolute_reaction)
            ),
            "von_mises_mpa": {
                "min": float(np.min(von_mises)),
                "max": float(np.max(von_mises)),
                "mean": float(np.mean(von_mises)),
                "p95": float(np.percentile(von_mises, 95)),
            },
            "peeq": {
                "min": float(np.min(peeq)),
                "max": float(np.max(peeq)),
                "mean": float(np.mean(peeq)),
                "positive_fraction": float(np.mean(peeq > 0.0)),
                "above_legacy_0_2_cap_fraction": float(np.mean(peeq > 0.2)),
            },
        },
        "derived_fields": {
            name: {
                "shape": list(array.shape),
                "sha256": _sha256(derived_dir / f"{name}.npy"),
            }
            for name, array in derived_arrays.items()
        }
        | {"preview_png_sha256": _sha256(preview_path)},
        "dic_fem_comparison": {
            "displacement": {
                "U1_rmse_mm_full_solve": float(
                    np.sqrt(np.mean(np.square(arrays["U"][..., 0] - displacement_x)))
                ),
                "U1_mae_mm_full_solve": float(
                    np.mean(np.abs(arrays["U"][..., 0] - displacement_x))
                ),
                "U2_rmse_mm_full_solve": float(
                    np.sqrt(np.mean(np.square(arrays["U"][..., 1] - displacement_y)))
                ),
                "U2_mae_mm_full_solve": float(
                    np.mean(np.abs(arrays["U"][..., 1] - displacement_y))
                ),
            },
            "equivalent_strain_full_solve": _scalar_metrics(dic_evm, fem_evm),
            "equivalent_strain_core_nodes": {
                "shape": list(dic_evm[core_node_slices].shape),
                **_scalar_metrics(
                    dic_evm[core_node_slices],
                    fem_evm[core_node_slices],
                ),
            },
        },
        "comparison_to_saved_campaign": comparison,
    }
    report_path = args.campaign / "validation-report.json"
    temporary_report = report_path.with_name(f".{report_path.name}.tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_report.replace(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all_fields_finite and hashes_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
