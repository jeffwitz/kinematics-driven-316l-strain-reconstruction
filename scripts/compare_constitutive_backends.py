#!/usr/bin/env python3
"""Persist a material-point comparison of the Python and MFront backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from fem_inhouse.core.constitutive import (
    PLANE_STRESS_VON_MISES_METRIC,
    consistent_tangent,
    make_hardening,
    return_mapping,
    von_mises,
)
from fem_inhouse.core.element import plane_stress_elasticity
from fem_inhouse.core.mfront import MFrontMaterialPointBatch

DEFAULT_LIBRARY = Path("build/mfront/src/libBehaviour.so")
DEFAULT_SOURCE = Path("mfront/PixelLudwikJ2Plasticity.mfront")


@dataclass(frozen=True, slots=True)
class Material:
    young_modulus_mpa: float = 205_000.0
    poisson_ratio: float = 0.3
    initial_yield_stress_mpa: float = 250.0
    hardening_coefficient_mpa: float = 380.0
    hardening_exponent: float = 0.245


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _strain_paths(steps: int) -> dict[str, NDArray]:
    fractions = np.linspace(0.0, 1.0, steps + 1)[:, None]
    return {
        "uniaxial_strain": fractions * np.array([0.01, 0.0, 0.0]),
        "equibiaxial_strain": fractions * np.array([0.006, 0.006, 0.0]),
        "simple_shear": fractions * np.array([0.0, 0.0, 0.012]),
    }


def _python_history(strain: NDArray, material: Material) -> dict[str, NDArray]:
    point_count = len(strain)
    elasticity = plane_stress_elasticity(
        material.young_modulus_mpa,
        material.poisson_ratio,
    )
    hardening, hardening_derivative = make_hardening(
        material.hardening_exponent,
        "tabular",
    )
    constitutive_metric = elasticity @ PLANE_STRESS_VON_MISES_METRIC
    cm11 = float(constitutive_metric[0, 0])
    cm12 = float(constitutive_metric[0, 1])
    cm33 = float(constitutive_metric[2, 2])
    yield_stress = np.array([material.initial_yield_stress_mpa])
    coefficient = np.array([material.hardening_coefficient_mpa])

    stress = np.zeros((point_count, 3))
    plastic_strain = np.zeros((point_count, 3))
    equivalent_plastic_strain = np.zeros(point_count)
    yield_radius = np.full(point_count, material.initial_yield_stress_mpa)
    tangent = np.repeat(elasticity[None, :, :], point_count, axis=0)

    current_plastic_strain = np.zeros((1, 3))
    current_equivalent_strain = np.zeros(1)
    for index, total_strain in enumerate(strain[1:], start=1):
        trial_stress = (elasticity @ (total_strain - current_plastic_strain[0]))[
            None,
            :,
        ]
        mapped_stress, plastic_increment, equivalent_increment = return_mapping(
            trial_stress,
            current_equivalent_strain,
            yield_stress,
            coefficient,
            hardening,
            cm11,
            cm12,
            cm33,
        )
        if equivalent_increment[0] > 0:
            tangent[index] = consistent_tangent(
                mapped_stress,
                equivalent_increment,
                current_equivalent_strain,
                yield_stress,
                coefficient,
                hardening,
                hardening_derivative,
                elasticity,
                cm11,
                cm12,
                cm33,
            )[0]
        current_plastic_strain += plastic_increment
        current_equivalent_strain += equivalent_increment
        stress[index] = mapped_stress[0]
        plastic_strain[index] = current_plastic_strain[0]
        equivalent_plastic_strain[index] = current_equivalent_strain[0]
        yield_radius[index] = (
            material.initial_yield_stress_mpa
            + material.hardening_coefficient_mpa
            * hardening(current_equivalent_strain)[0]
        )
    return {
        "stress_mpa": stress,
        "plastic_strain": plastic_strain,
        "equivalent_plastic_strain": equivalent_plastic_strain,
        "yield_surface_radius_mpa": yield_radius,
        "consistent_tangent_mpa": tangent,
    }


def _mfront_history(
    strain: NDArray,
    material: Material,
    library: Path,
) -> dict[str, NDArray]:
    point_count = len(strain)
    stress = np.zeros((point_count, 3))
    plastic_strain = np.zeros((point_count, 3))
    equivalent_plastic_strain = np.zeros(point_count)
    yield_radius = np.full(point_count, material.initial_yield_stress_mpa)
    elasticity = plane_stress_elasticity(
        material.young_modulus_mpa,
        material.poisson_ratio,
    )
    tangent = np.repeat(elasticity[None, :, :], point_count, axis=0)
    bridge = MFrontMaterialPointBatch(
        library,
        material.initial_yield_stress_mpa,
        material.hardening_coefficient_mpa,
        material.hardening_exponent,
    )
    time_increment = 1.0 / (point_count - 1)
    for index, total_strain in enumerate(strain[1:], start=1):
        result = bridge.evaluate(
            total_strain[None, :],
            time_increment=time_increment,
            consistent_tangent=True,
            commit=True,
        )
        stress[index] = result.stress_mpa[0]
        plastic_strain[index] = result.plastic_strain[0]
        equivalent_plastic_strain[index] = result.equivalent_plastic_strain[0]
        yield_radius[index] = result.yield_surface_radius_mpa[0]
        assert result.consistent_tangent_mpa is not None
        tangent[index] = result.consistent_tangent_mpa[0]
    return {
        "stress_mpa": stress,
        "plastic_strain": plastic_strain,
        "equivalent_plastic_strain": equivalent_plastic_strain,
        "yield_surface_radius_mpa": yield_radius,
        "consistent_tangent_mpa": tangent,
    }


def _relative_l2(prediction: NDArray, reference: NDArray) -> float:
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return float(np.linalg.norm(prediction - reference) / denominator)


def _metrics(
    python_result: dict[str, NDArray],
    mfront_result: dict[str, NDArray],
    material: Material,
) -> dict[str, float | bool]:
    stress_difference = python_result["stress_mpa"] - mfront_result["stress_mpa"]
    stress_error_norm = np.linalg.norm(stress_difference, axis=1)
    peeq_difference = (
        python_result["equivalent_plastic_strain"]
        - mfront_result["equivalent_plastic_strain"]
    )
    stress_relative_l2 = _relative_l2(
        python_result["stress_mpa"],
        mfront_result["stress_mpa"],
    )
    maximum_stress_error_mpa = float(stress_error_norm.max())
    maximum_peeq_error = float(np.abs(peeq_difference).max())
    metrics: dict[str, float | bool] = {
        "stress_relative_l2": stress_relative_l2,
        "maximum_stress_error_mpa": maximum_stress_error_mpa,
        "maximum_stress_error_over_initial_yield": (
            maximum_stress_error_mpa / material.initial_yield_stress_mpa
        ),
        "peeq_relative_l2": _relative_l2(
            python_result["equivalent_plastic_strain"],
            mfront_result["equivalent_plastic_strain"],
        ),
        "maximum_peeq_absolute_error": maximum_peeq_error,
        "plastic_strain_relative_l2": _relative_l2(
            python_result["plastic_strain"],
            mfront_result["plastic_strain"],
        ),
        "tangent_relative_l2": _relative_l2(
            python_result["consistent_tangent_mpa"][1:],
            mfront_result["consistent_tangent_mpa"][1:],
        ),
    }
    metrics["stress_threshold_passed"] = bool(
        stress_relative_l2 <= 5e-3
        and metrics["maximum_stress_error_over_initial_yield"] <= 5e-2
    )
    metrics["peeq_threshold_passed"] = bool(maximum_peeq_error <= 1e-4)
    return metrics


def _plot(
    histories: dict[str, dict[str, dict[str, NDArray]]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(len(histories), 2, figsize=(12, 10), squeeze=False)
    for row, (path_name, values) in enumerate(histories.items()):
        python_result = values["python"]
        mfront_result = values["mfront"]
        step = np.arange(len(python_result["stress_mpa"]))
        axes[row, 0].plot(
            step,
            von_mises(python_result["stress_mpa"]),
            label="Python tabulé",
        )
        axes[row, 0].plot(
            step,
            von_mises(mfront_result["stress_mpa"]),
            "--",
            label="MFront",
        )
        axes[row, 0].set_ylabel("Contrainte de von Mises [MPa]")
        axes[row, 0].set_title(path_name)
        axes[row, 0].grid(alpha=0.3)
        axes[row, 1].plot(
            step,
            python_result["equivalent_plastic_strain"],
            label="Python tabulé",
        )
        axes[row, 1].plot(
            step,
            mfront_result["equivalent_plastic_strain"],
            "--",
            label="MFront",
        )
        axes[row, 1].set_ylabel("PEEQ")
        axes[row, 1].grid(alpha=0.3)
    for axis in axes[-1]:
        axis.set_xlabel("Incrément")
    axes[0, 0].legend()
    axes[0, 1].legend()
    figure.tight_layout()
    temporary = output.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=160)
    plt.close(figure)
    temporary.replace(output)


def compare(
    library: Path,
    source: Path,
    output_directory: Path,
    *,
    steps: int,
) -> bool:
    if steps < 20:
        raise ValueError("steps must be at least 20")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    material = Material()
    histories: dict[str, dict[str, dict[str, NDArray]]] = {}
    report_paths: dict[str, dict[str, float | bool]] = {}
    archive: dict[str, NDArray] = {}

    for path_name, strain in _strain_paths(steps).items():
        python_result = _python_history(strain, material)
        mfront_result = _mfront_history(strain, material, library)
        histories[path_name] = {
            "python": python_result,
            "mfront": mfront_result,
        }
        report_paths[path_name] = _metrics(python_result, mfront_result, material)
        archive[f"{path_name}__total_strain"] = strain
        for backend, result in histories[path_name].items():
            for field, values in result.items():
                archive[f"{path_name}__{backend}__{field}"] = values

    numerical_path = output_directory / "material_point_histories.npz"
    temporary_numerical_path = numerical_path.with_suffix(".tmp")
    with temporary_numerical_path.open("wb") as stream:
        np.savez_compressed(stream, **archive)
    temporary_numerical_path.replace(numerical_path)
    _plot(histories, output_directory / "comparison.png")

    passed = all(
        values["stress_threshold_passed"] and values["peeq_threshold_passed"]
        for values in report_paths.values()
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "inputs": {
            "mfront_library": _portable_path(library),
            "mfront_library_sha256": _sha256(library),
            "mfront_source": _portable_path(source),
            "mfront_source_sha256": _sha256(source),
            "steps": steps,
            "material": {
                "young_modulus_mpa": material.young_modulus_mpa,
                "poisson_ratio": material.poisson_ratio,
                "initial_yield_stress_mpa": material.initial_yield_stress_mpa,
                "hardening_coefficient_mpa": material.hardening_coefficient_mpa,
                "hardening_exponent": material.hardening_exponent,
            },
        },
        "model_difference": (
            "Python uses the 1000-point piecewise-linear table clamped at PEEQ=0.2; "
            "MFront uses the same linear first segment followed by the analytical "
            "power law. The tested histories remain below PEEQ=0.2."
        ),
        "thresholds": {
            "stress_relative_l2_max": 5e-3,
            "maximum_stress_error_over_initial_yield_max": 5e-2,
            "maximum_peeq_absolute_error_max": 1e-4,
            "tangent_is_diagnostic_only": True,
        },
        "paths": report_paths,
        "passed": passed,
        "artifacts": {
            "histories": numerical_path.name,
            "plot": "comparison.png",
        },
    }
    report_path = output_directory / "report.json"
    temporary_report_path = report_path.with_suffix(".tmp")
    temporary_report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_report_path.replace(report_path)
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    arguments = parser.parse_args()
    passed = compare(
        arguments.library,
        arguments.source,
        arguments.output,
        steps=arguments.steps,
    )
    print(arguments.output / "report.json")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
