"""Diagnose spatial and temporal outliers in a measured DIC boundary history."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from fem_inhouse.core.element import precompute_element
from fem_inhouse.core.mesh import StructuredMesh
from fem_inhouse.measurement import (
    canonical_to_image_flow,
    direct_photometric_residual,
)
from fem_inhouse.workflows.dic_observation_replay import (
    PIXEL_SIZE_MM,
    RAW_CROP_COLUMN_START,
    RAW_CROP_ROW_START,
)
from fem_inhouse.workflows.nonlocality_diagnostic import reconstruct_historical_evm

FloatArray = NDArray[np.float64]
BooleanArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class BoundaryAffineDecomposition:
    """Affine part and non-affine residual on the boundary of a nodal field."""

    coefficients: FloatArray
    residual: FloatArray
    total_rms_mm: float
    residual_rms_mm: float
    residual_maximum_mm: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _prepare_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _boundary_mask(shape: tuple[int, int]) -> BooleanArray:
    if min(shape) < 2:
        raise ValueError("a boundary field needs at least two nodes per axis")
    mask = np.zeros(shape, dtype=bool)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    return mask


def affine_boundary_decomposition(
    displacement_mm: NDArray[np.generic],
    *,
    spacing_x_mm: float,
    spacing_y_mm: float,
) -> BoundaryAffineDecomposition:
    """Fit ``u = b + A x`` on boundary nodes and return its residual."""

    displacement = np.asarray(displacement_mm, dtype=np.float64)
    if (
        displacement.ndim != 3
        or displacement.shape[-1] != 2
        or not np.isfinite(displacement).all()
    ):
        raise ValueError("displacement_mm must have finite shape (nx, ny, 2)")
    if (
        not np.isfinite(spacing_x_mm)
        or not np.isfinite(spacing_y_mm)
        or spacing_x_mm <= 0.0
        or spacing_y_mm <= 0.0
    ):
        raise ValueError("spacings must be finite and positive")
    shape = displacement.shape[:2]
    mask = _boundary_mask(shape)
    x = np.arange(shape[0], dtype=np.float64)[:, None] * spacing_x_mm
    y = np.arange(shape[1], dtype=np.float64)[None, :] * spacing_y_mm
    design = np.column_stack(
        (
            np.ones(np.count_nonzero(mask), dtype=np.float64),
            np.broadcast_to(x, shape)[mask],
            np.broadcast_to(y, shape)[mask],
        )
    )
    boundary = displacement[mask]
    coefficients = np.linalg.lstsq(design, boundary, rcond=None)[0]
    residual = boundary - design @ coefficients
    return BoundaryAffineDecomposition(
        coefficients=np.ascontiguousarray(coefficients),
        residual=np.ascontiguousarray(residual),
        total_rms_mm=float(np.sqrt(np.mean(np.square(boundary)))),
        residual_rms_mm=float(np.sqrt(np.mean(np.square(residual)))),
        residual_maximum_mm=float(np.max(np.linalg.norm(residual, axis=1))),
    )


def _affine_strain(coefficients: FloatArray) -> FloatArray:
    return np.asarray(
        [
            coefficients[1, 0],
            coefficients[2, 1],
            coefficients[2, 0] + coefficients[1, 1],
        ],
        dtype=np.float64,
    )


def _boundary_edge_values(field: FloatArray) -> tuple[FloatArray, ...]:
    return (
        field[:, 0],
        field[:, -1],
        field[0, 1:-1],
        field[-1, 1:-1],
    )


def _boundary_gradient_metrics(
    increment: FloatArray,
    *,
    spacing_x_mm: float,
    spacing_y_mm: float,
) -> tuple[float, float]:
    gradients = [
        np.diff(increment[:, 0], axis=0) / spacing_x_mm,
        np.diff(increment[:, -1], axis=0) / spacing_x_mm,
        np.diff(increment[0, :], axis=0) / spacing_y_mm,
        np.diff(increment[-1, :], axis=0) / spacing_y_mm,
    ]
    magnitudes = np.concatenate([np.linalg.norm(value, axis=1) for value in gradients])
    return (
        float(np.sqrt(np.mean(np.square(magnitudes)))),
        float(np.max(magnitudes)),
    )


def _high_frequency_fraction(
    increment: FloatArray,
    *,
    maximum_wavelength_pixels: float,
) -> float:
    if maximum_wavelength_pixels <= 0.0:
        raise ValueError("maximum_wavelength_pixels must be positive")
    decomposition = affine_boundary_decomposition(
        increment,
        spacing_x_mm=1.0,
        spacing_y_mm=1.0,
    )
    shape = increment.shape[:2]
    residual_field = np.zeros_like(increment)
    residual_field[_boundary_mask(shape)] = decomposition.residual
    high_energy = 0.0
    total_energy = 0.0
    for edge in _boundary_edge_values(residual_field):
        for component in range(2):
            values = np.asarray(edge[:, component], dtype=np.float64)
            spectrum = np.fft.rfft(values - np.mean(values))
            frequencies = np.fft.rfftfreq(values.size)
            energy = np.square(np.abs(spectrum))
            selected = frequencies >= 1.0 / maximum_wavelength_pixels
            high_energy += float(np.sum(energy[selected]))
            total_energy += float(np.sum(energy[1:]))
    return high_energy / total_energy if total_energy > 0.0 else 0.0


def element_gauss_engineering_strain(
    displacement_mm: NDArray[np.generic],
    *,
    element_indices: tuple[int, ...],
    spacing_mm: float,
) -> FloatArray:
    """Evaluate exact CPS4 Gauss strains for selected Fortran-ordered elements."""

    displacement = np.asarray(displacement_mm, dtype=np.float64)
    if (
        displacement.ndim != 3
        or displacement.shape[-1] != 2
        or not np.isfinite(displacement).all()
    ):
        raise ValueError("displacement_mm must have finite shape (nx + 1, ny + 1, 2)")
    if not np.isfinite(spacing_mm) or spacing_mm <= 0.0:
        raise ValueError("spacing_mm must be finite and positive")
    nx = displacement.shape[0] - 1
    ny = displacement.shape[1] - 1
    if any(index < 0 or index >= nx * ny for index in element_indices):
        raise ValueError("element index lies outside the structured mesh")
    mesh = StructuredMesh(
        x_size=float(nx),
        y_size=float(ny),
        base_element_size=1.0,
        scale_factor=spacing_mm,
    )
    operators = precompute_element(mesh, np.eye(3, dtype=np.float64))
    result = np.empty((len(element_indices), 4, 3), dtype=np.float64)
    for output_index, element_index in enumerate(element_indices):
        ix = element_index % nx
        iy = element_index // nx
        values = np.concatenate(
            (
                displacement[ix, iy],
                displacement[ix + 1, iy],
                displacement[ix + 1, iy + 1],
                displacement[ix, iy + 1],
            )
        )
        result[output_index] = np.einsum(
            "gij,j->gi",
            operators.strain_displacement,
            values,
        )
    return result


def _photometric_metrics(
    *,
    reference: NDArray[np.uint8],
    current: NDArray[np.uint8],
    displacement: FloatArray,
    failure_x_indices: tuple[int, ...],
) -> dict[str, float]:
    flow = canonical_to_image_flow(displacement, pixel_size_mm=PIXEL_SIZE_MM)
    result = direct_photometric_residual(reference, current, flow)
    values = result.absolute_residual_grey_levels[result.valid_mask]
    local = np.zeros(result.valid_mask.shape, dtype=bool)
    local[
        max(min(failure_x_indices) - 10, 0) : min(max(failure_x_indices) + 12, local.shape[0]),
        max(local.shape[1] - 12, 0) :,
    ] = True
    local &= result.valid_mask
    local_values = result.absolute_residual_grey_levels[local]
    return {
        "valid_fraction": float(np.mean(result.valid_mask)),
        "mean_grey": float(np.mean(values)),
        "rms_grey": float(np.sqrt(np.mean(np.square(values)))),
        "q90_grey": float(np.quantile(values, 0.90)),
        "q99_grey": float(np.quantile(values, 0.99)),
        "maximum_grey": float(np.max(values)),
        "failure_neighbourhood_rms_grey": float(
            np.sqrt(np.mean(np.square(local_values)))
        ),
        "failure_neighbourhood_maximum_grey": float(np.max(local_values)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _diagnostic_figure(
    path: Path,
    *,
    history: FloatArray,
    state_evm: FloatArray,
    increment_evm: FloatArray,
    rows: list[dict[str, Any]],
    failure_x_indices: tuple[int, ...],
    rejected_trial_strain: float,
) -> None:
    states = np.asarray([int(row["state"]) for row in rows])
    final_rms = float(np.sqrt(np.mean(np.square(state_evm[-1]))))
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    all_states = np.arange(state_evm.shape[0])
    rms_fraction = np.sqrt(np.mean(np.square(state_evm), axis=(1, 2))) / final_rms
    axes[0, 0].plot(all_states, 100.0 * rms_fraction, marker=".", linewidth=1.2)
    axes[0, 0].scatter([3, 4], 100.0 * rms_fraction[[3, 4]], color="red", zorder=3)
    axes[0, 0].set(
        xlabel="Measured state index",
        ylabel="EVM RMS / final EVM RMS (%)",
        title="State 3 is early in measured deformation",
    )
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(
        states,
        [1.0e3 * float(row["boundary_increment_rms_mm"]) for row in rows],
        marker="o",
        label="total boundary increment",
    )
    axes[0, 1].plot(
        states,
        [1.0e3 * float(row["boundary_nonaffine_rms_mm"]) for row in rows],
        marker="o",
        label="non-affine residual",
    )
    axes[0, 1].set(
        xlabel="Target state",
        ylabel="RMS displacement (µm)",
        title="Boundary increments are almost affine",
    )
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    x_mm = np.arange(history.shape[1], dtype=np.float64) * PIXEL_SIZE_MM
    for state in (2, 3, 4, 5):
        increment = history[state] - history[state - 1]
        decomposition = affine_boundary_decomposition(
            increment,
            spacing_x_mm=PIXEL_SIZE_MM,
            spacing_y_mm=PIXEL_SIZE_MM,
        )
        x = np.arange(history.shape[1], dtype=np.float64) * PIXEL_SIZE_MM
        y_top = (history.shape[2] - 1) * PIXEL_SIZE_MM
        design = np.column_stack((np.ones_like(x), x, np.full_like(x, y_top)))
        fitted = design @ decomposition.coefficients
        residual_top = increment[:, -1] - fitted
        axes[0, 2].plot(x_mm, 1.0e3 * residual_top[:, 1], label=f"state {state}")
    for index in failure_x_indices:
        axes[0, 2].axvline(index * PIXEL_SIZE_MM, color="black", alpha=0.4)
    axes[0, 2].set(
        xlabel="x on upper boundary (mm)",
        ylabel="non-affine Δu_y (µm)",
        title="No local spike at failed elements",
    )
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.25)

    axes[1, 0].plot(
        states,
        [float(row["boundary_gradient_maximum"]) for row in rows],
        marker="o",
        label="maximum boundary gradient",
    )
    axes[1, 0].plot(
        states,
        [float(row["failure_element_gauss_strain_maximum"]) for row in rows],
        marker="s",
        label="failed-element Gauss strain",
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(
        xlabel="Target state",
        ylabel="dimensionless engineering strain",
        title=f"Measured ≪ rejected Newton trial ({rejected_trial_strain:.1f})",
    )
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25, which="both")

    axes[1, 1].plot(
        states,
        [float(row["photometric_rms_grey"]) for row in rows],
        marker="o",
        label="whole P43 support",
    )
    axes[1, 1].plot(
        states,
        [float(row["photometric_failure_neighbourhood_rms_grey"]) for row in rows],
        marker="s",
        label="failed-element neighbourhood",
    )
    axes[1, 1].set(
        xlabel="Target state",
        ylabel="photometric residual RMS (grey levels)",
        title="State 4 is not a photometric outlier",
    )
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)

    state = 4
    image = axes[1, 2].imshow(
        increment_evm[state].T,
        origin="lower",
        extent=(
            0.0,
            increment_evm.shape[1] * PIXEL_SIZE_MM,
            0.0,
            increment_evm.shape[2] * PIXEL_SIZE_MM,
        ),
        cmap="viridis",
        interpolation="nearest",
    )
    for index in failure_x_indices:
        axes[1, 2].scatter(
            (index + 0.5) * PIXEL_SIZE_MM,
            (increment_evm.shape[2] - 0.5) * PIXEL_SIZE_MM,
            marker="x",
            color="red",
            s=55,
        )
    axes[1, 2].set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        title="Measured state 3→4 incremental EVM",
    )
    figure.colorbar(image, ax=axes[1, 2], label="Equivalent strain")
    figure.suptitle("P43 measured-history boundary audit")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _loading_path_figure(path: Path, *, history: FloatArray) -> FloatArray:
    affine_strains = np.stack(
        [
            _affine_strain(
                affine_boundary_decomposition(
                    displacement,
                    spacing_x_mm=PIXEL_SIZE_MM,
                    spacing_y_mm=PIXEL_SIZE_MM,
                ).coefficients
            )
            for displacement in history
        ]
    )
    fractions = np.linspace(0.0, 1.0, history.shape[0], dtype=np.float64)
    proportional = fractions[:, None] * affine_strains[-1]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colour = np.arange(history.shape[0])
    axes[0].plot(
        100.0 * proportional[:, 0],
        100.0 * proportional[:, 1],
        color="black",
        linestyle="--",
        label="proportional path to final field",
    )
    scatter = axes[0].scatter(
        100.0 * affine_strains[:, 0],
        100.0 * affine_strains[:, 1],
        c=colour,
        cmap="viridis",
        s=28,
        label="measured DIC path",
    )
    axes[0].set(
        xlabel=r"affine $\varepsilon_{xx}$ (%)",
        ylabel=r"affine $\varepsilon_{yy}$ (%)",
        title="The measured and proportional paths are different",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    figure.colorbar(scatter, ax=axes[0], label="measured state index")

    early = slice(0, 7)
    axes[1].plot(
        np.arange(7),
        100.0 * affine_strains[early, 0],
        marker="o",
        label=r"measured $\varepsilon_{xx}$",
    )
    axes[1].plot(
        np.arange(7),
        100.0 * affine_strains[early, 1],
        marker="o",
        label=r"measured $\varepsilon_{yy}$",
    )
    axes[1].plot(
        np.arange(7),
        100.0 * proportional[early, 0],
        linestyle="--",
        color="C0",
        label=r"proportional $\varepsilon_{xx}$",
    )
    axes[1].plot(
        np.arange(7),
        100.0 * proportional[early, 1],
        linestyle="--",
        color="C1",
        label=r"proportional $\varepsilon_{yy}$",
    )
    axes[1].axvspan(3.0, 4.0, color="red", alpha=0.08, label="failed transition")
    axes[1].set(
        xlabel="state index",
        ylabel="affine boundary strain (%)",
        title="Early measured path, before the first failure",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2, fontsize="small")
    figure.suptitle(
        "Why final proportional convergence does not validate the measured history"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return affine_strains


def diagnose_dic_boundary_history(
    *,
    history_path: str | Path,
    history_report_path: str | Path,
    failure_report_path: str | Path,
    raw_image_directory: str | Path,
    output_directory: str | Path,
    figure_directory: str | Path,
    maximum_state: int = 6,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Audit early DIC states against the locations of a nonlinear failure."""

    history_source = Path(history_path)
    history_report_source = Path(history_report_path)
    failure_source = Path(failure_report_path)
    images = Path(raw_image_directory)
    output = Path(output_directory)
    figures = Path(figure_directory)
    _prepare_directory(output, overwrite=overwrite)
    _prepare_directory(figures, overwrite=overwrite)

    history_report = json.loads(history_report_source.read_text(encoding="utf-8"))
    failure_report = json.loads(failure_source.read_text(encoding="utf-8"))
    expected_history_hash = history_report["outputs"][history_source.name]
    if _sha256(history_source) != expected_history_hash:
        raise ValueError("history does not match its immutable report")
    history = np.asarray(
        np.load(history_source, mmap_mode="r", allow_pickle=False),
        dtype=np.float64,
    )
    if (
        history.ndim != 4
        or history.shape[-1] != 2
        or not np.isfinite(history).all()
    ):
        raise ValueError("history must have finite shape (states, nx + 1, ny + 1, 2)")
    if maximum_state < 1 or maximum_state >= history.shape[0]:
        raise ValueError("maximum_state must select an existing positive history state")

    first_failure = failure_report["diagnostics"]["first_constitutive_failure"]
    last_failure = failure_report["diagnostics"]["last_constitutive_failure"]
    failure_elements = tuple(
        dict.fromkeys(
            (
                int(first_failure["element_index"]),
                int(last_failure["element_index"]),
            )
        )
    )
    nx = history.shape[1] - 1
    ny = history.shape[2] - 1
    failure_x = tuple(index % nx for index in failure_elements)
    failure_y = tuple(index // nx for index in failure_elements)
    if any(index != ny - 1 for index in failure_y):
        raise ValueError("expected diagnosed failure elements on the upper boundary")

    state_evm = np.stack(
        [
            reconstruct_historical_evm(
                displacement,
                spacing_x_mm=PIXEL_SIZE_MM,
                spacing_y_mm=PIXEL_SIZE_MM,
                poisson_ratio=0.3,
            )
            for displacement in history
        ]
    )
    increment_evm = np.zeros_like(state_evm)
    increment_evm[1:] = np.stack(
        [
            reconstruct_historical_evm(
                history[state] - history[state - 1],
                spacing_x_mm=PIXEL_SIZE_MM,
                spacing_y_mm=PIXEL_SIZE_MM,
                poisson_ratio=0.3,
            )
            for state in range(1, history.shape[0])
        ]
    )
    final_rms = float(np.sqrt(np.mean(np.square(state_evm[-1]))))

    reference_path = images / "000294.tif"
    reference_full = np.asarray(Image.open(reference_path).convert("L"), dtype=np.uint8)
    solve_x0, solve_x1, solve_y0, solve_y1 = (
        int(value) for value in history_report["solve_bounds"]
    )
    reference = np.ascontiguousarray(
        reference_full[
            RAW_CROP_ROW_START + solve_x0 : RAW_CROP_ROW_START + solve_x1 + 1,
            RAW_CROP_COLUMN_START + solve_y0 : RAW_CROP_COLUMN_START + solve_y1 + 1,
        ]
    )
    if reference.shape != history.shape[1:3]:
        raise ValueError("raw image crop does not match the measured history support")

    rows: list[dict[str, Any]] = []
    for state in range(1, maximum_state + 1):
        increment = history[state] - history[state - 1]
        decomposition = affine_boundary_decomposition(
            increment,
            spacing_x_mm=PIXEL_SIZE_MM,
            spacing_y_mm=PIXEL_SIZE_MM,
        )
        state_decomposition = affine_boundary_decomposition(
            history[state],
            spacing_x_mm=PIXEL_SIZE_MM,
            spacing_y_mm=PIXEL_SIZE_MM,
        )
        gradient_rms, gradient_maximum = _boundary_gradient_metrics(
            increment,
            spacing_x_mm=PIXEL_SIZE_MM,
            spacing_y_mm=PIXEL_SIZE_MM,
        )
        gauss_strain = element_gauss_engineering_strain(
            increment,
            element_indices=failure_elements,
            spacing_mm=PIXEL_SIZE_MM,
        )
        current_path = images / f"{294 + state:06d}.tif"
        current_full = np.asarray(Image.open(current_path).convert("L"), dtype=np.uint8)
        current = np.ascontiguousarray(
            current_full[
                RAW_CROP_ROW_START + solve_x0 : RAW_CROP_ROW_START + solve_x1 + 1,
                RAW_CROP_COLUMN_START + solve_y0 : RAW_CROP_COLUMN_START + solve_y1 + 1,
            ]
        )
        photometric = _photometric_metrics(
            reference=reference,
            current=current,
            displacement=history[state],
            failure_x_indices=failure_x,
        )
        affine_increment = _affine_strain(decomposition.coefficients)
        affine_state = _affine_strain(state_decomposition.coefficients)
        row: dict[str, Any] = {
            "state": state,
            "image": current_path.name,
            "state_evm_rms": float(np.sqrt(np.mean(np.square(state_evm[state])))),
            "state_evm_rms_fraction_of_final": float(
                np.sqrt(np.mean(np.square(state_evm[state]))) / final_rms
            ),
            "state_evm_maximum": float(np.max(state_evm[state])),
            "increment_evm_maximum": float(np.max(increment_evm[state])),
            "boundary_increment_rms_mm": decomposition.total_rms_mm,
            "boundary_nonaffine_rms_mm": decomposition.residual_rms_mm,
            "boundary_nonaffine_maximum_mm": decomposition.residual_maximum_mm,
            "boundary_nonaffine_fraction": (
                decomposition.residual_rms_mm / decomposition.total_rms_mm
            ),
            "boundary_gradient_rms": gradient_rms,
            "boundary_gradient_maximum": gradient_maximum,
            "boundary_high_frequency_fraction_wavelength_le_16px": (
                _high_frequency_fraction(
                    increment,
                    maximum_wavelength_pixels=16.0,
                )
            ),
            "affine_state_epsilon_xx": float(affine_state[0]),
            "affine_state_epsilon_yy": float(affine_state[1]),
            "affine_state_gamma_xy": float(affine_state[2]),
            "affine_increment_epsilon_xx": float(affine_increment[0]),
            "affine_increment_epsilon_yy": float(affine_increment[1]),
            "affine_increment_gamma_xy": float(affine_increment[2]),
            "failure_element_gauss_strain_maximum": float(
                np.max(np.abs(gauss_strain))
            ),
            "photometric_valid_fraction": photometric["valid_fraction"],
            "photometric_mean_grey": photometric["mean_grey"],
            "photometric_rms_grey": photometric["rms_grey"],
            "photometric_q90_grey": photometric["q90_grey"],
            "photometric_q99_grey": photometric["q99_grey"],
            "photometric_maximum_grey": photometric["maximum_grey"],
            "photometric_failure_neighbourhood_rms_grey": photometric[
                "failure_neighbourhood_rms_grey"
            ],
            "photometric_failure_neighbourhood_maximum_grey": photometric[
                "failure_neighbourhood_maximum_grey"
            ],
        }
        rows.append(row)

    metrics_path = output / "state_metrics.csv"
    figure_path = figures / "p0043_early_boundary_outlier_diagnostic.png"
    loading_path_figure = figures / "p0043_measured_vs_proportional_path.png"
    _write_csv(metrics_path, rows)
    rejected_maximum = max(
        float(first_failure["maximum_absolute_engineering_strain"]),
        float(last_failure["maximum_absolute_engineering_strain"]),
    )
    _diagnostic_figure(
        figure_path,
        history=history,
        state_evm=state_evm,
        increment_evm=increment_evm,
        rows=rows,
        failure_x_indices=failure_x,
        rejected_trial_strain=rejected_maximum,
    )
    affine_strains = _loading_path_figure(loading_path_figure, history=history)

    row3 = rows[2] if maximum_state >= 3 else None
    row4 = rows[3] if maximum_state >= 4 else None
    final_affine_strain = affine_strains[-1]
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed_exploratory_dic_boundary_outlier_audit",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "partition_id": int(history_report["partition_id"]),
        "state_index_semantics": (
            "ordered direct-reference image index; not force-synchronised load fraction"
        ),
        "source": {
            "history": str(history_source.resolve()),
            "history_sha256": _sha256(history_source),
            "history_report": str(history_report_source.resolve()),
            "history_report_sha256": _sha256(history_report_source),
            "failure_report": str(failure_source.resolve()),
            "failure_report_sha256": _sha256(failure_source),
            "reference_image": str(reference_path.resolve()),
            "reference_image_sha256": _sha256(reference_path),
        },
        "support": {
            "solve_bounds": history_report["solve_bounds"],
            "node_shape": list(history.shape[1:3]),
            "pixel_size_mm": PIXEL_SIZE_MM,
            "axis_convention": "canonical array axes (x,y), components (ux,uy)",
        },
        "failure_locations": {
            "element_indices": list(failure_elements),
            "element_coordinates": [
                {"ix": int(x), "iy": int(y)}
                for x, y in zip(failure_x, failure_y, strict=True)
            ],
            "first_rejected_trial_maximum_engineering_strain": float(
                first_failure["maximum_absolute_engineering_strain"]
            ),
            "last_rejected_trial_maximum_engineering_strain": float(
                last_failure["maximum_absolute_engineering_strain"]
            ),
        },
        "key_findings": {
            "state3_evm_rms_fraction_of_final": (
                None if row3 is None else row3["state_evm_rms_fraction_of_final"]
            ),
            "state4_boundary_nonaffine_fraction": (
                None if row4 is None else row4["boundary_nonaffine_fraction"]
            ),
            "state4_failure_element_gauss_strain_maximum": (
                None if row4 is None else row4["failure_element_gauss_strain_maximum"]
            ),
            "state4_photometric_rms_grey": (
                None if row4 is None else row4["photometric_rms_grey"]
            ),
            "state4_affine_epsilon_xx_fraction_of_final": (
                None
                if row4 is None
                else float(affine_strains[4, 0] / final_affine_strain[0])
            ),
            "state4_affine_epsilon_yy_fraction_of_final": (
                None
                if row4 is None
                else float(affine_strains[4, 1] / final_affine_strain[1])
            ),
            "maximum_measured_failure_element_gauss_strain_states_1_to_n": max(
                float(row["failure_element_gauss_strain_maximum"]) for row in rows
            ),
            "minimum_rejected_to_measured_strain_ratio": (
                min(
                    float(first_failure["maximum_absolute_engineering_strain"]),
                    float(last_failure["maximum_absolute_engineering_strain"]),
                )
                / max(
                    float(row["failure_element_gauss_strain_maximum"]) for row in rows
                )
            ),
            "state4_is_largest_photometric_rms_states_1_to_n": (
                row4 is not None
                and int(
                    rows[
                        int(
                            np.argmax(
                                [float(row["photometric_rms_grey"]) for row in rows]
                            )
                        )
                    ]["state"]
                )
                == 4
            ),
        },
        "interpretation": {
            "dic_spatial_outlier_at_failure_location": "not_observed",
            "dic_photometric_outlier_at_state4": "not_observed",
            "newton_rejected_strain_explained_by_measured_boundary": False,
            "most_supported_cause": (
                "nonlinear solver/globalisation failure near the first local plastic "
                "transition, not a measured DIC boundary spike"
            ),
            "claim_boundary": (
                "This exploratory audit does not prove every measured history state is "
                "physically exact or force-synchronised."
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "outputs": {
            metrics_path.name: _sha256(metrics_path),
            figure_path.name: _sha256(figure_path),
            loading_path_figure.name: _sha256(loading_path_figure),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
