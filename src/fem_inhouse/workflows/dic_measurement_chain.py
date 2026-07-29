"""Characterise the declared DISFlow reproduction measurement chain."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from numpy.typing import NDArray
from scipy import fft

from fem_inhouse.measurement import (
    DISFlowConfig,
    WarpMode,
    profile_metrology,
    query_disflow_configuration,
    run_disflow,
    warp_image,
)
from fem_inhouse.workflows.nonlocality_diagnostic import reconstruct_historical_evm

matplotlib.use("Agg")
from matplotlib import pyplot as plt

FloatArray = NDArray[np.float64]

PIXEL_SIZE_UM = 1.84
CROP_ROWS = slice(400, 4_000)
CROP_COLUMNS = slice(1_211, 4_311)
TRANSFER_SIZE = 1_024
SINUSOIDAL_WAVELENGTHS = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
BAND_WIDTHS = (4, 8, 16, 32)


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "DIC measurement-chain validation requires the 'measurement' dependency"
        ) from error
    return cv2


def _load_tiff(path: Path) -> NDArray[np.uint8]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("TIFF input requires Pillow") from error
    if not path.is_file():
        raise FileNotFoundError(f"missing DIC image: {path}")
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"), dtype=np.uint8)
    if values.shape != (4_400, 5_400):
        raise ValueError(f"unexpected DIC image shape {values.shape}: {path}")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _prepare_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialised = list(rows)
    if not materialised:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(materialised[0]))
        writer.writeheader()
        writer.writerows(materialised)


def queried_disflow_configuration(config: DISFlowConfig) -> dict[str, Any]:
    """Return settings queried back from the OpenCV object."""

    return query_disflow_configuration(config)


def image_flow_to_historical_evm(
    flow_pixels: NDArray[np.generic],
    *,
    poisson_ratio: float = 0.3,
) -> FloatArray:
    """Convert image flow to EVM while respecting image row/column axes."""

    flow = np.asarray(flow_pixels, dtype=np.float64)
    if flow.ndim != 3 or flow.shape[-1] != 2 or not np.isfinite(flow).all():
        raise ValueError("flow_pixels must have finite shape (rows, columns, 2)")
    displacement = np.stack((flow[..., 0].T, flow[..., 1].T), axis=-1)
    return reconstruct_historical_evm(
        displacement * (PIXEL_SIZE_UM / 1_000.0),
        spacing_x_mm=PIXEL_SIZE_UM / 1_000.0,
        spacing_y_mm=PIXEL_SIZE_UM / 1_000.0,
        poisson_ratio=poisson_ratio,
    )


def _prepared_dic_evm(prepared_case: Path) -> FloatArray:
    ux = np.load(prepared_case / "displacement_x_mm.npy", mmap_mode="r", allow_pickle=False)
    uy = np.load(prepared_case / "displacement_y_mm.npy", mmap_mode="r", allow_pickle=False)
    if ux.shape != uy.shape:
        raise ValueError("prepared displacement components have different shapes")
    # Prepared fields already follow the repository axis convention.
    displacement = np.stack((np.asarray(ux), np.asarray(uy)), axis=-1)
    return reconstruct_historical_evm(
        displacement,
        spacing_x_mm=PIXEL_SIZE_UM / 1_000.0,
        spacing_y_mm=PIXEL_SIZE_UM / 1_000.0,
        poisson_ratio=0.3,
    )


def _statistics(values: NDArray[np.generic]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("statistics require finite values")
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "minimum": float(np.min(array)),
        "q50": float(np.quantile(array, 0.5)),
        "q90": float(np.quantile(array, 0.9)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
    }


def radial_autocorrelation(
    values: NDArray[np.generic],
    *,
    maximum_radius_pixels: int = 256,
) -> tuple[list[dict[str, float]], float | None]:
    """Return circular radial autocorrelation and its first 1/e crossing."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("autocorrelation input must be finite and two-dimensional")
    centred = array - float(np.mean(array))
    if not np.any(centred):
        return [{"radius_pixels": 0.0, "correlation": 1.0}], None
    spectrum = fft.rfft2(centred)
    correlation = fft.irfft2(np.square(np.abs(spectrum)), s=centred.shape)
    correlation /= correlation[0, 0]
    row_distance = np.minimum(np.arange(array.shape[0]), array.shape[0] - np.arange(array.shape[0]))
    column_distance = np.minimum(
        np.arange(array.shape[1]), array.shape[1] - np.arange(array.shape[1])
    )
    radius = np.floor(np.hypot(row_distance[:, None], column_distance[None, :])).astype(int)
    selected = radius <= maximum_radius_pixels
    sums = np.bincount(radius[selected], weights=correlation[selected])
    counts = np.bincount(radius[selected])
    profile = sums / counts
    rows = [
        {"radius_pixels": float(index), "correlation": float(value)}
        for index, value in enumerate(profile)
    ]
    threshold = 1.0 / np.e
    crossing = next((index for index, value in enumerate(profile) if value <= threshold), None)
    if crossing is None or crossing == 0:
        return rows, None if crossing is None else 0.0
    x0, x1 = crossing - 1.0, float(crossing)
    y0, y1 = profile[crossing - 1], profile[crossing]
    fraction = (threshold - y0) / (y1 - y0)
    return rows, float(x0 + fraction * (x1 - x0))


def _photometric_residual(
    reference: NDArray[np.uint8],
    deformed: NDArray[np.uint8],
    flow: NDArray[np.float32],
) -> FloatArray:
    cv2 = _cv2()
    rows, columns = np.indices(reference.shape, dtype=np.float32)
    mapped = cv2.remap(
        deformed,
        columns + flow[..., 0],
        rows + flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return np.asarray(mapped, dtype=np.float64) - np.asarray(reference, dtype=np.float64)


def _central_window(image: NDArray[np.uint8], size: int = TRANSFER_SIZE) -> NDArray[np.uint8]:
    row_start = (image.shape[0] - size) // 2
    column_start = (image.shape[1] - size) // 2
    return np.ascontiguousarray(
        image[row_start : row_start + size, column_start : column_start + size]
    )


def _sinusoidal_displacement(
    shape: tuple[int, int],
    *,
    wavelength_pixels: int,
    orientation: str,
    amplitude_pixels: float = 0.5,
) -> NDArray[np.float32]:
    rows, columns = np.indices(shape, dtype=np.float64)
    flow = np.zeros((*shape, 2), dtype=np.float32)
    if orientation == "horizontal":
        flow[..., 0] = amplitude_pixels * np.sin(2.0 * np.pi * columns / wavelength_pixels)
    elif orientation == "vertical":
        flow[..., 1] = amplitude_pixels * np.sin(2.0 * np.pi * rows / wavelength_pixels)
    else:
        raise ValueError("orientation must be horizontal or vertical")
    return flow


def _fit_sinusoid(
    values: NDArray[np.generic],
    *,
    wavelength_pixels: int,
    orientation: str,
    border: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    core = array[border:-border, border:-border]
    coordinate = (
        np.arange(array.shape[1], dtype=float)[border:-border]
        if orientation == "horizontal"
        else np.arange(array.shape[0], dtype=float)[border:-border]
    )
    profile = np.mean(core, axis=0 if orientation == "horizontal" else 1)
    angle = 2.0 * np.pi * coordinate / wavelength_pixels
    design = np.column_stack((np.sin(angle), np.cos(angle), np.ones_like(angle)))
    coefficients, *_ = np.linalg.lstsq(design, profile, rcond=None)
    amplitude = float(np.hypot(coefficients[0], coefficients[1]))
    phase = float(np.arctan2(coefficients[1], coefficients[0]))
    return amplitude, phase


def _band_displacement(
    shape: tuple[int, int],
    *,
    width_pixels: int,
    orientation: str,
    maximum_displacement_pixels: float = 1.5,
) -> tuple[NDArray[np.float32], FloatArray]:
    size = shape[1] if orientation == "horizontal" else shape[0]
    coordinate = np.arange(size, dtype=np.float64)
    centre = 0.5 * (size - 1)
    sigma = width_pixels / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gradient = np.exp(-0.5 * np.square((coordinate - centre) / sigma))
    cumulative = np.cumsum(gradient)
    cumulative -= cumulative[0]
    cumulative *= maximum_displacement_pixels / cumulative[-1]
    flow = np.zeros((*shape, 2), dtype=np.float32)
    if orientation == "horizontal":
        flow[..., 0] = cumulative[None, :]
    elif orientation == "vertical":
        flow[..., 1] = cumulative[:, None]
    else:
        raise ValueError("orientation must be horizontal or vertical")
    return flow, np.gradient(cumulative)


def _half_maximum_width(profile: FloatArray) -> tuple[float, float]:
    peak_index = int(np.argmax(profile))
    peak = float(profile[peak_index])
    if peak <= 0.0:
        return float("nan"), float(peak_index)
    selected = np.flatnonzero(profile >= 0.5 * peak)
    if selected.size < 2:
        return 0.0, float(peak_index)
    return float(selected[-1] - selected[0] + 1), float(peak_index)


def _mtf50(rows: list[dict[str, Any]], orientation: str) -> float | None:
    selected = sorted(
        (
            (float(row["wavelength_pixels"]), float(row["gain"]))
            for row in rows
            if row["orientation"] == orientation
        ),
        key=lambda item: item[0],
    )
    for (x0, y0), (x1, y1) in pairwise(selected):
        if (y0 - 0.5) * (y1 - 0.5) <= 0.0 and y1 != y0:
            return x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
    return None


def _null_figure(
    path: Path,
    *,
    reference: NDArray[np.uint8],
    deformed: NDArray[np.uint8],
    flow: NDArray[np.float32],
    evm: FloatArray,
    autocorrelation: list[dict[str, float]],
) -> None:
    stride = 4
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].imshow(reference[::stride, ::stride], cmap="gray")
    axes[0, 0].set_title("Frame 000334 (candidate final state)")
    axes[0, 1].imshow(deformed[::stride, ::stride], cmap="gray")
    axes[0, 1].set_title("Frame 000335 (candidate repeat)")
    magnitude = np.linalg.norm(flow, axis=-1)
    image = axes[0, 2].imshow(
        magnitude[::stride, ::stride],
        cmap="viridis",
        vmin=0.0,
        vmax=float(np.quantile(magnitude, 0.995)),
    )
    axes[0, 2].set_title("Recovered displacement magnitude (px)")
    figure.colorbar(image, ax=axes[0, 2])
    evm_image = axes[1, 0].imshow(
        evm.T,
        cmap="magma",
        vmin=0.0,
        vmax=float(np.quantile(evm, 0.995)),
        origin="upper",
    )
    axes[1, 0].set_title("Spurious total equivalent strain, EVM")
    figure.colorbar(evm_image, ax=axes[1, 0])
    axes[1, 1].hist(evm.ravel(), bins=100, color="0.25")
    axes[1, 1].set_xlabel("EVM")
    axes[1, 1].set_ylabel("Count")
    radii = [row["radius_pixels"] for row in autocorrelation]
    correlations = [row["correlation"] for row in autocorrelation]
    axes[1, 2].plot(radii, correlations)
    axes[1, 2].axhline(1.0 / np.e, color="black", linestyle="--", linewidth=1)
    axes[1, 2].set_xlabel("Radius (px)")
    axes[1, 2].set_ylabel("Autocorrelation")
    for axis in axes.flat[:4]:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("DISFlow reproduction chain — null test")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _transfer_figures(
    figure_directory: Path,
    *,
    sinusoidal_rows: list[dict[str, Any]],
    band_rows: list[dict[str, Any]],
    band_evm_cases: list[dict[str, Any]],
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for orientation in ("horizontal", "vertical"):
        selected = [row for row in sinusoidal_rows if row["orientation"] == orientation]
        axis.plot(
            [row["wavelength_pixels"] for row in selected],
            [row["gain"] for row in selected],
            marker="o",
            label=orientation,
        )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Imposed wavelength (px)")
    axis.set_ylabel("Recovered displacement amplitude / imposed amplitude")
    axis.set_title("DISFlow modulation transfer")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(figure_directory / "transfer_function.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for orientation in ("horizontal", "vertical"):
        selected = [row for row in band_rows if row["orientation"] == orientation]
        axes[0].plot(
            [row["imposed_width_pixels"] for row in selected],
            [row["recovered_width_pixels"] for row in selected],
            marker="o",
            label=orientation,
        )
        axes[1].plot(
            [row["imposed_width_pixels"] for row in selected],
            [row["peak_gain"] for row in selected],
            marker="o",
            label=orientation,
        )
    axes[0].plot([0, 32], [0, 32], color="black", linestyle="--", label="ideal")
    axes[0].set_xlabel("Imposed FWHM (px)")
    axes[0].set_ylabel("Recovered FWHM (px)")
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set_xlabel("Imposed FWHM (px)")
    axes[1].set_ylabel("Recovered peak / imposed peak")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("DISFlow imposed-band fidelity")
    figure.savefig(figure_directory / "band_width_fidelity.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(
        len(band_evm_cases),
        3,
        figsize=(14, 3.25 * len(band_evm_cases)),
        constrained_layout=True,
    )
    for row_index, case in enumerate(band_evm_cases):
        imposed_map = np.asarray(case["imposed_map"], dtype=np.float64)
        recovered_map = np.asarray(case["recovered_map"], dtype=np.float64)
        coordinate_um = np.asarray(case["coordinate_um"], dtype=np.float64)
        imposed_profile = np.asarray(case["imposed_profile"], dtype=np.float64)
        recovered_profile = np.asarray(case["recovered_profile"], dtype=np.float64)
        width_pixels = int(case["width_pixels"])
        width_um = width_pixels * PIXEL_SIZE_UM
        common_maximum = float(max(np.max(imposed_map), np.max(recovered_map)))
        half_extent_um = float(case["map_half_extent_pixels"]) * PIXEL_SIZE_UM
        extent = (-half_extent_um, half_extent_um, -half_extent_um, half_extent_um)

        imposed_image = axes[row_index, 0].imshow(
            imposed_map.T,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=common_maximum,
            aspect="equal",
        )
        axes[row_index, 0].axhline(0.0, color="cyan", linewidth=1.2)
        axes[row_index, 0].set_title(
            f"Imposed EVM — FWHM {width_pixels} px ({width_um:.2f} µm)"
        )
        axes[row_index, 1].imshow(
            recovered_map.T,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=common_maximum,
            aspect="equal",
        )
        axes[row_index, 1].axhline(0.0, color="cyan", linewidth=1.2)
        axes[row_index, 1].set_title(
            "DISFlow-recovered EVM\n"
            f"FWHM {case['recovered_width_pixels']:.0f} px"
        )
        figure.colorbar(
            imposed_image,
            ax=(axes[row_index, 0], axes[row_index, 1]),
            label="Total equivalent strain, EVM",
            shrink=0.88,
        )

        ideal_step = np.where(
            np.abs(coordinate_um) <= 0.5 * width_um,
            float(np.max(imposed_profile)),
            0.0,
        )
        axes[row_index, 2].plot(
            coordinate_um,
            imposed_profile,
            color="black",
            linewidth=1.8,
            label="Exact imposed Gaussian EVM",
        )
        axes[row_index, 2].plot(
            coordinate_um,
            recovered_profile,
            color="#d95f02",
            linewidth=1.8,
            label="Recovered EVM",
        )
        axes[row_index, 2].step(
            coordinate_um,
            ideal_step,
            where="mid",
            color="#1b9e77",
            linestyle="--",
            linewidth=1.5,
            label="FWHM reference step",
        )
        axes[row_index, 2].axvspan(
            -0.5 * width_um,
            0.5 * width_um,
            color="#1b9e77",
            alpha=0.08,
        )
        axes[row_index, 2].set_xlim(-4.0 * width_um, 4.0 * width_um)
        axes[row_index, 2].set_xlabel("Coordinate normal to the band (µm)")
        axes[row_index, 2].set_ylabel("Total equivalent strain, EVM")
        axes[row_index, 2].grid(alpha=0.25)
        axes[row_index, 2].legend(fontsize=8, loc="upper right")
        secondary = axes[row_index, 2].secondary_xaxis(
            "top",
            functions=(
                lambda values: values / PIXEL_SIZE_UM,
                lambda values: values * PIXEL_SIZE_UM,
            ),
        )
        secondary.set_xlabel("Coordinate normal to the band (px)")

        for axis in axes[row_index, :2]:
            axis.set_xlabel("Normal coordinate (µm)")
            axis.set_ylabel("Along-band coordinate (µm)")

    figure.suptitle(
        "Synthetic strain bands at native DISFlow scale 0\n"
        "Cyan line: section normal to the band; maps in each row share one scale"
    )
    figure.savefig(figure_directory / "synthetic_band_evm_sections.png", dpi=180)
    plt.close(figure)


def characterise_dic_measurement_chain(
    *,
    image_directory: str | Path,
    prepared_case: str | Path,
    output_directory: str | Path,
    figure_directory: str | Path,
    config: DISFlowConfig | None = None,
    profile_name: str = "declared_medium_v4",
    warp_mode: WarpMode = "legacy_approximate_inverse",
    overwrite: bool = False,
    run_transfer: bool = True,
) -> dict[str, Any]:
    """Run the pre-registered null and synthetic transfer diagnostics."""

    source = Path(image_directory)
    prepared = Path(prepared_case)
    output = Path(output_directory)
    figures = Path(figure_directory)
    _prepare_directory(output, overwrite=overwrite)
    _prepare_directory(figures, overwrite=overwrite)
    selected = DISFlowConfig() if config is None else config
    reference_path = source / "000294.tif"
    final_path = source / "000334.tif"
    repeat_path = source / "000335.tif"
    reference_full = _load_tiff(reference_path)
    final_full = _load_tiff(final_path)
    repeat_full = _load_tiff(repeat_path)
    reference = np.ascontiguousarray(reference_full[CROP_ROWS, CROP_COLUMNS])
    final = np.ascontiguousarray(final_full[CROP_ROWS, CROP_COLUMNS])
    repeat = np.ascontiguousarray(repeat_full[CROP_ROWS, CROP_COLUMNS])

    flow = run_disflow(final, repeat, config=selected)
    evm = image_flow_to_historical_evm(flow)
    autocorrelation, correlation_length = radial_autocorrelation(evm)
    residual = _photometric_residual(final, repeat, flow)
    dic_evm = _prepared_dic_evm(prepared)
    dic_rms = float(np.sqrt(np.mean(np.square(dic_evm))))
    evm_rms = float(np.sqrt(np.mean(np.square(evm))))
    ratio = evm_rms / dic_rms
    interpretation = "small"
    if ratio >= 0.3:
        interpretation = "materially_limits_amplitude_claims"
    elif ratio >= 0.1:
        interpretation = "non_negligible"

    null_report: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed_characterisation_no_pass_fail",
        "frame_mapping_status": "provisional_without_acquisition_log",
        "reference_frame": "000294.tif",
        "final_frame": "000334.tif",
        "repeat_frame": "000335.tif",
        "crop_rows": [CROP_ROWS.start, CROP_ROWS.stop],
        "crop_columns": [CROP_COLUMNS.start, CROP_COLUMNS.stop],
        "flow_column_pixels": _statistics(flow[..., 0]),
        "flow_row_pixels": _statistics(flow[..., 1]),
        "flow_column_centered_pixels": _statistics(flow[..., 0] - np.mean(flow[..., 0])),
        "flow_row_centered_pixels": _statistics(flow[..., 1] - np.mean(flow[..., 1])),
        "spurious_historical_evm": _statistics(evm),
        "step40_dic_historical_evm": {
            "rms": dic_rms,
            "support": list(dic_evm.shape),
            "operator": "reconstruct_historical_evm",
        },
        "spurious_to_step40_rms_ratio": ratio,
        "pre_registered_interpretation": interpretation,
        "radial_autocorrelation_first_one_over_e_pixels": correlation_length,
        "radial_autocorrelation_first_one_over_e_um": (
            None if correlation_length is None else correlation_length * PIXEL_SIZE_UM
        ),
        "photometric_residual_grey_levels": _statistics(residual),
        "evm_post_filter_applied": False,
    }
    _json(output / "null_test_report.json", null_report)
    _csv(output / "null_autocorrelation.csv", autocorrelation)
    _null_figure(
        figures / "null_test.png",
        reference=final,
        deformed=repeat,
        flow=flow,
        evm=evm,
        autocorrelation=autocorrelation,
    )

    transfer_report: dict[str, Any] = {"status": "not_requested"}
    if run_transfer:
        window = _central_window(reference)
        if selected.patch_size is None:
            raise ValueError("synthetic transfer requires an explicit patch_size")
        border = 2 * selected.patch_size
        sinusoidal_rows: list[dict[str, Any]] = []
        for orientation in ("horizontal", "vertical"):
            component = 0 if orientation == "horizontal" else 1
            for wavelength in SINUSOIDAL_WAVELENGTHS:
                imposed = _sinusoidal_displacement(
                    window.shape,
                    wavelength_pixels=wavelength,
                    orientation=orientation,
                )
                warped = warp_image(
                    window,
                    imposed,
                    mode=warp_mode,
                )
                recovered = run_disflow(window, warped, config=selected)
                amplitude, phase = _fit_sinusoid(
                    recovered[..., component],
                    wavelength_pixels=wavelength,
                    orientation=orientation,
                    border=border,
                )
                sinusoidal_rows.append(
                    {
                        "orientation": orientation,
                        "wavelength_pixels": wavelength,
                        "wavelength_um": wavelength * PIXEL_SIZE_UM,
                        "imposed_amplitude_pixels": 0.5,
                        "recovered_amplitude_pixels": amplitude,
                        "gain": amplitude / 0.5,
                        "phase_error_radians": phase,
                    }
                )
        band_rows: list[dict[str, Any]] = []
        band_evm_cases: list[dict[str, Any]] = []
        for orientation in ("horizontal", "vertical"):
            component = 0 if orientation == "horizontal" else 1
            profile_axis = 0 if orientation == "horizontal" else 1
            gradient_axis = 1 if orientation == "horizontal" else 0
            for width in BAND_WIDTHS:
                imposed, imposed_gradient = _band_displacement(
                    window.shape,
                    width_pixels=width,
                    orientation=orientation,
                )
                warped = warp_image(
                    window,
                    imposed,
                    mode=warp_mode,
                )
                recovered = run_disflow(window, warped, config=selected)
                recovered_gradient = np.gradient(
                    recovered[..., component],
                    axis=gradient_axis,
                )
                central = recovered_gradient[border:-border, border:-border]
                profile = np.asarray(np.median(central, axis=profile_axis), dtype=np.float64)
                imposed_core = imposed_gradient[border:-border]
                recovered_metrology = profile_metrology(profile)
                imposed_metrology = profile_metrology(imposed_core)
                recovered_width = recovered_metrology.subpixel_fwhm_pixels
                recovered_width_value = (
                    float("nan") if recovered_width is None else recovered_width
                )
                centroid_shift = (
                    None
                    if recovered_metrology.centroid_index_pixels is None
                    or imposed_metrology.centroid_index_pixels is None
                    else recovered_metrology.centroid_index_pixels
                    - imposed_metrology.centroid_index_pixels
                )
                band_rows.append(
                    {
                        "orientation": orientation,
                        "imposed_width_pixels": width,
                        "imposed_width_um": width * PIXEL_SIZE_UM,
                        "recovered_width_pixels": recovered_width_value,
                        "recovered_width_um": recovered_width_value * PIXEL_SIZE_UM,
                        "relative_width_error": recovered_width_value / width - 1.0,
                        "legacy_integer_fwhm_pixels": (
                            recovered_metrology.legacy_integer_fwhm_pixels
                        ),
                        "fwhm_status": recovered_metrology.fwhm_status,
                        "peak_gain": float(np.max(profile) / np.max(imposed_core)),
                        "peak_shift_pixels": (
                            recovered_metrology.peak_index_pixels
                            - imposed_metrology.peak_index_pixels
                        ),
                        "centroid_shift_pixels": centroid_shift,
                    }
                )
                if orientation == "horizontal":
                    imposed_evm = image_flow_to_historical_evm(imposed)
                    recovered_evm = image_flow_to_historical_evm(recovered)
                    imposed_profile = np.median(imposed_evm, axis=1)
                    recovered_evm_profile = np.median(recovered_evm, axis=1)
                    centre_index = int(np.argmax(imposed_profile))
                    coordinate_um = (
                        np.arange(imposed_profile.size, dtype=np.float64) - centre_index
                    ) * PIXEL_SIZE_UM
                    map_half_extent = 96
                    along_centre = imposed_evm.shape[1] // 2
                    normal_slice = slice(
                        centre_index - map_half_extent,
                        centre_index + map_half_extent,
                    )
                    along_slice = slice(
                        along_centre - map_half_extent,
                        along_centre + map_half_extent,
                    )
                    band_evm_cases.append(
                        {
                            "width_pixels": width,
                            "recovered_width_pixels": recovered_width_value,
                            "coordinate_um": coordinate_um,
                            "imposed_profile": imposed_profile,
                            "recovered_profile": recovered_evm_profile,
                            "imposed_map": imposed_evm[normal_slice, along_slice],
                            "recovered_map": recovered_evm[normal_slice, along_slice],
                            "map_half_extent_pixels": map_half_extent,
                        }
                    )
        _csv(output / "sinusoidal_transfer.csv", sinusoidal_rows)
        _csv(output / "band_width_fidelity.csv", band_rows)
        transfer_report = {
            "schema_version": 1,
            "status": "completed_algorithmic_transfer_characterisation",
            "window_shape": [TRANSFER_SIZE, TRANSFER_SIZE],
            "window_selection": "fixed_central_window_of_registered_crop",
            "border_excluded_pixels": border,
            "sinusoidal_wavelengths_pixels": list(SINUSOIDAL_WAVELENGTHS),
            "band_fwhm_pixels": list(BAND_WIDTHS),
            "mtf50_wavelength_pixels": {
                orientation: _mtf50(sinusoidal_rows, orientation)
                for orientation in ("horizontal", "vertical")
            },
            "synthetic_transfer_only": True,
            "experimental_artifacts_included": False,
            "evm_post_filter_applied": False,
            "sinusoidal_rows": sinusoidal_rows,
            "band_rows": band_rows,
        }
        _json(output / "transfer_report.json", transfer_report)
        _transfer_figures(
            figures,
            sinusoidal_rows=sinusoidal_rows,
            band_rows=band_rows,
            band_evm_cases=band_evm_cases,
        )

    cv2 = _cv2()
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "source_images": {
            path.name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in (reference_path, final_path, repeat_path)
        },
        "prepared_case": str(prepared.resolve()),
        "disflow_requested": selected.as_dict(),
        "disflow_queried": queried_disflow_configuration(selected),
        "disflow_profile": profile_name,
        "warp_mode": warp_mode,
        "historical_identity": {
            "status": "reproduction_not_bitwise_historical",
            "reported_parameters_applied": True,
            "missing_historical_opencv_version_and_remaining_defaults": True,
        },
        "pixel_size_um": PIXEL_SIZE_UM,
        "outputs": {
            path.name: _sha256(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
        "figures": {
            path.name: _sha256(path)
            for path in sorted(figures.iterdir())
            if path.is_file()
        },
    }
    _json(output / "manifest.json", manifest)
    return {
        "status": "completed",
        "manifest": str(output / "manifest.json"),
        "null_test": null_report,
        "transfer": transfer_report,
    }
