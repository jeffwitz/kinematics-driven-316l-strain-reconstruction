"""Measure temporal noise and the loading subspace of a measured DIC boundary history.

The states of a measured history are independent direct correlations of one
reference image onto each deformed image. Measurement noise is therefore
independent in time while the physical boundary path is smooth in time. This
module exploits that asymmetry only; it makes no material assumption, runs no
mechanics and never modifies the immutable history.

Stage 0 of ``validation/dic_boundary_temporal_regularisation_preregistration.md``.
"""

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

from fem_inhouse.workflows.dic_boundary_history import (
    affine_boundary_decomposition,
)
from fem_inhouse.workflows.dic_observation_replay import PIXEL_SIZE_MM

FloatArray = NDArray[np.float64]
BooleanArray = NDArray[np.bool_]

#: Second differences of an independent-noise series have six times its variance.
SECOND_DIFFERENCE_VARIANCE_FACTOR = 6.0

#: Scale turning a median absolute deviation into a Gaussian standard deviation.
MAD_TO_SIGMA = 1.4826

#: A mode is retained as loading signal below this temporal roughness.
SIGNAL_ROUGHNESS_THRESHOLD = 0.5

#: Retained modes must carry at least this fraction of boundary displacement energy.
SIGNAL_ENERGY_FLOOR = 0.90


@dataclass(frozen=True, slots=True)
class TemporalNoiseEstimate:
    """Upper bounds on the per-state measurement noise of a boundary history."""

    rms_mm: float
    robust_mm: float
    rms_px: float
    robust_px: float


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


def robust_sigma(values: NDArray[np.generic]) -> float:
    """Return the MAD-based standard deviation of ``values``."""

    sample = np.asarray(values, dtype=np.float64).ravel()
    if sample.size == 0 or not np.isfinite(sample).all():
        raise ValueError("values must be finite and non-empty")
    deviation = np.abs(sample - np.median(sample))
    return float(MAD_TO_SIGMA * np.median(deviation))


def second_time_difference(series: NDArray[np.generic]) -> FloatArray:
    """Return ``x[k + 1] - 2 x[k] + x[k - 1]`` along the leading axis."""

    values = np.asarray(series, dtype=np.float64)
    if values.shape[0] < 3:
        raise ValueError("a second temporal difference needs at least three states")
    return np.asarray(values[2:] - 2.0 * values[1:-1] + values[:-2], dtype=np.float64)


def temporal_noise_estimate(
    boundary_history: NDArray[np.generic],
    *,
    pixel_size_mm: float = PIXEL_SIZE_MM,
    selector: NDArray[np.bool_] | None = None,
) -> TemporalNoiseEstimate:
    """Bound the per-state noise of a boundary history from its temporal roughness.

    Both estimators are upper bounds: genuine temporal curvature of the loading
    path inflates them. The robust estimator is preferred because a single
    displaced state inflates the RMS one.
    """

    difference = second_time_difference(boundary_history)
    if selector is not None:
        difference = difference[np.asarray(selector, dtype=bool)]
        if difference.size == 0:
            raise ValueError("selector removed every second difference")
    scale = float(np.sqrt(SECOND_DIFFERENCE_VARIANCE_FACTOR))
    rms = float(np.sqrt(np.mean(np.square(difference)))) / scale
    robust = robust_sigma(difference) / scale
    return TemporalNoiseEstimate(
        rms_mm=rms,
        robust_mm=robust,
        rms_px=rms / pixel_size_mm,
        robust_px=robust / pixel_size_mm,
    )


def temporal_roughness(coefficients: NDArray[np.generic]) -> float:
    """Return the temporal roughness of one modal coefficient series.

    The ratio is near one for a temporally white series and much smaller for a
    smooth one, which is what separates measurement noise from loading signal
    when the noise is known to be independent in time.
    """

    series = np.asarray(coefficients, dtype=np.float64).ravel()
    magnitude = float(np.sqrt(np.mean(np.square(series))))
    if magnitude == 0.0:
        return 0.0
    difference = second_time_difference(series)
    rough = float(np.sqrt(np.mean(np.square(difference))))
    return rough / (float(np.sqrt(SECOND_DIFFERENCE_VARIANCE_FACTOR)) * magnitude)


def boundary_mask(shape: tuple[int, int]) -> BooleanArray:
    """Return the boundary nodes of a structured nodal support."""

    if min(shape) < 2:
        raise ValueError("a boundary field needs at least two nodes per axis")
    mask = np.zeros(shape, dtype=bool)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    return mask


def robust_z_scores(
    values: NDArray[np.generic],
    *,
    scale_mask: NDArray[np.bool_] | None = None,
) -> FloatArray:
    """Return MAD-based z-scores of ``values``.

    ``scale_mask`` selects the entries used to estimate the centre and spread.
    Every entry is still scored, which lets an excluded state be reported
    without letting it define the scale.
    """

    sample = np.asarray(values, dtype=np.float64).ravel()
    reference = sample if scale_mask is None else sample[np.asarray(scale_mask, dtype=bool)]
    spread = robust_sigma(reference)
    if spread == 0.0:
        return np.zeros_like(sample)
    return np.asarray((sample - np.median(reference)) / spread, dtype=np.float64)


def _affine_strain(coefficients: FloatArray) -> FloatArray:
    return np.asarray(
        [
            coefficients[1, 0],
            coefficients[2, 1],
            coefficients[2, 0] + coefficients[1, 1],
        ],
        dtype=np.float64,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty table")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _figure(
    path: Path,
    *,
    singular_values: FloatArray,
    roughness: FloatArray,
    loading: FloatArray,
    cumulative_strain: FloatArray,
    increment_strain: FloatArray,
    strain_noise: float,
    loading_z: FloatArray,
    strain_z: FloatArray,
    signal_to_noise: FloatArray,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16.5, 8.5))
    modes = np.arange(1, singular_values.size + 1)

    axes[0, 0].semilogy(modes, singular_values, "o-", markersize=3)
    axes[0, 0].set_title("Boundary singular spectrum")
    axes[0, 0].set_xlabel("mode")
    axes[0, 0].set_ylabel("singular value (mm)")

    axes[0, 1].plot(modes, roughness, "o-", markersize=3)
    axes[0, 1].axhline(SIGNAL_ROUGHNESS_THRESHOLD, color="crimson", linestyle="--")
    axes[0, 1].axhline(1.0, color="grey", linestyle=":")
    axes[0, 1].set_title("Temporal roughness per mode")
    axes[0, 1].set_xlabel("mode")
    axes[0, 1].set_ylabel("R (1 = temporally white)")

    states = np.arange(loading.size)
    axes[0, 2].plot(states, loading, "o-", markersize=3)
    axes[0, 2].set_title("Dominant loading coefficient")
    axes[0, 2].set_xlabel("state")
    axes[0, 2].set_ylabel("phi (mm)")

    increments = np.arange(1, increment_strain.size + 1)
    axes[1, 0].plot(increments, increment_strain, "o-", markersize=3, label="measured")
    axes[1, 0].fill_between(
        increments,
        increment_strain - np.sqrt(2.0) * strain_noise,
        increment_strain + np.sqrt(2.0) * strain_noise,
        color="grey",
        alpha=0.3,
        label="noise band",
    )
    axes[1, 0].set_title("Affine transverse strain increment")
    axes[1, 0].set_xlabel("state")
    axes[1, 0].set_ylabel("d eps_xx")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(np.arange(1, loading_z.size + 1), np.abs(loading_z), "o-", markersize=3,
                    label="loading coefficient")
    axes[1, 1].plot(np.arange(1, strain_z.size + 1), np.abs(strain_z), "s-", markersize=3,
                    label="affine strain")
    axes[1, 1].axhline(3.0, color="crimson", linestyle="--")
    axes[1, 1].set_title("Robust outlier score")
    axes[1, 1].set_xlabel("state")
    axes[1, 1].set_ylabel("|z| of second temporal difference")
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].semilogy(increments, np.maximum(signal_to_noise, 1e-3), "o-", markersize=3)
    axes[1, 2].axhline(1.0, color="crimson", linestyle="--")
    axes[1, 2].set_title("Per-increment signal-to-noise")
    axes[1, 2].set_xlabel("state")
    axes[1, 2].set_ylabel("|d eps_xx| / (sqrt(2) sigma_eps)")

    for axis in axes.ravel():
        axis.grid(alpha=0.3)
    figure.suptitle("P43 measured boundary history: temporal noise and loading subspace")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    _ = cumulative_strain


def diagnose_dic_boundary_loading_subspace(
    *,
    history_path: str | Path,
    history_report_path: str | Path,
    output_directory: str | Path,
    figure_directory: str | Path,
    pixel_size_mm: float = PIXEL_SIZE_MM,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Measure the noise and loading subspace of an immutable DIC boundary history."""

    history_source = Path(history_path)
    history_report_source = Path(history_report_path)
    output = Path(output_directory)
    figures = Path(figure_directory)
    _prepare_directory(output, overwrite=overwrite)
    _prepare_directory(figures, overwrite=overwrite)

    history_report = json.loads(history_report_source.read_text(encoding="utf-8"))
    expected_hash = history_report["outputs"][history_source.name]
    if _sha256(history_source) != expected_hash:
        raise ValueError("history does not match its immutable report")

    history = np.load(history_source, mmap_mode="r", allow_pickle=False)
    if history.ndim != 4 or history.shape[-1] != 2 or history.shape[0] < 3:
        raise ValueError("history must have shape (states, nx + 1, ny + 1, 2)")
    states = int(history.shape[0])
    shape = (int(history.shape[1]), int(history.shape[2]))
    mask = boundary_mask(shape)
    boundary_nodes = int(np.count_nonzero(mask))

    boundary = np.stack(
        [np.asarray(history[state], dtype=np.float64)[mask].ravel() for state in range(states)]
    )
    if not np.isfinite(boundary).all():
        raise ValueError("boundary history must be finite")

    # A second difference centred on a linearly interpolated repaired state is
    # zero by construction, so it is not a noise realisation. The excluded list
    # is read from the immutable repair report, never from a magnitude cutoff.
    repaired_states = [
        int(state) for state in history_report.get("repair", {}).get("corrupted_states", [])
    ]
    centred_states = np.arange(1, states - 1)
    noise_selector = ~np.isin(centred_states, repaired_states)
    if not noise_selector.any():
        raise ValueError("every centred state was excluded from the noise estimate")

    noise = temporal_noise_estimate(
        boundary, pixel_size_mm=pixel_size_mm, selector=noise_selector
    )

    affine_strain = np.stack(
        [
            _affine_strain(
                affine_boundary_decomposition(
                    np.asarray(history[state], dtype=np.float64),
                    spacing_x_mm=pixel_size_mm,
                    spacing_y_mm=pixel_size_mm,
                ).coefficients
            )
            for state in range(states)
        ]
    )
    increment_strain = np.diff(affine_strain, axis=0)

    # The second temporal differences are noise realisations carrying the true
    # spatial correlation of the measurement, so propagating them through the
    # same affine fit needs no spatial independence assumption.
    scale = float(np.sqrt(SECOND_DIFFERENCE_VARIANCE_FACTOR))
    noise_decompositions = [
        affine_boundary_decomposition(
            (
                np.asarray(history[state + 1], dtype=np.float64)
                - 2.0 * np.asarray(history[state], dtype=np.float64)
                + np.asarray(history[state - 1], dtype=np.float64)
            )
            / scale,
            spacing_x_mm=pixel_size_mm,
            spacing_y_mm=pixel_size_mm,
        )
        for state in centred_states[noise_selector]
    ]
    noise_strain = np.stack(
        [_affine_strain(item.coefficients) for item in noise_decompositions]
    )
    # How much of the measurement noise the affine fit absorbs decides whether
    # stage 1 can regularise a handful of coefficients instead of every node.
    noise_nonaffine_fraction = np.asarray(
        [item.residual_rms_mm / item.total_rms_mm for item in noise_decompositions]
    )
    affine_strain_sigma = np.asarray(
        [robust_sigma(noise_strain[:, component]) for component in range(3)]
    )

    left, spectrum_raw, _ = np.linalg.svd(boundary, full_matrices=False)
    spectrum = np.asarray(spectrum_raw, dtype=np.float64)
    singular_values = spectrum
    modal = left * spectrum
    roughness = np.asarray([temporal_roughness(modal[:, mode]) for mode in range(states)])
    energy = np.square(spectrum) / float(np.sum(np.square(spectrum)))
    signal_modes = np.flatnonzero(roughness < SIGNAL_ROUGHNESS_THRESHOLD)
    retained_energy = float(np.sum(energy[signal_modes])) if signal_modes.size else 0.0

    loading = np.asarray(modal[:, 0], dtype=np.float64)
    loading_z = robust_z_scores(second_time_difference(loading), scale_mask=noise_selector)
    strain_z = robust_z_scores(
        second_time_difference(affine_strain[:, 0]), scale_mask=noise_selector
    )
    increment_noise = float(np.sqrt(2.0) * affine_strain_sigma[0])
    signal_to_noise = np.abs(increment_strain[:, 0]) / increment_noise

    rows: list[dict[str, Any]] = []
    for state in range(1, states):
        index = state - 1
        rows.append(
            {
                "state": state,
                "affine_epsilon_xx": float(affine_strain[state, 0]),
                "affine_epsilon_yy": float(affine_strain[state, 1]),
                "affine_gamma_xy": float(affine_strain[state, 2]),
                "increment_epsilon_xx": float(increment_strain[index, 0]),
                "increment_epsilon_yy": float(increment_strain[index, 1]),
                "increment_gamma_xy": float(increment_strain[index, 2]),
                "loading_coefficient_mm": float(loading[state]),
                # d2[j] is centred on state j + 1, so state k reads index k - 1
                # and the last state has no centred second difference.
                "loading_second_difference_z": (
                    float(loading_z[index]) if state <= states - 2 else ""
                ),
                "affine_strain_second_difference_z": (
                    float(strain_z[index]) if state <= states - 2 else ""
                ),
                "signal_to_noise_epsilon_xx": float(signal_to_noise[index]),
            }
        )
    _write_csv(output / "state_metrics.csv", rows)

    figure_name = "p0043_boundary_loading_subspace.png"
    _figure(
        output / figure_name,
        singular_values=singular_values,
        roughness=roughness,
        loading=loading,
        cumulative_strain=affine_strain[:, 0],
        increment_strain=increment_strain[:, 0],
        strain_noise=float(affine_strain_sigma[0]),
        loading_z=loading_z,
        strain_z=strain_z,
        signal_to_noise=signal_to_noise,
    )
    figure_copy = figures / figure_name
    figure_copy.write_bytes((output / figure_name).read_bytes())

    peak_loading = int(np.argmax(np.abs(loading_z))) + 1
    peak_strain = int(np.argmax(np.abs(strain_z))) + 1
    state4_scores = (float(abs(loading_z[3])), float(abs(strain_z[3])))
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed_stage0_boundary_noise_and_loading_subspace",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "partition_id": int(history_report["partition_id"]),
        "preregistration": (
            "validation/dic_boundary_temporal_regularisation_preregistration.md"
        ),
        "source": {
            "history": str(history_source.resolve()),
            "history_sha256": expected_hash,
            "history_report": str(history_report_source.resolve()),
            "history_report_sha256": _sha256(history_report_source),
        },
        "support": {
            "axis_convention": "canonical array axes (x,y), components (ux,uy)",
            "node_shape": list(shape),
            "boundary_nodes": boundary_nodes,
            "states": states,
            "pixel_size_mm": pixel_size_mm,
            "solve_bounds": list(history_report["solve_bounds"]),
            "core_bounds": list(history_report["core_bounds"]),
        },
        "temporal_noise": {
            "estimator": "second temporal difference, Var(d2) = 6 sigma^2",
            "claim_boundary": "upper bound; real temporal curvature inflates it",
            "rms_mm": noise.rms_mm,
            "rms_px": noise.rms_px,
            "robust_mm": noise.robust_mm,
            "robust_px": noise.robust_px,
            "archived_repeated_frame_sigma_px": 0.06283,
        },
        "affine_noise": {
            "method": "affine fit of second-difference noise realisations",
            "epsilon_xx_sigma": float(affine_strain_sigma[0]),
            "epsilon_yy_sigma": float(affine_strain_sigma[1]),
            "gamma_xy_sigma": float(affine_strain_sigma[2]),
            "increment_epsilon_xx_sigma": increment_noise,
            "median_nonaffine_fraction": float(np.median(noise_nonaffine_fraction)),
            "maximum_nonaffine_fraction": float(np.max(noise_nonaffine_fraction)),
            "excluded_repaired_states": repaired_states,
            "noise_realisations": int(np.count_nonzero(noise_selector)),
        },
        "loading_subspace": {
            "roughness_threshold": SIGNAL_ROUGHNESS_THRESHOLD,
            "energy_floor": SIGNAL_ENERGY_FLOOR,
            "signal_mode_count": int(signal_modes.size),
            "signal_mode_indices": [int(mode) + 1 for mode in signal_modes],
            "retained_energy_fraction": retained_energy,
            "leading_roughness": [float(value) for value in roughness[:6]],
            "leading_energy_fraction": [float(value) for value in energy[:6]],
            "low_dimensional_model_supported": bool(
                signal_modes.size > 0 and retained_energy >= SIGNAL_ENERGY_FLOOR
            ),
        },
        "outliers": {
            "registered_expectation": "state 4 keeps |z| >= 3 on at least one score",
            "state4_loading_z": state4_scores[0],
            "state4_affine_strain_z": state4_scores[1],
            "state4_is_registered_outlier": bool(max(state4_scores) >= 3.0),
            "largest_loading_z_state": peak_loading,
            "largest_affine_strain_z_state": peak_strain,
        },
        "signal_to_noise": {
            "minimum": float(np.min(signal_to_noise)),
            "minimum_state": int(np.argmin(signal_to_noise)) + 1,
            "median": float(np.median(signal_to_noise)),
            "maximum": float(np.max(signal_to_noise)),
            "increments_below_unity": int(np.count_nonzero(signal_to_noise < 1.0)),
        },
        "mechanics_rerun": False,
        "history_modified": False,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "outputs": {
            "state_metrics.csv": _sha256(output / "state_metrics.csv"),
            figure_name: _sha256(output / figure_name),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
