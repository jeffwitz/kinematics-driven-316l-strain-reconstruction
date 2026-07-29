"""Prepare direct-reference DIC histories and run measured-boundary diagnostics."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap
from numpy.typing import NDArray
from PIL import Image

from fem_inhouse.config import (
    CaseStudyConfig,
    MaterialConfig,
    MeshConfig,
    NonlocalPlasticityConfig,
    SolverConfig,
)
from fem_inhouse.core.nonlinear import NonlinearConvergenceError
from fem_inhouse.measurement import disflow_profile, image_flow_to_canonical, run_disflow
from fem_inhouse.solver import run_case_study
from fem_inhouse.workflows.campaign_access import (
    load_json_object,
    partition_from_manifest,
)
from fem_inhouse.workflows.dic_observation_replay import (
    PIXEL_SIZE_MM,
    RAW_CROP_COLUMN_START,
    RAW_CROP_ROW_START,
)
from fem_inhouse.workflows.nonlocality_diagnostic import reconstruct_historical_evm

FloatArray = NDArray[np.float64]
RAW_CROP_SHAPE = (3600, 3100)
CORRUPTED_MEASURED_STATES = (31, 32)
REPAIR_BRACKETING_STATES = (30, 33)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def anchor_displacement_history(
    raw_history_mm: NDArray[np.generic],
    prepared_final_mm: NDArray[np.generic],
) -> FloatArray:
    """Linearly anchor a direct-reference history to an immutable endpoint."""

    history = np.asarray(raw_history_mm, dtype=np.float64)
    final = np.asarray(prepared_final_mm, dtype=np.float64)
    if history.ndim != 4 or history.shape[-1] != 2:
        raise ValueError("raw_history_mm must have shape (steps + 1, nx, ny, 2)")
    if final.shape != history.shape[1:]:
        raise ValueError("prepared_final_mm must match one history state")
    if not np.isfinite(history).all() or not np.isfinite(final).all():
        raise ValueError("history and endpoint must contain finite values")
    if not np.allclose(history[0], 0.0, rtol=0.0, atol=1.0e-14):
        raise ValueError("raw history must start from zero")
    fractions = np.linspace(0.0, 1.0, history.shape[0], dtype=np.float64)
    correction = final - history[-1]
    return history + fractions[:, None, None, None] * correction


def repair_corrupted_displacement_states(
    history_mm: NDArray[np.generic],
    *,
    corrupted_states: tuple[int, ...] = CORRUPTED_MEASURED_STATES,
    bracketing_states: tuple[int, int] = REPAIR_BRACKETING_STATES,
) -> FloatArray:
    """Linearly interpolate pre-registered corrupted measured states."""

    history = np.asarray(history_mm, dtype=np.float64)
    if history.ndim != 4 or history.shape[-1] != 2:
        raise ValueError("history_mm must have shape (steps + 1, nx, ny, 2)")
    if not np.isfinite(history).all():
        raise ValueError("history_mm must contain finite values")
    before, after = bracketing_states
    if not 0 <= before < after < history.shape[0]:
        raise ValueError("bracketing states must be ordered and inside the history")
    if any(state <= before or state >= after for state in corrupted_states):
        raise ValueError("corrupted states must lie strictly between the brackets")
    repaired = history.copy()
    interval = float(after - before)
    for state in corrupted_states:
        weight = (state - before) / interval
        repaired[state] = (1.0 - weight) * history[before] + weight * history[after]
    return repaired


def _history_evm(
    history: FloatArray,
    *,
    spacing_mm: float,
    poisson_ratio: float,
) -> tuple[FloatArray, FloatArray]:
    state = np.stack(
        [
            reconstruct_historical_evm(
                displacement,
                spacing_x_mm=spacing_mm,
                spacing_y_mm=spacing_mm,
                poisson_ratio=poisson_ratio,
            )
            for displacement in history
        ]
    )
    increment = np.zeros_like(state)
    increment[1:] = np.stack(
        [
            reconstruct_historical_evm(
                history[index] - history[index - 1],
                spacing_x_mm=spacing_mm,
                spacing_y_mm=spacing_mm,
                poisson_ratio=poisson_ratio,
            )
            for index in range(1, history.shape[0])
        ]
    )
    return state, increment


def _corrupted_frame_figure(
    path: Path,
    *,
    original_state: FloatArray,
    repaired_state: FloatArray,
    original_increment: FloatArray,
    repaired_increment: FloatArray,
) -> None:
    states = range(29, 35)
    figure, axes = plt.subplots(4, 6, figsize=(15, 9), constrained_layout=True)
    rows = (
        ("Original state EVM", original_state),
        ("Repaired state EVM", repaired_state),
        ("Original increment EVM", original_increment),
        ("Repaired increment EVM", repaired_increment),
    )
    state_limit = float(np.max(original_state[29:35]))
    increment_limit = float(np.max(original_increment[29:35]))
    for row_index, (label, values) in enumerate(rows):
        limit = state_limit if row_index < 2 else increment_limit
        for column, state in enumerate(states):
            image = axes[row_index, column].imshow(
                values[state].T,
                origin="lower",
                vmin=0.0,
                vmax=limit,
                cmap="viridis",
                interpolation="nearest",
            )
            axes[row_index, column].set_title(f"state {state}")
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
        axes[row_index, 0].set_ylabel(label)
        figure.colorbar(image, ax=axes[row_index], shrink=0.72)
    figure.suptitle("P43 documented corrupted-frame repair (common scales by field type)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _history_figure(
    path: Path,
    *,
    raw: FloatArray,
    anchored: FloatArray,
    final: FloatArray,
) -> None:
    fractions = np.linspace(0.0, 1.0, raw.shape[0])
    proportional = fractions[:, None, None, None] * final
    raw_deviation = np.sqrt(np.mean(np.square(raw - proportional), axis=(1, 2, 3)))
    anchored_deviation = np.sqrt(np.mean(np.square(anchored - proportional), axis=(1, 2, 3)))
    endpoint_correction = np.sqrt(np.mean(np.square(anchored - raw), axis=(1, 2, 3)))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(fractions, raw_deviation * 1.0e3, label="direct OpenCV 4.14")
    axes[0].plot(fractions, anchored_deviation * 1.0e3, label="endpoint anchored")
    axes[0].set_xlabel("Ordered image fraction")
    axes[0].set_ylabel("RMS deviation from proportional path (um)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(fractions, endpoint_correction * 1.0e3)
    axes[1].set_xlabel("Ordered image fraction")
    axes[1].set_ylabel("RMS endpoint correction (um)")
    axes[1].grid(alpha=0.25)
    figure.suptitle("P43 direct-reference DIC boundary history")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def prepare_dic_multistep_history(
    *,
    image_directory: str | Path,
    prepared_case: str | Path,
    source_campaign: str | Path,
    partition_id: int,
    output_directory: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Reconstruct the 40 direct-reference P43 displacement states."""

    images = Path(image_directory)
    prepared = Path(prepared_case)
    campaign = Path(source_campaign)
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = campaign / "manifest.json"
    campaign_manifest = load_json_object(manifest_path)
    _layout, partition = partition_from_manifest(campaign_manifest, partition_id)
    sx0, sx1, sy0, sy1 = partition.solve_bounds
    support_shape = (sx1 - sx0 + 1, sy1 - sy0 + 1)
    history_shape = (41, *support_shape, 2)

    reference_path = images / "000294.tif"
    step_paths = [images / f"{index:06d}.tif" for index in range(295, 335)]
    for path in (reference_path, *step_paths):
        if not path.is_file():
            raise FileNotFoundError(f"missing DIC image: {path}")
    reference_full = np.asarray(Image.open(reference_path).convert("L"), dtype=np.uint8)
    crop = (
        slice(RAW_CROP_ROW_START, RAW_CROP_ROW_START + RAW_CROP_SHAPE[0]),
        slice(RAW_CROP_COLUMN_START, RAW_CROP_COLUMN_START + RAW_CROP_SHAPE[1]),
    )
    reference = np.ascontiguousarray(reference_full[crop])
    if reference.shape != RAW_CROP_SHAPE:
        raise ValueError("reference image does not contain the canonical crop")

    raw_path = output / "raw_direct_history_mm.npy"
    raw = open_memmap(raw_path, mode="w+", dtype=np.float32, shape=history_shape)
    raw[0] = 0.0
    profile = disflow_profile("legacy_script_2021")
    image_hashes = {reference_path.name: _sha256(reference_path)}
    for step, path in enumerate(step_paths, start=1):
        current_full = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        current = np.ascontiguousarray(current_full[crop])
        if current.shape != reference.shape:
            raise ValueError(f"unexpected image crop shape: {path}")
        flow = run_disflow(reference, current, config=profile.config)
        displacement = image_flow_to_canonical(flow, pixel_size_mm=PIXEL_SIZE_MM)
        raw[step] = displacement[sx0 : sx1 + 1, sy0 : sy1 + 1]
        raw.flush()
        image_hashes[path.name] = _sha256(path)
        (output / "progress.json").write_text(
            json.dumps(
                {"completed_steps": step, "last_image": path.name},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    raw_values = np.asarray(raw, dtype=np.float64)

    ux = np.load(prepared / "displacement_x_mm.npy", mmap_mode="r", allow_pickle=False)
    uy = np.load(prepared / "displacement_y_mm.npy", mmap_mode="r", allow_pickle=False)
    prepared_final = np.stack(
        (
            np.asarray(ux[sx0 : sx1 + 1, sy0 : sy1 + 1], dtype=np.float64),
            np.asarray(uy[sx0 : sx1 + 1, sy0 : sy1 + 1], dtype=np.float64),
        ),
        axis=-1,
    )
    anchored = anchor_displacement_history(raw_values, prepared_final)
    anchored_path = output / "anchored_history_mm.npy"
    correction_path = output / "endpoint_correction_mm.npy"
    np.save(anchored_path, np.asarray(anchored, dtype=np.float32))
    np.save(
        correction_path,
        np.asarray(prepared_final - raw_values[-1], dtype=np.float32),
    )
    endpoint_difference = raw_values[-1] - prepared_final
    fractions = np.linspace(0.0, 1.0, anchored.shape[0])
    proportional = fractions[:, None, None, None] * prepared_final
    history_deviation = anchored - proportional
    figure_path = output / "boundary_history_diagnostic.png"
    _history_figure(
        figure_path,
        raw=raw_values,
        anchored=anchored,
        final=prepared_final,
    )
    report = {
        "schema_version": 1,
        "status": "completed_direct_reference_history_endpoint_anchored",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "partition_id": partition_id,
        "solve_bounds": list(partition.solve_bounds),
        "core_bounds": list(partition.core_bounds),
        "step_count": 40,
        "ordered_pseudo_time_only": True,
        "load_cell_synchronisation_available": False,
        "mask": {
            "mode": "declared_all_valid",
            "historical_mask_reproduced": False,
        },
        "profile": profile.manifest(),
        "pixel_size_mm": PIXEL_SIZE_MM,
        "source": {
            "campaign_manifest": str(manifest_path.resolve()),
            "campaign_manifest_sha256": _sha256(manifest_path),
            "prepared_manifest_sha256": _sha256(prepared / "manifest.json"),
            "image_hashes": image_hashes,
        },
        "endpoint_compatibility": {
            "component_rms_difference_mm": float(np.sqrt(np.mean(np.square(endpoint_difference)))),
            "maximum_absolute_component_difference_mm": float(np.max(np.abs(endpoint_difference))),
            "relative_vector_norm": float(
                np.linalg.norm(endpoint_difference) / np.linalg.norm(prepared_final)
            ),
            "anchored_final_max_abs_difference_mm": float(
                np.max(np.abs(anchored[-1] - prepared_final))
            ),
        },
        "nonproportionality": {
            "maximum_rms_deviation_from_proportional_mm": float(
                np.max(np.sqrt(np.mean(np.square(history_deviation), axis=(1, 2, 3))))
            ),
            "final_rms_deviation_from_proportional_mm": float(
                np.sqrt(np.mean(np.square(history_deviation[-1])))
            ),
        },
        "outputs": {
            raw_path.name: _sha256(raw_path),
            anchored_path.name: _sha256(anchored_path),
            correction_path.name: _sha256(correction_path),
            figure_path.name: _sha256(figure_path),
        },
        "mechanics_rerun": False,
        "micromorphic_identification_run": False,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def repair_dic_multistep_history(
    *,
    history_directory: str | Path,
    source_campaign: str | Path,
    partition_id: int,
    output_directory: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Archive the pre-registered temporal repair and its strain diagnostics."""

    history_root = Path(history_directory)
    campaign = Path(source_campaign)
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_report_path = history_root / "report.json"
    source_report = load_json_object(source_report_path)
    if source_report.get("status") != "completed_direct_reference_history_endpoint_anchored":
        raise ValueError("source history campaign is not complete")
    if int(source_report["partition_id"]) != partition_id:
        raise ValueError("source history identifies another partition")
    source_path = history_root / "anchored_history_mm.npy"
    if _sha256(source_path) != source_report["outputs"][source_path.name]:
        raise ValueError("source history hash does not match its report")
    original = np.asarray(
        np.load(source_path, mmap_mode="r", allow_pickle=False),
        dtype=np.float64,
    )
    repaired = repair_corrupted_displacement_states(original)

    manifest_path = campaign / "manifest.json"
    manifest = load_json_object(manifest_path)
    _layout, partition = partition_from_manifest(manifest, partition_id)
    material_config = MaterialConfig(**manifest["config"]["material"])
    spacing_mm = float(manifest["config"]["mesh"]["base_pixel_size_mm"]) * float(
        manifest["config"]["mesh"]["scale_factor"]
    )
    original_state, original_increment = _history_evm(
        original,
        spacing_mm=spacing_mm,
        poisson_ratio=material_config.poisson_ratio,
    )
    repaired_state, repaired_increment = _history_evm(
        repaired,
        spacing_mm=spacing_mm,
        poisson_ratio=material_config.poisson_ratio,
    )
    original_max = np.max(original_state, axis=(1, 2))
    repaired_max = np.max(repaired_state, axis=(1, 2))
    original_increment_max = np.max(original_increment, axis=(1, 2))
    repaired_increment_max = np.max(repaired_increment, axis=(1, 2))
    unaffected = [
        index for index in range(original.shape[0]) if index not in CORRUPTED_MEASURED_STATES
    ]
    if not np.array_equal(repaired[unaffected], original[unaffected]):
        raise RuntimeError("repair changed a state outside the pre-registered set")

    repaired_path = output / "repaired_history_mm.npy"
    figure_path = output / "corrupted_frames_diagnostic.png"
    np.save(repaired_path, np.asarray(repaired, dtype=np.float32))
    _corrupted_frame_figure(
        figure_path,
        original_state=original_state,
        repaired_state=repaired_state,
        original_increment=original_increment,
        repaired_increment=repaired_increment,
    )
    rows = []
    for state in range(original.shape[0]):
        rows.append(
            {
                "state": state,
                "image": None if state == 0 else f"{294 + state:06d}.tif",
                "repaired": state in CORRUPTED_MEASURED_STATES,
                "original_state_evm_max": float(original_max[state]),
                "repaired_state_evm_max": float(repaired_max[state]),
                "original_increment_evm_max": float(original_increment_max[state]),
                "repaired_increment_evm_max": float(repaired_increment_max[state]),
            }
        )
    report = {
        "schema_version": 1,
        "status": "completed_documented_corrupted_frame_repair",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "partition_id": partition_id,
        "solve_bounds": list(partition.solve_bounds),
        "core_bounds": list(partition.core_bounds),
        "repair": {
            "corrupted_states": list(CORRUPTED_MEASURED_STATES),
            "bracketing_states": list(REPAIR_BRACKETING_STATES),
            "method": "piecewise_linear_displacement_interpolation",
            "legacy_source_declares_corrupted_array_indices": [31, 32],
            "legacy_source_repairs_evm_only": True,
        },
        "source": {
            "history_report": str(source_report_path.resolve()),
            "history_report_sha256": _sha256(source_report_path),
            "history_sha256": _sha256(source_path),
            "campaign_manifest_sha256": _sha256(manifest_path),
        },
        "checks": {
            "finite": bool(np.isfinite(repaired).all()),
            "unaffected_states_bitwise_identical": True,
            "final_state_bitwise_identical": bool(np.array_equal(repaired[-1], original[-1])),
            "largest_repaired_increment_evm_max": float(np.max(repaired_increment_max)),
            "largest_original_increment_evm_max": float(np.max(original_increment_max)),
        },
        "state_diagnostics": rows,
        "outputs": {
            repaired_path.name: _sha256(repaired_path),
            figure_path.name: _sha256(figure_path),
        },
        "mechanics_rerun": False,
        "micromorphic_identification_run": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run_dic_multistep_mechanics(
    *,
    prepared_case: str | Path,
    source_campaign: str | Path,
    history_directory: str | Path,
    partition_id: int,
    mode: str,
    output_directory: str | Path,
    newton_line_search: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run local P43 mechanics with measured or proportional 40-step boundaries."""

    if mode not in {"measured", "proportional"}:
        raise ValueError("mode must be measured or proportional")
    prepared = Path(prepared_case)
    campaign = Path(source_campaign)
    history_root = Path(history_directory)
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = campaign / "manifest.json"
    source_manifest = load_json_object(manifest_path)
    _layout, partition = partition_from_manifest(source_manifest, partition_id)
    sx0, sx1, sy0, sy1 = partition.solve_bounds
    config_data = source_manifest["config"]
    source_solver = SolverConfig(**config_data["solver"])
    source_mesh = config_data["mesh"]
    config = CaseStudyConfig(
        mesh=MeshConfig(
            nx=sx1 - sx0,
            ny=sy1 - sy0,
            base_pixel_size_mm=float(source_mesh["base_pixel_size_mm"]),
            scale_factor=float(source_mesh["scale_factor"]),
        ),
        material=MaterialConfig(**config_data["material"]),
        solver=replace(
            source_solver,
            increments=40,
            newton_line_search=newton_line_search,
        ),
        nonlocal_plasticity=replace(
            NonlocalPlasticityConfig(**config_data["nonlocal_plasticity"]),
            enabled=False,
            coupling_modulus_mpa=0.0,
        ),
    )
    history_report_path = history_root / "report.json"
    history_report = load_json_object(history_report_path)
    accepted_statuses = {
        "completed_direct_reference_history_endpoint_anchored": "anchored_history_mm.npy",
        "completed_documented_corrupted_frame_repair": "repaired_history_mm.npy",
    }
    history_status = str(history_report.get("status"))
    if history_status not in accepted_statuses:
        raise ValueError("history campaign is not complete")
    if int(history_report["partition_id"]) != partition_id:
        raise ValueError("history campaign identifies another partition")
    history_path = history_root / accepted_statuses[history_status]
    if _sha256(history_path) != history_report["outputs"][history_path.name]:
        raise ValueError("anchored history hash does not match its report")
    history = np.array(
        np.load(history_path, mmap_mode="r", allow_pickle=False),
        dtype=np.float64,
        copy=True,
    )

    ux = np.load(prepared / "displacement_x_mm.npy", mmap_mode="r", allow_pickle=False)
    uy = np.load(prepared / "displacement_y_mm.npy", mmap_mode="r", allow_pickle=False)
    yield_map = np.load(prepared / "yield_stress_mpa.npy", mmap_mode="r", allow_pickle=False)
    hardening_map = np.load(
        prepared / "hardening_coefficient_mpa.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    final_x = np.asarray(ux[sx0 : sx1 + 1, sy0 : sy1 + 1], dtype=np.float64)
    final_y = np.asarray(uy[sx0 : sx1 + 1, sy0 : sy1 + 1], dtype=np.float64)
    prepared_final = np.stack((final_x, final_y), axis=-1)
    endpoint_storage_roundoff_mm = float(np.max(np.abs(history[-1] - prepared_final)))
    history[-1] = prepared_final
    local_yield = np.asarray(yield_map[sx0:sx1, sy0:sy1], dtype=np.float64)
    local_hardening = np.asarray(hardening_map[sx0:sx1, sy0:sy1], dtype=np.float64)
    try:
        result = run_case_study(
            config,
            displacement_x_mm=final_x,
            displacement_y_mm=final_y,
            yield_stress_mpa=local_yield,
            hardening_coefficient_mpa=local_hardening,
            boundary_displacement_history_mm=history if mode == "measured" else None,
            snapshots=(0.25, 0.5, 0.75, 1.0),
            verbose=True,
        )
    except NonlinearConvergenceError as error:
        failure_report = {
            "schema_version": 1,
            "status": "failed_local_measured_boundary_history",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_sha": _git_sha(),
            "partition_id": partition_id,
            "mode": mode,
            "error": str(error),
            "diagnostics": error.diagnostics,
            "nominal_increments": 40,
            "newton_line_search_enabled": newton_line_search,
            "config": asdict(config),
            "source": {
                "campaign_manifest_sha256": _sha256(manifest_path),
                "prepared_manifest_sha256": _sha256(prepared / "manifest.json"),
                "history_report_sha256": _sha256(history_report_path),
                "history_sha256": _sha256(history_path),
            },
            "mechanics_completed": False,
            "constitutive_state_committed_after_failure": False,
            "micromorphic_identification_run": False,
        }
        (output / "failure_report.json").write_text(
            json.dumps(failure_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    fields = {
        "U.npy": result.displacement_mm,
        "S.npy": result.stress_mpa,
        "E.npy": result.total_strain,
        "PE.npy": result.plastic_strain,
        "PEEQ.npy": result.equivalent_plastic_strain,
        "RF.npy": result.reaction_force,
    }
    output_hashes: dict[str, str] = {}
    for name, values in fields.items():
        path = output / name
        np.save(path, values)
        output_hashes[name] = _sha256(path)
    frame_hashes: dict[str, dict[str, str]] = {}
    for fraction, frame in sorted(result.frames.items()):
        frame_key = f"{fraction:.2f}"
        frame_hashes[frame_key] = {}
        frame_fields = {
            "U": frame.displacement_mm,
            "E": frame.total_strain,
            "S": frame.stress_mpa,
            "PEEQ": frame.equivalent_plastic_strain,
        }
        for field_name, values in frame_fields.items():
            name = f"frame_{frame_key}_{field_name}.npy"
            path = output / name
            np.save(path, values)
            digest = _sha256(path)
            output_hashes[name] = digest
            frame_hashes[frame_key][field_name] = digest
    report = {
        "schema_version": 1,
        "status": "completed_local_measured_boundary_history",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "partition_id": partition_id,
        "mode": mode,
        "boundary_history": (
            (
                "measured_direct_reference_endpoint_anchored_corrupted_frames_repaired"
                if history_status == "completed_documented_corrupted_frame_repair"
                else "measured_direct_reference_endpoint_anchored"
            )
            if mode == "measured"
            else "proportional_to_prepared_final"
        ),
        "nominal_increments": 40,
        "newton_line_search_enabled": newton_line_search,
        "snapshot_fractions": [0.25, 0.5, 0.75, 1.0],
        "config": asdict(config),
        "solve_bounds": list(partition.solve_bounds),
        "core_bounds": list(partition.core_bounds),
        "source": {
            "campaign_manifest_sha256": _sha256(manifest_path),
            "prepared_manifest_sha256": _sha256(prepared / "manifest.json"),
            "history_report_sha256": _sha256(history_report_path),
            "history_sha256": _sha256(history_path),
            "runtime_endpoint_storage_roundoff_correction_max_mm": (endpoint_storage_roundoff_mm),
        },
        "diagnostics": (asdict(result.diagnostics) if result.diagnostics is not None else None),
        "frames": frame_hashes,
        "outputs": output_hashes,
        "micromorphic_identification_run": False,
        "nonlocal_plasticity_enabled": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
