"""Reproducible preparation of the versioned DIC case-study inputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from numpy.lib.format import open_memmap
from numpy.typing import NDArray

from fem_inhouse import __version__

NonfinitePolicy = Literal["error", "nearest"]
NodalCompletion = Literal["edge-pad-upper"]

RAW_FILENAMES = {
    "displacement_y_pixels": "U_40.npy",
    "displacement_x_pixels": "V_40.npy",
    "yield_stress_mpa": "el_thresh50.npy",
    "hardening_multiplier": "Hardening_coeff_el_Thresh50.npy",
}
CANONICAL_FILENAMES = {
    "displacement_x_mm": "displacement_x_mm.npy",
    "displacement_y_mm": "displacement_y_mm.npy",
    "yield_stress_mpa": "yield_stress_mpa.npy",
    "hardening_coefficient_mpa": "hardening_coefficient_mpa.npy",
}


@dataclass(frozen=True, slots=True)
class PreparationConfig:
    """Every scientific choice applied while canonicalizing the raw arrays."""

    pixel_size_um: float = 1.84
    hardening_scale_mpa: float = 380.0
    nonfinite_policy: NonfinitePolicy = "error"
    nodal_completion: NodalCompletion = "edge-pad-upper"
    crop_nx: int | None = None
    crop_ny: int | None = None

    def __post_init__(self) -> None:
        if self.pixel_size_um <= 0:
            raise ValueError("pixel_size_um must be positive")
        if self.hardening_scale_mpa <= 0:
            raise ValueError("hardening_scale_mpa must be positive")
        if self.nonfinite_policy not in ("error", "nearest"):
            raise ValueError(f"unsupported nonfinite_policy {self.nonfinite_policy!r}")
        if self.nodal_completion != "edge-pad-upper":
            raise ValueError(f"unsupported nodal_completion {self.nodal_completion!r}")
        if (self.crop_nx is None) != (self.crop_ny is None):
            raise ValueError("crop_nx and crop_ny must be specified together")
        invalid_crop = self.crop_nx is not None and (
            self.crop_nx < 1 or self.crop_ny is None or self.crop_ny < 1
        )
        if invalid_crop:
            raise ValueError("crop dimensions must be positive")

    @property
    def displacement_scale_mm_per_pixel(self) -> float:
        return self.pixel_size_um / 1000.0


def fingerprint_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Return the SHA-256 of one file without loading it all into memory."""

    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    digest = sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _read_raw_manifest(raw_directory: Path) -> dict[str, Any]:
    manifest_path = raw_directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"raw manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid raw manifest: {manifest_path}") from error
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("unsupported raw manifest schema")
    return manifest


def verify_raw_case_study(raw_directory: str | Path) -> dict[str, Any]:
    """Verify every raw NPY against the versioned manifest and return it."""

    source = Path(raw_directory)
    manifest = _read_raw_manifest(source)
    declared_files = manifest["files"]
    for filename in RAW_FILENAMES.values():
        path = source / filename
        if not path.is_file():
            raise FileNotFoundError(f"raw input not found: {path}")
        try:
            declared = declared_files[filename]
            expected_bytes = int(declared["bytes"])
            expected_sha256 = str(declared["sha256"])
            expected_shape = tuple(int(value) for value in declared["shape"])
            expected_dtype = np.dtype(declared["dtype"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"incomplete manifest entry for {filename}") from error
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"{filename} byte size does not match the raw manifest")
        if fingerprint_file(path) != expected_sha256:
            raise ValueError(f"{filename} SHA-256 does not match the raw manifest")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != expected_shape:
            raise ValueError(f"{filename} shape {array.shape} does not match {expected_shape}")
        if array.dtype != expected_dtype:
            raise ValueError(f"{filename} dtype {array.dtype} does not match {expected_dtype}")
    return manifest


def _create_memmap(path: Path, *, shape: tuple[int, ...]) -> np.memmap:
    return open_memmap(path, mode="w+", dtype=np.float64, shape=shape)


def _center_crop(
    values: NDArray[Any],
    *,
    rows: int,
    columns: int,
) -> tuple[NDArray[Any], tuple[int, int, int, int]]:
    if rows > values.shape[0] or columns > values.shape[1]:
        raise ValueError(f"cannot crop shape {values.shape} to {(rows, columns)}")
    row_start = (values.shape[0] - rows) // 2
    column_start = (values.shape[1] - columns) // 2
    bounds = (row_start, row_start + rows, column_start, column_start + columns)
    return values[bounds[0] : bounds[1], bounds[2] : bounds[3]], bounds


def _write_scaled_rows(
    source: NDArray[Any],
    destination: NDArray[np.float64],
    *,
    scale: float,
    rows_per_chunk: int = 128,
) -> None:
    for start in range(0, source.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, source.shape[0])
        destination[start:stop] = np.asarray(source[start:stop], dtype=np.float64) * scale


def _write_edge_padded_displacement(
    source: NDArray[Any],
    destination: NDArray[np.float64],
    *,
    scale: float,
) -> None:
    _write_scaled_rows(source, destination[:-1, :-1], scale=scale)
    destination[-1, :-1] = np.asarray(source[-1, :], dtype=np.float64) * scale
    destination[:-1, -1] = np.asarray(source[:, -1], dtype=np.float64) * scale
    destination[-1, -1] = float(source[-1, -1]) * scale


def _nearest_finite_value(values: NDArray[Any], row: int, column: int) -> float:
    maximum_radius = max(values.shape)
    for radius in range(1, maximum_radius):
        row_start = max(0, row - radius)
        row_stop = min(values.shape[0], row + radius + 1)
        column_start = max(0, column - radius)
        column_stop = min(values.shape[1], column + radius + 1)
        window = np.asarray(values[row_start:row_stop, column_start:column_stop])
        finite_indices = np.argwhere(np.isfinite(window))
        if finite_indices.size:
            global_rows = finite_indices[:, 0] + row_start
            global_columns = finite_indices[:, 1] + column_start
            distances = (global_rows - row) ** 2 + (global_columns - column) ** 2
            selected = int(np.argmin(distances))
            return float(values[global_rows[selected], global_columns[selected]])
    raise ValueError("hardening multiplier has no finite value")


def _repair_nonfinite_nearest(
    source: NDArray[Any],
    destination: NDArray[np.float64],
    *,
    scale: float,
) -> list[list[int]]:
    nonfinite_indices = np.argwhere(~np.isfinite(source))
    for row, column in nonfinite_indices:
        destination[row, column] = _nearest_finite_value(source, int(row), int(column)) * scale
    return nonfinite_indices.astype(int).tolist()


def _output_file_metadata(directory: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for logical_name, filename in CANONICAL_FILENAMES.items():
        path = directory / filename
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        metadata[logical_name] = {
            "bytes": path.stat().st_size,
            "dtype": str(array.dtype),
            "filename": filename,
            "sha256": fingerprint_file(path),
            "shape": list(array.shape),
        }
    return metadata


def _existing_preparation(
    destination: Path,
    *,
    raw_manifest_sha256: str,
    config: PreparationConfig,
) -> dict[str, Any] | None:
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid existing preparation manifest: {manifest_path}") from error
    expected = {
        "config": asdict(config),
        "raw_manifest_sha256": raw_manifest_sha256,
    }
    actual = {
        "config": manifest.get("config"),
        "raw_manifest_sha256": manifest.get("source", {}).get("raw_manifest_sha256"),
    }
    if actual != expected:
        raise RuntimeError("existing prepared data use a different source or configuration")
    for file_metadata in manifest.get("outputs", {}).values():
        path = destination / file_metadata["filename"]
        if not path.is_file() or fingerprint_file(path) != file_metadata["sha256"]:
            raise RuntimeError(f"existing prepared output is missing or corrupted: {path}")
    return manifest


def prepare_case_study(
    raw_directory: str | Path,
    output_directory: str | Path,
    *,
    config: PreparationConfig | None = None,
) -> dict[str, Any]:
    """Prepare canonical solver arrays atomically and return their manifest."""

    source = Path(raw_directory)
    destination = Path(output_directory)
    selected_config = PreparationConfig() if config is None else config
    raw_manifest = verify_raw_case_study(source)
    raw_manifest_path = source / "manifest.json"
    raw_manifest_sha256 = fingerprint_file(raw_manifest_path)
    existing = _existing_preparation(
        destination,
        raw_manifest_sha256=raw_manifest_sha256,
        config=selected_config,
    )
    if existing is not None:
        return existing
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"output directory is not empty: {destination}")

    full_raw_arrays = {
        logical_name: np.load(source / filename, mmap_mode="r", allow_pickle=False)
        for logical_name, filename in RAW_FILENAMES.items()
    }
    crop_bounds: tuple[int, int, int, int] | None = None
    if selected_config.crop_nx is not None and selected_config.crop_ny is not None:
        raw_arrays = {}
        for logical_name, array in full_raw_arrays.items():
            cropped, bounds = _center_crop(
                array,
                rows=selected_config.crop_nx,
                columns=selected_config.crop_ny,
            )
            raw_arrays[logical_name] = cropped
            if crop_bounds is None:
                crop_bounds = bounds
            elif crop_bounds != bounds:
                raise ValueError("raw arrays do not share the same crop coordinates")
    else:
        raw_arrays = full_raw_arrays
    element_shape = raw_arrays["yield_stress_mpa"].shape
    if len(element_shape) != 2:
        raise ValueError(f"material fields must be two-dimensional, got {element_shape}")
    if raw_arrays["hardening_multiplier"].shape != element_shape:
        raise ValueError("material maps must have the same shape")
    for name in ("displacement_x_pixels", "displacement_y_pixels"):
        if raw_arrays[name].shape != element_shape:
            raise ValueError("the supported raw DIC grid must match the material-map grid")
    yield_map = raw_arrays["yield_stress_mpa"]
    if not np.isfinite(yield_map).all() or np.any(yield_map <= 0):
        raise ValueError("yield-stress map must contain finite positive values")
    hardening_multiplier = raw_arrays["hardening_multiplier"]
    nonfinite_indices = np.argwhere(~np.isfinite(hardening_multiplier))
    if nonfinite_indices.size and selected_config.nonfinite_policy == "error":
        raise ValueError(
            f"hardening multiplier contains {len(nonfinite_indices)} non-finite values; "
            "select nonfinite_policy='nearest' explicitly to repair them"
        )
    finite_hardening = np.asarray(hardening_multiplier[np.isfinite(hardening_multiplier)])
    if finite_hardening.size == 0 or np.any(finite_hardening < 0):
        raise ValueError("finite hardening multipliers must be nonnegative")

    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        nodal_shape = (element_shape[0] + 1, element_shape[1] + 1)
        displacement_scale = selected_config.displacement_scale_mm_per_pixel
        ux = _create_memmap(temporary / CANONICAL_FILENAMES["displacement_x_mm"], shape=nodal_shape)
        uy = _create_memmap(temporary / CANONICAL_FILENAMES["displacement_y_mm"], shape=nodal_shape)
        sy = _create_memmap(
            temporary / CANONICAL_FILENAMES["yield_stress_mpa"],
            shape=element_shape,
        )
        hardening = _create_memmap(
            temporary / CANONICAL_FILENAMES["hardening_coefficient_mpa"],
            shape=element_shape,
        )
        _write_edge_padded_displacement(
            raw_arrays["displacement_x_pixels"],
            ux,
            scale=displacement_scale,
        )
        _write_edge_padded_displacement(
            raw_arrays["displacement_y_pixels"],
            uy,
            scale=displacement_scale,
        )
        _write_scaled_rows(yield_map, sy, scale=1.0)
        _write_scaled_rows(
            hardening_multiplier,
            hardening,
            scale=selected_config.hardening_scale_mpa,
        )
        repaired_indices: list[list[int]] = []
        if nonfinite_indices.size:
            repaired_indices = _repair_nonfinite_nearest(
                hardening_multiplier,
                hardening,
                scale=selected_config.hardening_scale_mpa,
            )
        for array in (ux, uy, sy, hardening):
            array.flush()
        del ux, uy, sy, hardening

        manifest = {
            "schema_version": 1,
            "software": {
                "name": "kinematics-driven-316l-strain-reconstruction",
                "version": __version__,
            },
            "source": {
                "raw_manifest_sha256": raw_manifest_sha256,
                "raw_files": {
                    filename: raw_manifest["files"][filename]["sha256"]
                    for filename in RAW_FILENAMES.values()
                },
            },
            "config": asdict(selected_config),
            "transformations": {
                "axis_convention": "axis 0=x/transverse; axis 1=y/tensile",
                "displacement_mapping": "V_40 -> u_x; U_40 -> u_y",
                "displacement_scale_mm_per_pixel": displacement_scale,
                "hardening_nonfinite_repaired_indices": repaired_indices,
                "nodal_completion": (
                    "duplicate the final row and column on the upper array bounds"
                ),
                "source_crop_bounds_axis_0_axis_1": list(crop_bounds) if crop_bounds else None,
            },
            "outputs": _output_file_metadata(temporary),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.rmdir()
        temporary.replace(destination)
        return manifest
    finally:
        if temporary.exists():
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()
