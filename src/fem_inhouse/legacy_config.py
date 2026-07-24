"""Portable data contract for the historical analysis scripts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

NX = int(os.environ.get("FEM_INHOUSE_LEGACY_NX", "10"))
NY = int(os.environ.get("FEM_INHOUSE_LEGACY_NY", "10"))
ELEMENT_SIZE = float(os.environ.get("FEM_INHOUSE_BASE_PIXEL_MM", "0.001"))
SCALE_FACTOR = float(os.environ.get("FEM_INHOUSE_SCALE_FACTOR", "1.84"))
N_EXP = float(os.environ.get("FEM_INHOUSE_HARDENING_EXPONENT", "0.245"))
X_SIZE = NX * ELEMENT_SIZE
Y_SIZE = NY * ELEMENT_SIZE
JOB_NAME = os.environ.get("FEM_INHOUSE_JOB_NAME", f"case5_{NX}x{NY}")
FEM_TAG = os.environ.get("FEM_INHOUSE_FEM_TAG", f"fem_test_{NX}x{NY}")
DIC_FINAL_FRAME = int(os.environ.get("FEM_INHOUSE_DIC_FINAL_FRAME", "40"))
PX_TO_MM = ELEMENT_SIZE * SCALE_FACTOR


@dataclass(frozen=True, slots=True)
class LegacyCasePaths:
    """Locations required by scripts retained for article-result migration."""

    input_directory: Path
    dic_directory: Path
    macro_stress_strain_file: Path
    validation_directory: Path

    @classmethod
    def from_environment(cls) -> LegacyCasePaths:
        data_root = Path(os.environ.get("FEM_INHOUSE_DATA_DIR", Path.cwd() / "data"))
        results_root = Path(os.environ.get("FEM_INHOUSE_RESULTS_DIR", Path.cwd() / "results"))
        return cls(
            input_directory=Path(os.environ.get("FEM_INHOUSE_INPUT_DIR", data_root / "case_study")),
            dic_directory=Path(os.environ.get("FEM_INHOUSE_DIC_DIR", data_root / "dic")),
            macro_stress_strain_file=Path(
                os.environ.get(
                    "FEM_INHOUSE_MACRO_FILE",
                    data_root / "stress_strain.npy",
                )
            ),
            validation_directory=Path(
                os.environ.get(
                    "FEM_INHOUSE_VALIDATION_DIR",
                    results_root / "final_validation",
                )
            ),
        )


PATHS = LegacyCasePaths.from_environment()
DIC_DIR = str(PATHS.dic_directory)
FINAL_VALIDATION_DIR = str(PATHS.validation_directory)
AB_OUT = str(PATHS.validation_directory / "abaqus")
FEM_OUT = str(PATHS.validation_directory / "fem_single")
MACRO_STRESS_STRAIN_FILE = str(PATHS.macro_stress_strain_file)


def crop_center(values: ArrayLike, rows: int, columns: int) -> NDArray:
    """Return a centred crop along the first two array axes."""

    array = np.asarray(values)
    if rows < 1 or columns < 1:
        raise ValueError("crop dimensions must be positive")
    if array.ndim < 2 or array.shape[0] < rows or array.shape[1] < columns:
        raise ValueError(f"cannot crop shape {array.shape} to first axes {(rows, columns)}")
    start_row = (array.shape[0] - rows) // 2
    start_column = (array.shape[1] - columns) // 2
    return array[
        start_row : start_row + rows,
        start_column : start_column + columns,
        ...,
    ]


def _load_field(path: Path, *, shape: tuple[int, int], name: str) -> NDArray:
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}. See docs/legacy_data_contract.md.")
    values = np.load(path, mmap_mode="r")
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D .npy array, got shape {values.shape}")
    if values.shape != shape:
        values = crop_center(values, *shape)
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


def load_case5_inputs(
    paths: LegacyCasePaths | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Load the four explicitly named arrays used by historical scripts."""

    selected = PATHS if paths is None else paths
    nodal_shape = (NX + 1, NY + 1)
    element_shape = (NX, NY)
    displacement_x = _load_field(
        selected.input_directory / "displacement_x_mm.npy",
        shape=nodal_shape,
        name="displacement_x_mm",
    )
    displacement_y = _load_field(
        selected.input_directory / "displacement_y_mm.npy",
        shape=nodal_shape,
        name="displacement_y_mm",
    )
    yield_stress = _load_field(
        selected.input_directory / "yield_stress_mpa.npy",
        shape=element_shape,
        name="yield_stress_mpa",
    )
    hardening = _load_field(
        selected.input_directory / "hardening_coefficient_mpa.npy",
        shape=element_shape,
        name="hardening_coefficient_mpa",
    )
    if np.any(yield_stress <= 0):
        raise ValueError("yield_stress_mpa must be strictly positive")
    if np.any(hardening < 0):
        raise ValueError("hardening_coefficient_mpa must be nonnegative")
    return displacement_x, displacement_y, yield_stress, hardening


def window_tag() -> str:
    """Return the deterministic historical window label."""

    return f"{NX}x{NY}"


__all__ = [
    "AB_OUT",
    "DIC_DIR",
    "DIC_FINAL_FRAME",
    "ELEMENT_SIZE",
    "FEM_OUT",
    "FEM_TAG",
    "FINAL_VALIDATION_DIR",
    "JOB_NAME",
    "MACRO_STRESS_STRAIN_FILE",
    "NX",
    "NY",
    "N_EXP",
    "PATHS",
    "PX_TO_MM",
    "SCALE_FACTOR",
    "X_SIZE",
    "Y_SIZE",
    "LegacyCasePaths",
    "crop_center",
    "load_case5_inputs",
    "window_tag",
]
