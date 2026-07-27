"""Prepare immutable material-map controls from a canonical case study."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np

from fem_inhouse.data_preparation import fingerprint_file

ControlMode = Literal["homogeneous", "translated"]
_FIELDS = (
    "displacement_x_mm",
    "displacement_y_mm",
    "yield_stress_mpa",
    "hardening_coefficient_mpa",
)


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing canonical input manifest: {manifest_path}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("canonical input manifest must contain a JSON object")
    return value


def _verified_field(source: Path, manifest: dict[str, Any], name: str) -> np.ndarray:
    path = source / f"{name}.npy"
    declaration = manifest.get("outputs", {}).get(name)
    if not isinstance(declaration, dict):
        raise ValueError(f"source manifest does not declare {name}")
    if fingerprint_file(path) != declaration.get("sha256"):
        raise RuntimeError(f"source field fails its manifest hash: {path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError(f"source field {name} must be finite and numeric")
    return values


def prepare_material_map_control(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    mode: ControlMode,
    homogeneous_yield_stress_mpa: float = 124.0,
    homogeneous_hardening_coefficient_mpa: float = 380.0,
    shift_x_pixels: int = 600,
    shift_y_pixels: int = 500,
) -> dict[str, Any]:
    """Create a self-contained homogeneous or translated-map control input."""

    source = Path(source_directory)
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing control input: {output}")
    if mode not in ("homogeneous", "translated"):
        raise ValueError(f"unsupported material-map control mode: {mode}")
    for name, value in (
        ("homogeneous_yield_stress_mpa", homogeneous_yield_stress_mpa),
        ("homogeneous_hardening_coefficient_mpa", homogeneous_hardening_coefficient_mpa),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive")

    source_manifest = _load_manifest(source)
    fields = {name: _verified_field(source, source_manifest, name) for name in _FIELDS}
    if fields["yield_stress_mpa"].shape != fields["hardening_coefficient_mpa"].shape:
        raise ValueError("source material maps do not have the same shape")

    output.mkdir(parents=True)
    for name in ("displacement_x_mm", "displacement_y_mm"):
        shutil.copy2(source / f"{name}.npy", output / f"{name}.npy")

    if mode == "homogeneous":
        yield_control = np.full(
            fields["yield_stress_mpa"].shape,
            homogeneous_yield_stress_mpa,
            dtype=np.float64,
        )
        hardening_control = np.full(
            fields["hardening_coefficient_mpa"].shape,
            homogeneous_hardening_coefficient_mpa,
            dtype=np.float64,
        )
        transformation: dict[str, Any] = {
            "mode": mode,
            "yield_stress_mpa": float(homogeneous_yield_stress_mpa),
            "hardening_coefficient_mpa": float(homogeneous_hardening_coefficient_mpa),
        }
    else:
        shifts = (int(shift_x_pixels), int(shift_y_pixels))
        if shifts == (0, 0):
            raise ValueError("translated control requires a non-zero shift")
        yield_control = np.roll(fields["yield_stress_mpa"], shift=shifts, axis=(0, 1))
        hardening_control = np.roll(
            fields["hardening_coefficient_mpa"],
            shift=shifts,
            axis=(0, 1),
        )
        transformation = {
            "mode": mode,
            "shift_x_pixels": shifts[0],
            "shift_y_pixels": shifts[1],
            "boundary_rule": "periodic_toroidal_roll",
            "joint_pairing_preserved": True,
        }
    np.save(output / "yield_stress_mpa.npy", yield_control)
    np.save(output / "hardening_coefficient_mpa.npy", hardening_control)

    declarations: dict[str, Any] = {}
    for name in _FIELDS:
        path = output / f"{name}.npy"
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        declarations[name] = {
            "filename": path.name,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "bytes": path.stat().st_size,
            "sha256": fingerprint_file(path),
        }
    manifest = {
        "schema_version": 1,
        "kind": "material_map_control",
        "source": {
            "directory": str(source),
            "manifest_sha256": fingerprint_file(source / "manifest.json"),
            "field_sha256": {
                name: source_manifest["outputs"][name]["sha256"] for name in _FIELDS
            },
        },
        "transformation": transformation,
        "outputs": declarations,
        "claim_boundary": (
            "The control changes only sigma_y and K. Displacements and all solver "
            "settings remain unchanged. The translated control preserves map "
            "distributions and local sigma_y/K pairing but introduces periodic seams."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
