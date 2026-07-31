"""Compare DISFlow profiles by how well they reproduce the archived field.

Selecting an observation operator by its agreement with the model is circular.
Reproduction of the archived measured displacement is not: it compares a
recomputation against data produced by the historical chain, with no mechanics
involved.

Registered in
`validation/dic_profile_endpoint_reproduction_preregistration.md`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from fem_inhouse.measurement import disflow_profile, image_flow_to_canonical, run_disflow
from fem_inhouse.workflows.dic_multistep import RAW_CROP_SHAPE
from fem_inhouse.workflows.dic_observation_replay import (
    PIXEL_SIZE_MM,
    RAW_CROP_COLUMN_START,
    RAW_CROP_ROW_START,
)

FloatArray = NDArray[np.float64]

#: Registered separation factor on the relative vector norm.
DISCRIMINATION_FACTOR = 1.5

#: Archived legacy_script_2021 figure, recomputed and checked rather than quoted.
#: It was computed on the P43 partition support, not on the full field.
ARCHIVED_LEGACY_RELATIVE_NORM = 0.01583

#: Solve bounds of the P43 partition, the support of the archived figure.
P43_SOLVE_BOUNDS = (1290, 1950, 780, 1390)


@dataclass(frozen=True, slots=True)
class ReproductionMetrics:
    """Agreement between a recomputed field and the archived prepared one."""

    component_rms_mm: float
    maximum_absolute_mm: float
    relative_vector_norm: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def reproduction_metrics(
    recomputed: NDArray[np.generic],
    prepared: NDArray[np.generic],
) -> ReproductionMetrics:
    """Compare two displacement fields of identical support."""

    a = np.asarray(recomputed, dtype=np.float64)
    b = np.asarray(prepared, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 2:
        raise ValueError("both fields must share a shape (nx, ny, 2)")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("fields must be finite")
    difference = a - b
    reference_norm = float(np.sqrt(np.sum(np.square(b))))
    if reference_norm == 0.0:
        raise ValueError("the prepared field is identically zero")
    return ReproductionMetrics(
        component_rms_mm=float(np.sqrt(np.mean(np.square(difference)))),
        maximum_absolute_mm=float(np.max(np.abs(difference))),
        relative_vector_norm=float(
            np.sqrt(np.sum(np.square(difference))) / reference_norm
        ),
    )


def _crop(path: Path) -> NDArray[np.uint8]:
    full = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    window = (
        slice(RAW_CROP_ROW_START, RAW_CROP_ROW_START + RAW_CROP_SHAPE[0]),
        slice(RAW_CROP_COLUMN_START, RAW_CROP_COLUMN_START + RAW_CROP_SHAPE[1]),
    )
    cropped = np.ascontiguousarray(full[window])
    if cropped.shape != RAW_CROP_SHAPE:
        raise ValueError(f"{path.name} does not contain the declared crop")
    return cropped


def compare_profile_reproduction(
    *,
    raw_image_directory: str | Path,
    prepared_case: str | Path,
    output_directory: str | Path,
    reference_image: str = "000294.tif",
    final_image: str = "000334.tif",
    profiles: tuple[str, ...] = ("legacy_script_2021", "declared_medium_v4"),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Rank DISFlow profiles by their reproduction of the archived field."""

    images = Path(raw_image_directory)
    prepared_root = Path(prepared_case)
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if len(profiles) < 2:
        raise ValueError("at least two profiles are needed for a comparison")

    reference = _crop(images / reference_image)
    final = _crop(images / final_image)

    # The prepared field is nodal; its last row and column are an edge-pad copy,
    # not a measurement, so the comparison uses the measured sub-block only.
    prepared_x = np.load(prepared_root / "displacement_x_mm.npy", mmap_mode="r")
    prepared_y = np.load(prepared_root / "displacement_y_mm.npy", mmap_mode="r")
    nx, ny = RAW_CROP_SHAPE
    prepared = np.stack(
        (
            np.asarray(prepared_x[:nx, :ny], dtype=np.float64),
            np.asarray(prepared_y[:nx, :ny], dtype=np.float64),
        ),
        axis=-1,
    )

    results: dict[str, Any] = {}
    for name in profiles:
        profile = disflow_profile(name)
        flow = run_disflow(reference, final, config=profile.config)
        recomputed = image_flow_to_canonical(flow, pixel_size_mm=PIXEL_SIZE_MM)
        if recomputed.shape != prepared.shape:
            raise ValueError(f"{name} produced an incompatible support")
        metrics = reproduction_metrics(recomputed, prepared)
        x0, x1, y0, y1 = P43_SOLVE_BOUNDS
        window = (slice(x0, x1 + 1), slice(y0, y1 + 1))
        partition = reproduction_metrics(recomputed[window], prepared[window])
        results[name] = {
            "source": profile.source,
            "patch_size": profile.config.patch_size,
            "patch_stride": profile.config.patch_stride,
            "component_rms_mm": metrics.component_rms_mm,
            "maximum_absolute_mm": metrics.maximum_absolute_mm,
            "relative_vector_norm": metrics.relative_vector_norm,
            "p43_component_rms_mm": partition.component_rms_mm,
            "p43_maximum_absolute_mm": partition.maximum_absolute_mm,
            "p43_relative_vector_norm": partition.relative_vector_norm,
        }

    ranked = sorted(results, key=lambda n: results[n]["relative_vector_norm"])
    best, worst = ranked[0], ranked[-1]
    ratio = (
        results[worst]["relative_vector_norm"] / results[best]["relative_vector_norm"]
    )
    discriminates = bool(ratio >= DISCRIMINATION_FACTOR)
    if not discriminates:
        verdict = "reproduction_does_not_discriminate"
    elif best == "legacy_script_2021":
        verdict = "provenance_confirmed_by_reproduction"
    else:
        verdict = "provenance_contradicted"

    # The archived figure must be recovered, or the campaign is void.
    legacy = results.get("legacy_script_2021", {}).get("p43_relative_vector_norm")
    consistent = (
        None
        if legacy is None
        else bool(abs(legacy - ARCHIVED_LEGACY_RELATIVE_NORM) <= 5.0e-4)
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed_profile_endpoint_reproduction",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "preregistration": (
            "validation/dic_profile_endpoint_reproduction_preregistration.md"
        ),
        "support": {
            "compared_shape": [nx, ny],
            "p43_solve_bounds": list(P43_SOLVE_BOUNDS),
            "excludes_edge_pad_completion": True,
            "pixel_size_mm": PIXEL_SIZE_MM,
            "raw_crop_row_start": RAW_CROP_ROW_START,
            "raw_crop_column_start": RAW_CROP_COLUMN_START,
        },
        "sources": {
            "reference_image": str((images / reference_image).resolve()),
            "reference_image_sha256": _sha256(images / reference_image),
            "final_image": str((images / final_image).resolve()),
            "final_image_sha256": _sha256(images / final_image),
            "prepared_manifest_sha256": _sha256(prepared_root / "manifest.json"),
        },
        "profiles": results,
        "comparison": {
            "best": best,
            "relative_norm_ratio": ratio,
            "discrimination_factor": DISCRIMINATION_FACTOR,
            "discriminates": discriminates,
        },
        "archived_consistency": {
            "archived_legacy_relative_vector_norm": ARCHIVED_LEGACY_RELATIVE_NORM,
            "archived_support": "P43 partition, not the full field",
            "recomputed_p43_matches_archive": consistent,
        },
        "verdict": verdict,
        "claim_boundary": (
            "reproduction fidelity to the historical chain only; says nothing "
            "about which profile measures displacement more accurately, and does "
            "not identify the historical configuration uniquely"
        ),
        "mechanics_rerun": False,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
