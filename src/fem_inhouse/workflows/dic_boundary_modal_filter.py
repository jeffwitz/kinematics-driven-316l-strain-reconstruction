"""Truncate a measured boundary history to its resolved modes.

The boundary ring is replaced by a low-rank reconstruction of its departure
from the straight endpoint ramp, so origin and endpoint stay bit-identical and
no discontinuity appears at the last increment. The interior of the history
array is untouched and the solver keeps exact Dirichlet enforcement.

Registered in
`validation/dic_multistep_p0043_modal_boundary_filter_preregistration.md`.
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

from fem_inhouse.workflows.dic_boundary_loading_subspace import (
    boundary_mask,
    temporal_roughness,
)
from fem_inhouse.workflows.dic_observation_replay import PIXEL_SIZE_MM

FloatArray = NDArray[np.float64]

#: Registered per-state boundary noise, in pixels, from the stage-0 campaign.
MEASURED_NOISE_PX = 0.0511

#: Registered truncation rank.
DEFAULT_RANK = 3


@dataclass(frozen=True, slots=True)
class ModalTruncation:
    """A low-rank reconstruction and the content it removed."""

    reconstruction: FloatArray
    removed_rms: float
    retained_energy_fraction: float
    roughness: FloatArray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def endpoint_ramp_deviation(states: NDArray[np.generic]) -> tuple[FloatArray, FloatArray]:
    """Split a history into a straight endpoint ramp and its deviation."""

    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("states must be a (steps, dof) matrix with at least three steps")
    count = values.shape[0] - 1
    fraction = (np.arange(count + 1, dtype=np.float64) / count)[:, None]
    ramp = fraction * values[-1][None, :]
    return ramp, values - ramp


def pin_endpoints(matrix: NDArray[np.generic]) -> FloatArray:
    """Remove the linear field that makes the first and last rows exactly zero.

    A rank-truncated reconstruction does not preserve a zero row, so the two
    pinned states have to be restored explicitly. Subtracting a field linear in
    the state index injects no new temporal structure, and the two end rows
    cancel exactly in IEEE arithmetic.
    """

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("matrix must have at least two rows")
    count = values.shape[0] - 1
    fraction = (np.arange(count + 1, dtype=np.float64) / count)[:, None]
    correction = (1.0 - fraction) * values[0][None, :] + fraction * values[-1][None, :]
    return np.asarray(values - correction, dtype=np.float64)


def truncate_modes(deviation: NDArray[np.generic], *, rank: int) -> ModalTruncation:
    """Keep the leading ``rank`` modes of a deviation matrix."""

    values = np.asarray(deviation, dtype=np.float64)
    if rank < 1 or rank > min(values.shape):
        raise ValueError("rank must be positive and no larger than the matrix rank")
    left, spectrum, right = np.linalg.svd(values, full_matrices=False)
    kept = pin_endpoints(left[:, :rank] * spectrum[:rank] @ right[:rank])
    removed = values - kept
    energy = np.square(spectrum)
    total = float(np.sum(energy))
    reported = range(min(6, spectrum.size))
    roughness = np.asarray(
        [temporal_roughness(left[:, mode] * spectrum[mode]) for mode in reported]
    )
    return ModalTruncation(
        reconstruction=np.ascontiguousarray(kept),
        removed_rms=float(np.sqrt(np.mean(np.square(removed)))),
        retained_energy_fraction=(
            float(np.sum(energy[:rank]) / total) if total > 0.0 else 1.0
        ),
        roughness=roughness,
    )


def filter_dic_boundary_history(
    *,
    history_path: str | Path,
    history_report_path: str | Path,
    output_directory: str | Path,
    rank: int = DEFAULT_RANK,
    pixel_size_mm: float = PIXEL_SIZE_MM,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a boundary-filtered copy of an immutable measured history."""

    history_source = Path(history_path)
    report_source = Path(history_report_path)
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_report = json.loads(report_source.read_text(encoding="utf-8"))
    expected = source_report["outputs"][history_source.name]
    if _sha256(history_source) != expected:
        raise ValueError("history does not match its immutable report")

    history = np.array(
        np.load(history_source, mmap_mode="r", allow_pickle=False),
        dtype=np.float64,
        copy=True,
    )
    if history.ndim != 4 or history.shape[-1] != 2:
        raise ValueError("history must have shape (states, nx + 1, ny + 1, 2)")
    shape = (int(history.shape[1]), int(history.shape[2]))
    mask = boundary_mask(shape)

    boundary = np.stack([state[mask].ravel() for state in history])
    ramp, deviation = endpoint_ramp_deviation(boundary)
    truncation = truncate_modes(deviation, rank=rank)
    filtered_boundary = ramp + truncation.reconstruction

    # Origin and endpoint are exact by construction; assert it rather than trust it.
    if not np.array_equal(filtered_boundary[0], boundary[0]):
        raise RuntimeError("the filter moved the reference state")
    if not np.allclose(filtered_boundary[-1], boundary[-1], rtol=0.0, atol=0.0):
        raise RuntimeError("the filter moved the endpoint")

    filtered = history.copy()
    node_count = int(np.count_nonzero(mask))
    for index in range(filtered.shape[0]):
        filtered[index][mask] = filtered_boundary[index].reshape(node_count, 2)

    interior = ~mask
    if not np.array_equal(filtered[:, interior, :], history[:, interior, :]):
        raise RuntimeError("the filter touched the interior")

    removed_px = truncation.removed_rms / pixel_size_mm
    within_noise = bool(removed_px <= MEASURED_NOISE_PX)

    destination = output / "modal_filtered_history_mm.npy"
    np.save(destination, filtered)
    digest = _sha256(destination)

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed_modal_boundary_filter",
        "filter": {
            "kind": "modal_truncation_of_endpoint_ramp_deviation",
            "rank": int(rank),
            "removed_rms_mm": truncation.removed_rms,
            "removed_rms_px": removed_px,
            "measured_noise_px": MEASURED_NOISE_PX,
            "removed_within_measured_noise": within_noise,
            "retained_deviation_energy_fraction": truncation.retained_energy_fraction,
            "leading_deviation_roughness": [float(v) for v in truncation.roughness],
            "origin_and_endpoint_preserved": True,
            "interior_untouched": True,
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "partition_id": int(source_report["partition_id"]),
        "preregistration": (
            "validation/dic_multistep_p0043_modal_boundary_filter_preregistration.md"
        ),
        "solve_bounds": list(source_report["solve_bounds"]),
        "core_bounds": list(source_report["core_bounds"]),
        "source": {
            "history": str(history_source.resolve()),
            "history_sha256": expected,
            "history_report": str(report_source.resolve()),
        },
        "support": {"node_shape": list(shape), "boundary_nodes": node_count},
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "outputs": {destination.name: digest},
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
