"""Resumable partition-by-partition reconstruction workflow."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse import __version__
from fem_inhouse.config import CaseStudyConfig, MeshConfig
from fem_inhouse.partitioning import (
    PartitionLayout,
    extract_partition_field,
    stitch_partition_files,
)
from fem_inhouse.solver import run_case_study

LOGGER = logging.getLogger(__name__)
FieldLocation = Literal["element", "node"]
RESULT_FIELDS: dict[str, tuple[str, FieldLocation]] = {
    "U": ("displacement_mm", "node"),
    "S": ("stress_mpa", "element"),
    "E": ("total_strain", "element"),
    "PEEQ": ("equivalent_plastic_strain", "element"),
}


def fingerprint_array(values: ArrayLike, *, chunk_elements: int = 131_072) -> str:
    """Hash array metadata and C-order values without materializing a full copy."""

    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")
    array = np.asarray(values)
    digest = sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    iterator = np.nditer(
        array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="C",
        buffersize=chunk_elements,
    )
    for chunk in iterator:
        digest.update(np.asarray(chunk).tobytes(order="C"))
    return digest.hexdigest()


def _fingerprint_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint() -> str:
    package_root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for source_path in sorted(package_root.rglob("*.py")):
        digest.update(source_path.relative_to(package_root).as_posix().encode())
        digest.update(source_path.read_bytes())
    return digest.hexdigest()


def _canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_save_array(path: Path, values: NDArray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, values)
    temporary.replace(path)


@dataclass(slots=True)
class PartitionWorkflow:
    """Own one deterministic, resumable partitioned reconstruction run."""

    config: CaseStudyConfig
    layout: PartitionLayout
    displacement_x_mm: ArrayLike
    displacement_y_mm: ArrayLike
    yield_stress_mpa: ArrayLike
    hardening_coefficient_mpa: ArrayLike
    output_directory: Path
    _manifest_digest: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.layout.global_shape != (self.config.mesh.nx, self.config.mesh.ny):
            raise ValueError("layout global_shape must match the configured mesh")
        nodal_shape = (self.config.mesh.nx + 1, self.config.mesh.ny + 1)
        element_shape = (self.config.mesh.nx, self.config.mesh.ny)
        for name, values, expected_shape in (
            ("displacement_x_mm", self.displacement_x_mm, nodal_shape),
            ("displacement_y_mm", self.displacement_y_mm, nodal_shape),
            ("yield_stress_mpa", self.yield_stress_mpa, element_shape),
            (
                "hardening_coefficient_mpa",
                self.hardening_coefficient_mpa,
                element_shape,
            ),
        ):
            if np.shape(values) != expected_shape:
                raise ValueError(f"{name} has shape {np.shape(values)}, expected {expected_shape}")
        self.output_directory = Path(self.output_directory)

    @property
    def manifest_path(self) -> Path:
        return self.output_directory / "manifest.json"

    def _manifest_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "software": {
                "name": "kinematics-driven-316l-strain-reconstruction",
                "version": __version__,
                "source_sha256": _source_fingerprint(),
            },
            "config": asdict(self.config),
            "layout": self.layout.as_dict(),
            "inputs": {
                "displacement_x_mm": fingerprint_array(self.displacement_x_mm),
                "displacement_y_mm": fingerprint_array(self.displacement_y_mm),
                "yield_stress_mpa": fingerprint_array(self.yield_stress_mpa),
                "hardening_coefficient_mpa": fingerprint_array(self.hardening_coefficient_mpa),
            },
            "result_fields": sorted(RESULT_FIELDS),
        }

    def prepare(self) -> str:
        """Create or verify the immutable run manifest and return its digest."""

        if self._manifest_digest is not None:
            return self._manifest_digest
        content = _canonical_json(self._manifest_data())
        digest = sha256(content.encode()).hexdigest()
        if self.manifest_path.exists():
            existing = self.manifest_path.read_text(encoding="utf-8")
            if existing != content:
                raise RuntimeError(
                    "existing run manifest does not match configuration or input fields"
                )
        else:
            _atomic_write_text(self.manifest_path, content)
        self._manifest_digest = digest
        return digest

    def _partition_directory(self, partition_id: int) -> Path:
        return self.output_directory / "partitions" / f"{partition_id:04d}"

    def _status_path(self, partition_id: int) -> Path:
        return self._partition_directory(partition_id) / "status.json"

    def _result_path(self, partition_id: int, field_name: str) -> Path:
        return self._partition_directory(partition_id) / f"{field_name}.npy"

    def _completed_status(self, partition_id: int, manifest_digest: str) -> dict[str, Any] | None:
        status_path = self._status_path(partition_id)
        if not status_path.exists():
            return None
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if not status.get("complete") or status.get("manifest_sha256") != manifest_digest:
                return None
            expected_outputs = status["outputs"]
            for field_name in RESULT_FIELDS:
                path = self._result_path(partition_id, field_name)
                if not path.is_file() or _fingerprint_file(path) != expected_outputs[field_name]:
                    return None
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            return None
        return status

    def pending_partition_ids(self) -> list[int]:
        """Return incomplete or corrupted partitions in deterministic order."""

        manifest_digest = self.prepare()
        return [
            partition.partition_id
            for partition in self.layout
            if self._completed_status(partition.partition_id, manifest_digest) is None
        ]

    def _local_config(self, partition_id: int) -> CaseStudyConfig:
        partition = self.layout.get(partition_id)
        nx, ny = partition.solve_shape
        global_mesh = self.config.mesh
        local_mesh = MeshConfig(
            nx=nx,
            ny=ny,
            base_pixel_size_mm=global_mesh.base_pixel_size_mm,
            scale_factor=global_mesh.scale_factor,
        )
        return replace(self.config, mesh=local_mesh)

    def solve_partition(self, partition_id: int, *, force: bool = False) -> dict[str, Any]:
        """Solve and atomically record one partition, or reuse its valid result."""

        partition = self.layout.get(partition_id)
        manifest_digest = self.prepare()
        completed = self._completed_status(partition_id, manifest_digest)
        if completed is not None and not force:
            LOGGER.info("Partition %s already complete", partition_id)
            return completed

        LOGGER.info("Solving partition %s of %s", partition_id, self.layout.count)
        result = run_case_study(
            self._local_config(partition_id),
            displacement_x_mm=extract_partition_field(
                self.displacement_x_mm,
                layout=self.layout,
                partition=partition,
                location="node",
            ),
            displacement_y_mm=extract_partition_field(
                self.displacement_y_mm,
                layout=self.layout,
                partition=partition,
                location="node",
            ),
            yield_stress_mpa=extract_partition_field(
                self.yield_stress_mpa,
                layout=self.layout,
                partition=partition,
                location="element",
            ),
            hardening_coefficient_mpa=extract_partition_field(
                self.hardening_coefficient_mpa,
                layout=self.layout,
                partition=partition,
                location="element",
            ),
        )
        write_started_at = time.perf_counter()
        outputs: dict[str, str] = {}
        for field_name, (attribute, _location) in RESULT_FIELDS.items():
            path = self._result_path(partition_id, field_name)
            _atomic_save_array(path, getattr(result, attribute))
            outputs[field_name] = _fingerprint_file(path)
        status = {
            "complete": True,
            "diagnostics": {
                **(asdict(result.diagnostics) if result.diagnostics else {}),
                "write_seconds": time.perf_counter() - write_started_at,
            },
            "manifest_sha256": manifest_digest,
            "partition_id": partition_id,
            "outputs": outputs,
        }
        _atomic_write_text(self._status_path(partition_id), _canonical_json(status))
        return status

    def solve_pending(self) -> list[int]:
        """Solve every incomplete partition and return the IDs processed."""

        pending = self.pending_partition_ids()
        for partition_id in pending:
            self.solve_partition(partition_id)
        return pending

    def stitch(self, field_name: str, *, output_path: str | Path | None = None) -> np.memmap:
        """Stitch a complete result field into one global memory-mapped array."""

        if field_name not in RESULT_FIELDS:
            raise KeyError(f"unknown result field {field_name!r}")
        pending = self.pending_partition_ids()
        if pending:
            raise RuntimeError(f"cannot stitch incomplete partitions: {pending}")
        _attribute, location = RESULT_FIELDS[field_name]
        files = {
            partition.partition_id: self._result_path(partition.partition_id, field_name)
            for partition in self.layout
        }
        destination = (
            Path(output_path)
            if output_path is not None
            else self.output_directory / "global" / f"{field_name}.npy"
        )
        return stitch_partition_files(
            self.layout,
            files,
            location=location,
            output_path=destination,
        )
