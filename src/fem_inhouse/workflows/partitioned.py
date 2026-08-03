"""Resumable partition-by-partition reconstruction workflow."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

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
SNAPSHOT_FIELDS: dict[str, tuple[str, FieldLocation]] = {
    "U": ("displacement_mm", "node"),
    "E": ("total_strain", "element"),
    "PEEQ": ("equivalent_plastic_strain", "element"),
}
BASE_RESULT_FIELDS: dict[str, tuple[str, FieldLocation]] = {
    "U": ("displacement_mm", "node"),
    "S": ("stress_mpa", "element"),
    "S_3D": ("stress_tensor_mpa", "element"),
    "E": ("total_strain", "element"),
    "E_3D": ("total_strain_tensor", "element"),
    "EE_3D": ("elastic_strain_tensor", "element"),
    "PE": ("plastic_strain", "element"),
    "PE_3D": ("plastic_strain_tensor", "element"),
    "PEEQ": ("equivalent_plastic_strain", "element"),
    "S33_RESIDUAL_MPA": ("plane_stress_residual_mpa", "element"),
    "PLANE_STRESS_RESIDUAL_MPA": (
        "plane_stress_residual_vector_mpa",
        "element",
    ),
    "RF": ("reaction_force", "node"),
}
REDUCED_RESULT_FIELDS: dict[str, tuple[str, FieldLocation]] = {
    "HOURGLASS_ENERGY_BY_ELEMENT": ("hourglass_energy_by_element", "element"),
}
NONLOCAL_RESULT_FIELDS: dict[str, tuple[str, FieldLocation]] = {
    "PEEQ_NONLOCAL": ("nonlocal_equivalent_plastic_strain", "element"),
    "PEEQ_MISMATCH": ("equivalent_plastic_strain_mismatch", "element"),
    "NONLOCAL_HARDENING_MPA": ("nonlocal_hardening_mpa", "element"),
    "YIELD_SURFACE_RADIUS_MPA": ("yield_surface_radius_mpa", "element"),
    "NONLOCAL_RESIDUAL": ("nonlocal_residual", "element"),
}
RESULT_FIELDS = {**BASE_RESULT_FIELDS, **NONLOCAL_RESULT_FIELDS}
RESULT_FIELD_METADATA: dict[str, dict[str, str]] = {
    "U": {"components": "[u1, u2]", "unit": "mm"},
    "S": {"components": "[s11, s22, s12]", "unit": "MPa"},
    "S_3D": {"components": "symmetric 3x3 Cauchy stress", "unit": "MPa"},
    "E": {"components": "[e11, e22, gamma12]", "unit": "1"},
    "E_3D": {"components": "symmetric 3x3 total strain", "unit": "1"},
    "EE_3D": {"components": "symmetric 3x3 elastic strain", "unit": "1"},
    "PE": {"components": "[ep11, ep22, gamma_p12]", "unit": "1"},
    "PE_3D": {"components": "symmetric 3x3 plastic strain", "unit": "1"},
    "PEEQ": {"components": "accumulated equivalent plastic strain", "unit": "1"},
    "S33_RESIDUAL_MPA": {"components": "s33", "unit": "MPa"},
    "PLANE_STRESS_RESIDUAL_MPA": {
        "components": "[s33, s13, s23]",
        "unit": "MPa",
    },
    "RF": {"components": "[r1, r2]", "unit": "N for implicit 1 mm thickness"},
    "HOURGLASS_ENERGY_BY_ELEMENT": {
        "components": "numerical stabilisation energy per element",
        "unit": "N mm for implicit 1 mm thickness",
    },
    "PEEQ_NONLOCAL": {
        "components": "element-centred micromorphic equivalent plastic strain chi",
        "unit": "1",
    },
    "PEEQ_MISMATCH": {"components": "p_e - chi", "unit": "1"},
    "NONLOCAL_HARDENING_MPA": {
        "components": "H_chi (p_e - chi)",
        "unit": "MPa",
    },
    "YIELD_SURFACE_RADIUS_MPA": {
        "components": "final element-averaged yield surface radius",
        "unit": "MPa",
    },
    "NONLOCAL_RESIDUAL": {
        "components": "chi - Helmholtz(p_e)",
        "unit": "1",
    },
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
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_array(path: Path, values: NDArray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.tmp.npy")
    try:
        np.save(temporary, values)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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
    snapshot_fractions: tuple[float, ...] = ()
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
        snapshot_fractions = tuple(float(value) for value in self.snapshot_fractions)
        if any(not np.isfinite(value) or not 0.0 < value <= 1.0 for value in snapshot_fractions):
            raise ValueError("snapshot_fractions must be finite and lie in (0, 1]")
        if len(set(snapshot_fractions)) != len(snapshot_fractions):
            raise ValueError("snapshot_fractions must be unique")
        self.snapshot_fractions = tuple(sorted(snapshot_fractions))
        self.output_directory = Path(self.output_directory)

    @property
    def manifest_path(self) -> Path:
        return self.output_directory / "manifest.json"

    @property
    def _result_fields(self) -> dict[str, tuple[str, FieldLocation]]:
        fields = dict(BASE_RESULT_FIELDS)
        if self.config.solver.element_formulation == "cps4r":
            fields.update(REDUCED_RESULT_FIELDS)
        if self.config.nonlocal_plasticity.enabled:
            fields.update(NONLOCAL_RESULT_FIELDS)
        return fields

    def _crystal_provenance(self) -> dict[str, Any] | None:
        """Section 5. Where this campaign's crystal parameters came from.

        Absent for a J2 campaign, which has no selectable parameter set. Present
        and complete for a crystal one, so that a reader of the archive can tell
        an identified value from a transposed one without leaving the manifest.
        """

        from fem_inhouse.core.mfront_behaviours import MFRONT_BEHAVIOURS
        from fem_inhouse.core.srix_parameters import srix_provenance

        identifier = self.config.solver.mfront_behaviour_id
        if identifier is None:
            return None
        behaviour = MFRONT_BEHAVIOURS.get(identifier)
        if behaviour.parameter_registry != "srix":
            return None
        options = self.config.solver.constitutive_options
        return srix_provenance(
            parameter_set=options.get("parameter_set"),
            explicit=options.get("parameters"),
            mfront_source=Path("mfront") / f"{behaviour.tridimensional_behaviour}.mfront",
        )

    def _manifest_data(self) -> dict[str, Any]:
        result_fields = self._result_fields
        crystal = self._crystal_provenance()
        return {
            "schema_version": 1,
            "software": {
                "name": "kinematics-driven-316l-strain-reconstruction",
                "version": __version__,
                "source_sha256": _source_fingerprint(),
            },
            **({"crystal_parameters": crystal} if crystal is not None else {}),
            "config": asdict(self.config),
            "layout": self.layout.as_dict(),
            "inputs": {
                "displacement_x_mm": fingerprint_array(self.displacement_x_mm),
                "displacement_y_mm": fingerprint_array(self.displacement_y_mm),
                "yield_stress_mpa": fingerprint_array(self.yield_stress_mpa),
                "hardening_coefficient_mpa": fingerprint_array(self.hardening_coefficient_mpa),
            },
            "result_fields": sorted(result_fields),
            "result_field_metadata": {name: RESULT_FIELD_METADATA[name] for name in result_fields},
            "snapshots": {
                "fractions": list(self.snapshot_fractions),
                "fields": sorted(SNAPSHOT_FIELDS),
                "purpose": "load-history identifiability diagnostics",
            },
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

    def _snapshot_path(
        self,
        partition_id: int,
        fraction: float,
        field_name: str,
    ) -> Path:
        fraction_id = f"{fraction:.6f}".replace(".", "p")
        return (
            self._partition_directory(partition_id)
            / "snapshots"
            / fraction_id
            / f"{field_name}.npy"
        )

    def _completed_status(self, partition_id: int, manifest_digest: str) -> dict[str, Any] | None:
        status_path = self._status_path(partition_id)
        if not status_path.exists():
            return None
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if not status.get("complete") or status.get("manifest_sha256") != manifest_digest:
                return None
            expected_outputs = status["outputs"]
            for field_name in self._result_fields:
                path = self._result_path(partition_id, field_name)
                if not path.is_file() or _fingerprint_file(path) != expected_outputs[field_name]:
                    return None
            expected_snapshots = status.get("snapshots", {})
            for fraction in self.snapshot_fractions:
                fraction_id = f"{fraction:.6f}"
                snapshot_outputs = expected_snapshots[fraction_id]
                for field_name in SNAPSHOT_FIELDS:
                    path = self._snapshot_path(partition_id, fraction, field_name)
                    if (
                        not path.is_file()
                        or _fingerprint_file(path) != snapshot_outputs[field_name]
                    ):
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
        snapshot_arguments: dict[str, Any] = (
            {"snapshots": self.snapshot_fractions} if self.snapshot_fractions else {}
        )
        result = run_case_study(
            self._local_config(partition_id),
            # Follow the caller's logging level rather than a separate flag.
            # `--verbose` used to raise the level without ever reaching the
            # solver, so a partition solve stayed silent from the first line to
            # the last and a multi-hour campaign could not be followed.
            verbose=LOGGER.isEnabledFor(logging.INFO),
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
            **snapshot_arguments,
        )
        write_started_at = time.perf_counter()
        outputs: dict[str, str] = {}
        for field_name, (attribute, _location) in self._result_fields.items():
            path = self._result_path(partition_id, field_name)
            values = getattr(result, attribute)
            if (
                values is None
                and field_name == "PLANE_STRESS_RESIDUAL_MPA"
                and result.plane_stress_residual_mpa is not None
            ):
                scalar = np.asarray(result.plane_stress_residual_mpa)
                values = np.zeros((*scalar.shape, 3), dtype=scalar.dtype)
                values[..., 0] = scalar
            if values is None:
                raise RuntimeError(
                    f"completed solve did not provide required result field {field_name}"
                )
            _atomic_save_array(path, values)
            outputs[field_name] = _fingerprint_file(path)
        snapshot_outputs: dict[str, dict[str, str]] = {}
        for fraction in self.snapshot_fractions:
            try:
                frame = result.frames[fraction]
            except KeyError as error:
                raise RuntimeError(
                    f"completed solve did not provide requested snapshot {fraction}"
                ) from error
            fraction_outputs: dict[str, str] = {}
            for field_name, (attribute, _location) in SNAPSHOT_FIELDS.items():
                path = self._snapshot_path(partition_id, fraction, field_name)
                _atomic_save_array(path, np.asarray(getattr(frame, attribute)))
                fraction_outputs[field_name] = _fingerprint_file(path)
            snapshot_outputs[f"{fraction:.6f}"] = fraction_outputs
        status = {
            "complete": True,
            "diagnostics": {
                **(asdict(result.diagnostics) if result.diagnostics else {}),
                "write_seconds": time.perf_counter() - write_started_at,
            },
            "manifest_sha256": manifest_digest,
            "partition_id": partition_id,
            "outputs": outputs,
            "snapshots": snapshot_outputs,
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

        if field_name not in self._result_fields:
            raise KeyError(f"unknown result field {field_name!r}")
        pending = self.pending_partition_ids()
        if pending:
            raise RuntimeError(f"cannot stitch incomplete partitions: {pending}")
        _attribute, location = self._result_fields[field_name]
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
