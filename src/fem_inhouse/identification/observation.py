"""Recorded DIC observation operator shared by identification fidelities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class DICObservationOperatorConfig:
    """Versioned definition of the FEM-to-DIC measurement operator.

    The current case study uses coincident structured grids. Unsupported
    interpolation or filter names fail explicitly so metadata can never claim
    that an operation was applied when it was not.
    """

    schema_version: int = 1
    strain_measure: str = "historical_plane_stress_evm_from_displacement"
    support: str = "element_centres"
    grid_mapping: Literal["identity", "coincident-node-stride"] = "identity"
    grid_reduction: int = 1
    spatial_filter: Literal["none"] = "none"
    missing_value_policy: Literal["finite-intersection"] = "finite-intersection"
    use_core_only: bool = True
    displacement_unit: Literal["mm"] = "mm"
    strain_unit: Literal["dimensionless"] = "dimensionless"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported observation-operator schema version")
        if self.strain_measure != "historical_plane_stress_evm_from_displacement":
            raise ValueError("unsupported strain measure")
        if self.support != "element_centres":
            raise ValueError("unsupported observation support")
        if self.grid_reduction < 1:
            raise ValueError("grid_reduction must be at least one")
        if self.grid_mapping == "identity" and self.grid_reduction != 1:
            raise ValueError("identity mapping requires grid_reduction == 1")
        if self.grid_mapping == "coincident-node-stride" and self.grid_reduction == 1:
            raise ValueError("coincident-node-stride requires grid_reduction > 1")

    def as_dict(self) -> dict[str, Any]:
        """Return the complete, serializable operator definition."""

        return asdict(self)

    def fingerprint(self) -> str:
        """Return a stable hash suitable for campaign cache keys."""

        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ObservationResult:
    """Observed field and its exact valid-value support."""

    element_field: FloatArray
    valid_mask: BoolArray
    spacing_x_mm: float
    spacing_y_mm: float
    operator_sha256: str


@dataclass(frozen=True, slots=True)
class DICObservationOperator:
    """Apply the recorded DIC measurement operator to nodal displacements."""

    config: DICObservationOperatorConfig
    poisson_ratio: float

    def observe_displacement(
        self,
        displacement_mm: ArrayLike,
        *,
        spacing_x_mm: float,
        spacing_y_mm: float,
        core_slice: tuple[slice, slice] | None = None,
        mask: ArrayLike | None = None,
    ) -> ObservationResult:
        """Reconstruct EVM, then apply the declared support and core mask."""

        # Imported lazily because the workflows package also exports this
        # observation operator. Keeping the numerical operator in one place
        # avoids formula duplication without creating an import cycle.
        from fem_inhouse.workflows.nonlocality_diagnostic import (
            reconstruct_historical_evm,
        )

        displacement = np.asarray(displacement_mm, dtype=np.float64)
        if displacement.ndim != 3 or displacement.shape[-1] != 2:
            raise ValueError("displacement_mm must have shape (nx + 1, ny + 1, 2)")
        if not np.isfinite(displacement).all():
            raise ValueError("displacement_mm must contain only finite values")
        if not np.isfinite(spacing_x_mm) or spacing_x_mm <= 0.0:
            raise ValueError("spacing_x_mm must be finite and positive")
        if not np.isfinite(spacing_y_mm) or spacing_y_mm <= 0.0:
            raise ValueError("spacing_y_mm must be finite and positive")

        factor = self.config.grid_reduction
        if factor > 1:
            if (displacement.shape[0] - 1) % factor or (
                displacement.shape[1] - 1
            ) % factor:
                raise ValueError(
                    "nodal dimensions minus one must be divisible by grid_reduction"
                )
            displacement = displacement[::factor, ::factor, :]
        observed_spacing_x = float(spacing_x_mm) * factor
        observed_spacing_y = float(spacing_y_mm) * factor
        field = reconstruct_historical_evm(
            displacement,
            spacing_x_mm=observed_spacing_x,
            spacing_y_mm=observed_spacing_y,
            poisson_ratio=self.poisson_ratio,
        )

        valid = np.isfinite(field)
        if mask is not None:
            provided_mask = np.asarray(mask, dtype=bool)
            if factor > 1:
                expected_fine_shape = (
                    field.shape[0] * factor,
                    field.shape[1] * factor,
                )
                if provided_mask.shape == expected_fine_shape:
                    provided_mask = _block_all(provided_mask, factor)
            if provided_mask.shape != field.shape:
                raise ValueError("mask shape is incompatible with the observed element field")
            valid &= provided_mask

        if core_slice is not None:
            if not self.config.use_core_only:
                raise ValueError("core_slice supplied while use_core_only is false")
            reduced_core = _reduce_core_slice(core_slice, factor)
            field = np.asarray(field[reduced_core], dtype=np.float64)
            valid = np.asarray(valid[reduced_core], dtype=bool)
        elif self.config.use_core_only:
            raise ValueError("the configured observation operator requires a core_slice")

        if not valid.any():
            raise ValueError("no valid observed values remain")
        return ObservationResult(
            element_field=np.asarray(field, dtype=np.float64),
            valid_mask=np.asarray(valid, dtype=bool),
            spacing_x_mm=observed_spacing_x,
            spacing_y_mm=observed_spacing_y,
            operator_sha256=self.config.fingerprint(),
        )


def _reduce_core_slice(
    core_slice: tuple[slice, slice],
    factor: int,
) -> tuple[slice, slice]:
    reduced: list[slice] = []
    for axis_slice in core_slice:
        if axis_slice.step not in (None, 1):
            raise ValueError("core slices must have unit stride")
        if axis_slice.start is None or axis_slice.stop is None:
            raise ValueError("core slices require explicit start and stop")
        if axis_slice.start % factor or axis_slice.stop % factor:
            raise ValueError("core bounds must be divisible by grid_reduction")
        reduced.append(slice(axis_slice.start // factor, axis_slice.stop // factor))
    return reduced[0], reduced[1]


def _block_all(mask: BoolArray, factor: int) -> BoolArray:
    nx, ny = mask.shape
    if nx % factor or ny % factor:
        raise ValueError("fine mask dimensions must be divisible by grid_reduction")
    return np.asarray(
        mask.reshape(nx // factor, factor, ny // factor, factor).all(axis=(1, 3)),
        dtype=bool,
    )
