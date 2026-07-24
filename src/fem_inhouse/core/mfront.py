"""Optional MFront/MGIS constitutive backend for plane-stress material points."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

_SQRT_TWO = np.sqrt(2.0)
_PLANE_STRESS_COMPONENTS = np.array([0, 1, 3])
_KELVIN_TO_ENGINEERING_STRESS_SCALE = np.array([1.0, 1.0, 1.0 / _SQRT_TWO])
_ENGINEERING_TO_KELVIN_STRAIN_SCALE = np.array([1.0, 1.0, 1.0 / _SQRT_TWO])


class MFrontUnavailableError(RuntimeError):
    """Raised when the optional TFEL/MGIS runtime cannot be loaded."""


class MFrontIntegrationError(RuntimeError):
    """Raised when MFront fails to integrate a material-point batch."""


@dataclass(frozen=True, slots=True)
class MFrontIntegrationResult:
    """Engineering-component result of one MFront material-point evaluation."""

    stress_mpa: NDArray
    plastic_strain: NDArray
    equivalent_plastic_strain: NDArray
    yield_surface_radius_mpa: NDArray
    consistent_tangent_mpa: NDArray | None


def engineering_strain_to_kelvin(strain: ArrayLike) -> NDArray:
    """Convert ``[e11, e22, gamma12]`` to MFront's 2D Kelvin stensor."""

    values = np.asarray(strain, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("engineering strain must have a trailing dimension of 3")
    kelvin = np.zeros((*values.shape[:-1], 4), dtype=float)
    kelvin[..., 0] = values[..., 0]
    kelvin[..., 1] = values[..., 1]
    kelvin[..., 3] = values[..., 2] / _SQRT_TWO
    return kelvin


def kelvin_strain_to_engineering(strain: ArrayLike) -> NDArray:
    """Convert a MFront 2D Kelvin strain to ``[e11, e22, gamma12]``."""

    values = np.asarray(strain, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 4:
        raise ValueError("Kelvin strain must have a trailing dimension of 4")
    return np.stack(
        (values[..., 0], values[..., 1], _SQRT_TWO * values[..., 3]),
        axis=-1,
    )


def kelvin_stress_to_engineering(stress: ArrayLike) -> NDArray:
    """Convert a MFront 2D Kelvin stress to ``[s11, s22, s12]``."""

    values = np.asarray(stress, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 4:
        raise ValueError("Kelvin stress must have a trailing dimension of 4")
    selected = np.take(values, _PLANE_STRESS_COMPONENTS, axis=-1)
    return selected * _KELVIN_TO_ENGINEERING_STRESS_SCALE


def kelvin_tangent_to_engineering(tangent: ArrayLike) -> NDArray:
    """Convert a 4x4 Kelvin tangent to engineering plane-stress components."""

    values = np.asarray(tangent, dtype=float)
    if values.ndim < 2 or values.shape[-2:] != (4, 4):
        raise ValueError("Kelvin tangent must have trailing dimensions (4, 4)")
    selected = np.take(values, _PLANE_STRESS_COMPONENTS, axis=-2)
    selected = np.take(selected, _PLANE_STRESS_COMPONENTS, axis=-1)
    return (
        selected
        * _KELVIN_TO_ENGINEERING_STRESS_SCALE[:, None]
        * _ENGINEERING_TO_KELVIN_STRAIN_SCALE[None, :]
    )


def _load_mgis() -> Any:
    try:
        return import_module("mgis.behaviour")
    except (ImportError, OSError) as error:
        raise MFrontUnavailableError(
            "MGIS Python bindings are unavailable; source the TFEL environment "
            "before starting Python"
        ) from error


def _load_mgis_root() -> Any:
    try:
        return import_module("mgis")
    except (ImportError, OSError) as error:
        raise MFrontUnavailableError(
            "MGIS Python bindings are unavailable; source the TFEL environment "
            "before starting Python"
        ) from error


def _broadcast_material_properties(
    initial_yield_stress_mpa: ArrayLike,
    hardening_coefficient_mpa: ArrayLike,
    hardening_exponent: ArrayLike,
) -> tuple[NDArray, NDArray, NDArray]:
    values = np.broadcast_arrays(
        np.asarray(initial_yield_stress_mpa, dtype=float),
        np.asarray(hardening_coefficient_mpa, dtype=float),
        np.asarray(hardening_exponent, dtype=float),
    )
    flattened = tuple(np.ravel(value).copy() for value in values)
    yield_stress, coefficient, exponent = flattened
    if yield_stress.size == 0:
        raise ValueError("at least one material point is required")
    if not all(np.isfinite(value).all() for value in flattened):
        raise ValueError("MFront material properties must be finite")
    if np.any(yield_stress <= 0):
        raise ValueError("initial yield stress must be positive")
    if np.any(coefficient < 0):
        raise ValueError("hardening coefficient must be non-negative")
    if np.any(exponent <= 0):
        raise ValueError("hardening exponent must be positive")
    return yield_stress, coefficient, exponent


class MFrontMaterialPointBatch:
    """Stateful MGIS bridge for independent plane-stress material points.

    Evaluations are non-committing by default so that a finite-element Newton
    loop can retry several trial strains from the same converged state.
    """

    def __init__(
        self,
        library_path: str | Path,
        initial_yield_stress_mpa: ArrayLike,
        hardening_coefficient_mpa: ArrayLike,
        hardening_exponent: ArrayLike,
        *,
        temperature_k: float = 293.15,
        thread_count: int = 1,
    ) -> None:
        library = Path(library_path)
        if not library.is_file():
            raise FileNotFoundError(f"MFront behaviour library not found: {library}")
        if not np.isfinite(temperature_k) or temperature_k <= 0:
            raise ValueError("temperature_k must be finite and positive")
        if isinstance(thread_count, bool) or not isinstance(thread_count, int):
            raise TypeError("thread_count must be an integer")
        if thread_count < 1:
            raise ValueError("thread_count must be at least 1")

        yield_stress, coefficient, exponent = _broadcast_material_properties(
            initial_yield_stress_mpa,
            hardening_coefficient_mpa,
            hardening_exponent,
        )
        mgis = _load_mgis()
        hypothesis = mgis.Hypothesis.PlaneStress
        behaviour = mgis.load(
            str(library.resolve()),
            "PixelLudwikJ2Plasticity",
            hypothesis,
        )
        manager = mgis.MaterialDataManager(behaviour, yield_stress.size)

        material_values = {
            "InitialYieldStress": yield_stress,
            "HardeningCoefficient": coefficient,
            "HardeningExponent": exponent,
        }
        storage_mode = mgis.MaterialStateManagerStorageMode.ExternalStorage
        for name, property_values in material_values.items():
            mgis.setMaterialProperty(manager.s0, name, property_values, storage_mode)
            mgis.setMaterialProperty(manager.s1, name, property_values, storage_mode)

        temperature_values = np.full(yield_stress.size, temperature_k)
        mgis.setExternalStateVariable(
            manager.s0,
            "Temperature",
            temperature_values,
            storage_mode,
        )
        mgis.setExternalStateVariable(
            manager.s1,
            "Temperature",
            temperature_values,
            storage_mode,
        )

        self._mgis = mgis
        self._behaviour = behaviour
        self._manager = manager
        self._point_count = yield_stress.size
        self._material_values = material_values
        self._temperature_values = temperature_values
        self._thread_pool = _load_mgis_root().ThreadPool(thread_count) if thread_count > 1 else None
        self._equivalent_plastic_strain_offset = mgis.getVariableOffset(
            behaviour.isvs,
            "EquivalentPlasticStrain",
            hypothesis,
        )
        self._yield_surface_radius_offset = mgis.getVariableOffset(
            behaviour.isvs,
            "YieldSurfaceRadius",
            hypothesis,
        )
        self._elastic_strain_offset = mgis.getVariableOffset(
            behaviour.isvs,
            "ElasticStrain",
            hypothesis,
        )
        self._has_trial_state = False

    @property
    def point_count(self) -> int:
        """Number of independent material points in the batch."""

        return self._point_count

    def evaluate(
        self,
        total_strain: ArrayLike,
        *,
        time_increment: float = 1.0,
        consistent_tangent: bool = True,
        commit: bool = False,
    ) -> MFrontIntegrationResult:
        """Integrate a trial total strain from the last committed state."""

        if not np.isfinite(time_increment) or time_increment <= 0:
            raise ValueError("time_increment must be finite and positive")
        strain = np.asarray(total_strain, dtype=float)
        expected_shape = (self._point_count, 3)
        if strain.shape != expected_shape:
            raise ValueError(f"total_strain must have shape {expected_shape}")
        if not np.isfinite(strain).all():
            raise ValueError("total_strain must be finite")

        # Newton iterations must always start from the last converged state,
        # never from the previous (uncommitted) trial.
        if self._has_trial_state:
            self._mgis.revert(self._manager)
            self._has_trial_state = False

        total_kelvin = engineering_strain_to_kelvin(strain)
        self._manager.s1.gradients[:, :] = total_kelvin
        integration_type = (
            self._mgis.IntegrationType.IntegrationWithConsistentTangentOperator
            if consistent_tangent
            else self._mgis.IntegrationType.IntegrationWithoutTangentOperator
        )
        if self._thread_pool is None:
            status = self._mgis.integrate(
                self._manager,
                integration_type,
                float(time_increment),
                0,
                self._point_count,
            )
        else:
            status = self._mgis.integrate(
                self._thread_pool,
                self._manager,
                integration_type,
                float(time_increment),
            )
        if status != 1:
            self._mgis.revert(self._manager)
            self._has_trial_state = False
            raise MFrontIntegrationError(f"MFront integration failed with status {status}")

        stress = kelvin_stress_to_engineering(
            self._manager.s1.thermodynamic_forces,
        ).copy()
        elastic_offset = self._elastic_strain_offset
        elastic_kelvin = self._manager.s1.internal_state_variables[
            :, elastic_offset : elastic_offset + 4
        ]
        plastic_strain = kelvin_strain_to_engineering(total_kelvin - elastic_kelvin).copy()
        equivalent_plastic_strain = self._manager.s1.internal_state_variables[
            :, self._equivalent_plastic_strain_offset
        ].copy()
        yield_surface_radius = self._manager.s1.internal_state_variables[
            :, self._yield_surface_radius_offset
        ].copy()
        tangent = (
            kelvin_tangent_to_engineering(self._manager.K).copy() if consistent_tangent else None
        )
        result = MFrontIntegrationResult(
            stress_mpa=stress,
            plastic_strain=plastic_strain,
            equivalent_plastic_strain=equivalent_plastic_strain,
            yield_surface_radius_mpa=yield_surface_radius,
            consistent_tangent_mpa=tangent,
        )
        self._has_trial_state = True
        if commit:
            self.commit()
        return result

    def commit(self) -> None:
        """Commit the latest successful trial state."""

        if not self._has_trial_state:
            raise RuntimeError("no successful MFront trial state to commit")
        self._mgis.update(self._manager)
        self._has_trial_state = False

    def revert(self) -> None:
        """Discard the latest trial and restore the committed state."""

        self._mgis.revert(self._manager)
        self._has_trial_state = False
