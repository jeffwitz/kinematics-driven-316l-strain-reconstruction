"""Optional MFront/MGIS constitutive backend for plane-stress material points."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.tensor_reconstruction import (
    FullTensorState,
    engineering_strain_2d_to_tensor,
    kelvin_plane_stress_to_tensor,
    reconstruct_native_plane_stress_state,
    tensor_to_engineering_strain_2d,
    tensor_to_engineering_stress_2d,
    tensor_to_kelvin_plane_stress,
)

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

    tensor = engineering_strain_2d_to_tensor(strain, 0.0)
    return tensor_to_kelvin_plane_stress(tensor, quantity="strain")


def kelvin_strain_to_engineering(strain: ArrayLike) -> NDArray:
    """Convert a MFront 2D Kelvin strain to ``[e11, e22, gamma12]``."""

    return tensor_to_engineering_strain_2d(
        kelvin_plane_stress_to_tensor(strain, quantity="strain")
    )


def kelvin_stress_to_engineering(stress: ArrayLike) -> NDArray:
    """Convert a MFront 2D Kelvin stress to ``[s11, s22, s12]``."""

    return tensor_to_engineering_stress_2d(
        kelvin_plane_stress_to_tensor(stress, quantity="stress")
    )


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


def _variable_offset(
    mgis: Any,
    variables: Any,
    name: str,
    hypothesis: Any,
    *,
    expected_size: int,
    required: bool = True,
) -> int | None:
    matches = [variable for variable in variables if variable.name == name]
    if not matches:
        if required:
            raise MFrontUnavailableError(
                f"MFront behaviour does not expose required variable {name!r}"
            )
        return None
    if len(matches) != 1:
        raise MFrontUnavailableError(f"MFront behaviour exposes {name!r} more than once")
    actual_size = int(mgis.getVariableSize(matches[0], hypothesis))
    if actual_size != expected_size:
        raise MFrontUnavailableError(
            f"MFront variable {name!r} has size {actual_size}, expected {expected_size}"
        )
    return int(mgis.getVariableOffset(variables, name, hypothesis))


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
        _variable_offset(
            mgis,
            behaviour.gradients,
            "Strain",
            hypothesis,
            expected_size=4,
        )
        _variable_offset(
            mgis,
            behaviour.thermodynamic_forces,
            "Stress",
            hypothesis,
            expected_size=4,
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
        equivalent_plastic_strain_offset = _variable_offset(
            mgis,
            behaviour.isvs,
            "EquivalentPlasticStrain",
            hypothesis,
            expected_size=1,
        )
        yield_surface_radius_offset = _variable_offset(
            mgis,
            behaviour.isvs,
            "YieldSurfaceRadius",
            hypothesis,
            expected_size=1,
        )
        elastic_strain_offset = _variable_offset(
            mgis,
            behaviour.isvs,
            "ElasticStrain",
            hypothesis,
            expected_size=4,
        )
        self._axial_strain_offset = _variable_offset(
            mgis,
            behaviour.isvs,
            "AxialStrain",
            hypothesis,
            expected_size=1,
            required=False,
        )
        assert equivalent_plastic_strain_offset is not None
        assert yield_surface_radius_offset is not None
        assert elastic_strain_offset is not None
        self._equivalent_plastic_strain_offset = equivalent_plastic_strain_offset
        self._yield_surface_radius_offset = yield_surface_radius_offset
        self._elastic_strain_offset = elastic_strain_offset
        self._has_trial_state = False

    @property
    def point_count(self) -> int:
        """Number of independent material points in the batch."""

        return self._point_count

    @property
    def has_native_plane_stress_state(self) -> bool:
        """Whether MGIS exposes the native axial strain used by plane stress."""

        return self._axial_strain_offset is not None

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

    def current_full_tensor_state(self) -> FullTensorState | None:
        """Extract the latest successful native MFront trial without committing it."""

        if not self._has_trial_state:
            raise RuntimeError("no successful MFront trial state is available")
        if self._axial_strain_offset is None:
            return None
        total_kelvin = self._manager.s1.gradients.copy()
        total_kelvin[:, 2] = self._manager.s1.internal_state_variables[
            :, self._axial_strain_offset
        ]
        elastic_offset = self._elastic_strain_offset
        elastic_kelvin = self._manager.s1.internal_state_variables[
            :, elastic_offset : elastic_offset + 4
        ].copy()
        stress_kelvin = self._manager.s1.thermodynamic_forces.copy()
        return reconstruct_native_plane_stress_state(
            total_kelvin,
            elastic_kelvin,
            stress_kelvin,
        )

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
