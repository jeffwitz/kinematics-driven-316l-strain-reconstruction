"""Optional MFront/MGIS constitutive backend for plane-stress material points."""

from __future__ import annotations

import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.linear_solver import LinearSystemMatrixType
from fem_inhouse.core.plane_stress_material import (
    ConstitutiveIntegrationError,
    ConstitutiveTrial,
    InPlaneConstitutiveTrial,
    LocalPlaneStressConvergenceError,
    PlaneStressBatchStatistics,
)
from fem_inhouse.core.tensor_reconstruction import (
    FullTensorState,
    kelvin_3d_to_tensor,
    kelvin_plane_stress_to_tensor,
    reconstruct_native_plane_stress_state,
    tensor_to_engineering_strain_2d,
    tensor_to_engineering_stress_2d,
)

_SQRT_TWO = np.sqrt(2.0)
_PLANE_STRESS_COMPONENTS = np.array([0, 1, 3])
_TRANSVERSE_COMPONENTS_3D = np.array([2, 4, 5])
_KELVIN_TO_ENGINEERING_STRESS_SCALE = np.array([1.0, 1.0, 1.0 / _SQRT_TWO])
_ENGINEERING_TO_KELVIN_STRAIN_SCALE = np.array([1.0, 1.0, 1.0 / _SQRT_TWO])
_SYMMETRIC_POSITIVE_DEFINITE_J2_BEHAVIOURS = frozenset(
    {
        "PixelLudwikJ2Plasticity",
        "PixelMicromorphicLudwikJ2Plasticity",
        "PixelLudwikJ2Plasticity3D",
        "PixelMicromorphicLudwikJ2Plasticity3D",
    }
)


class MFrontUnavailableError(RuntimeError):
    """Raised when the optional TFEL/MGIS runtime cannot be loaded."""


class MFrontIntegrationError(ConstitutiveIntegrationError):
    """Raised when MFront fails to integrate a material-point batch."""


@dataclass(frozen=True, slots=True)
class MFrontIntegrationResult:
    """Engineering-component result of one MFront material-point evaluation."""

    stress_mpa: NDArray
    plastic_strain: NDArray
    equivalent_plastic_strain: NDArray
    yield_surface_radius_mpa: NDArray
    consistent_tangent_mpa: NDArray | None


@dataclass(frozen=True, slots=True)
class MFrontTimingStatistics:
    """Accumulated wall times for the native MGIS constitutive bridge."""

    integration_without_tangent_seconds: float = 0.0
    integration_with_tangent_seconds: float = 0.0
    kelvin_conversion_seconds: float = 0.0
    tensor_reconstruction_seconds: float = 0.0
    integration_without_tangent_calls: int = 0
    integration_with_tangent_calls: int = 0
    tensor_reconstruction_calls: int = 0


def engineering_strain_to_kelvin(
    strain: ArrayLike,
    *,
    out: NDArray | None = None,
) -> NDArray:
    """Convert ``[e11, e22, gamma12]`` to MFront's 2D Kelvin stensor."""

    values = np.asarray(strain, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("engineering strain must have trailing dimension 3")
    expected_shape = (*values.shape[:-1], 4)
    result = np.empty(expected_shape, dtype=float) if out is None else out
    if result.shape != expected_shape:
        raise ValueError(f"out must have shape {expected_shape}")
    result[..., 0] = values[..., 0]
    result[..., 1] = values[..., 1]
    result[..., 2] = 0.0
    result[..., 3] = values[..., 2] / _SQRT_TWO
    return result


def kelvin_strain_to_engineering(strain: ArrayLike) -> NDArray:
    """Convert a MFront 2D Kelvin strain to ``[e11, e22, gamma12]``."""

    return tensor_to_engineering_strain_2d(kelvin_plane_stress_to_tensor(strain, quantity="strain"))


def kelvin_stress_to_engineering(stress: ArrayLike) -> NDArray:
    """Convert a MFront 2D Kelvin stress to ``[s11, s22, s12]``."""

    return tensor_to_engineering_stress_2d(kelvin_plane_stress_to_tensor(stress, quantity="stress"))


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


def _broadcast_point_property(
    values: ArrayLike,
    point_count: int,
    *,
    name: str,
    nonnegative: bool = False,
) -> NDArray:
    """Broadcast one scalar material-point property to external storage."""

    array = np.asarray(values, dtype=float)
    try:
        broadcast = np.broadcast_to(array, (point_count,)).copy()
    except ValueError as error:
        raise ValueError(f"{name} must be scalar or have shape {(point_count,)}") from error
    if not np.isfinite(broadcast).all():
        raise ValueError(f"{name} must be finite")
    if nonnegative and np.any(broadcast < 0):
        raise ValueError(f"{name} must be nonnegative")
    return broadcast


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
        behaviour_name: str = "PixelLudwikJ2Plasticity",
        micromorphic_coupling_modulus_mpa: ArrayLike | None = None,
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
            behaviour_name,
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
        nonlocal_values_s0: NDArray | None = None
        nonlocal_values_s1: NDArray | None = None
        committed_nonlocal_values: NDArray | None = None
        trial_nonlocal_values: NDArray | None = None
        if micromorphic_coupling_modulus_mpa is not None:
            coupling = _broadcast_point_property(
                micromorphic_coupling_modulus_mpa,
                yield_stress.size,
                name="micromorphic_coupling_modulus_mpa",
                nonnegative=True,
            )
            _variable_offset(
                mgis,
                behaviour.mps,
                "MicromorphicCouplingModulus",
                hypothesis,
                expected_size=1,
            )
            _variable_offset(
                mgis,
                behaviour.esvs,
                "NonlocalEquivalentPlasticStrain",
                hypothesis,
                expected_size=1,
            )
            material_values["MicromorphicCouplingModulus"] = coupling
            nonlocal_values_s0 = np.zeros(yield_stress.size)
            nonlocal_values_s1 = np.zeros(yield_stress.size)
            committed_nonlocal_values = np.zeros(yield_stress.size)
            trial_nonlocal_values = np.zeros(yield_stress.size)
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
        if nonlocal_values_s0 is not None and nonlocal_values_s1 is not None:
            mgis.setExternalStateVariable(
                manager.s0,
                "NonlocalEquivalentPlasticStrain",
                nonlocal_values_s0,
                storage_mode,
            )
            mgis.setExternalStateVariable(
                manager.s1,
                "NonlocalEquivalentPlasticStrain",
                nonlocal_values_s1,
                storage_mode,
            )

        self._mgis = mgis
        self._behaviour = behaviour
        self._manager = manager
        self._point_count = yield_stress.size
        self._material_values = material_values
        self._temperature_values = temperature_values
        self._behaviour_name = behaviour_name
        self._nonlocal_values_s0 = nonlocal_values_s0
        self._nonlocal_values_s1 = nonlocal_values_s1
        self._committed_nonlocal_values = committed_nonlocal_values
        self._trial_nonlocal_values = trial_nonlocal_values
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
        self._total_kelvin_buffer = np.empty((self._point_count, 4), dtype=float)
        self._integration_without_tangent_seconds = 0.0
        self._integration_with_tangent_seconds = 0.0
        self._kelvin_conversion_seconds = 0.0
        self._tensor_reconstruction_seconds = 0.0
        self._integration_without_tangent_calls = 0
        self._integration_with_tangent_calls = 0
        self._tensor_reconstruction_calls = 0
        self._has_trial_state = False

    @property
    def point_count(self) -> int:
        """Number of independent material points in the batch."""

        return self._point_count

    @property
    def behaviour_name(self) -> str:
        """Return the exact MFront behaviour selected by this bridge."""

        return self._behaviour_name

    @property
    def linear_system_matrix_type(self) -> LinearSystemMatrixType:
        """Return the verified matrix capability of the selected behaviour."""

        if self._behaviour_name in _SYMMETRIC_POSITIVE_DEFINITE_J2_BEHAVIOURS:
            return "symmetric_positive_definite"
        return "nonsymmetric"

    @property
    def has_native_plane_stress_state(self) -> bool:
        """Whether MGIS exposes the native axial strain used by plane stress."""

        return self._axial_strain_offset is not None

    @property
    def supports_nonlocal_equivalent_plastic_strain(self) -> bool:
        """Whether this behaviour declares the micromorphic external field."""

        return self._trial_nonlocal_values is not None

    @property
    def timing_statistics(self) -> MFrontTimingStatistics:
        """Return immutable accumulated timings for this material batch."""

        return MFrontTimingStatistics(
            integration_without_tangent_seconds=(
                self._integration_without_tangent_seconds
            ),
            integration_with_tangent_seconds=self._integration_with_tangent_seconds,
            kelvin_conversion_seconds=self._kelvin_conversion_seconds,
            tensor_reconstruction_seconds=self._tensor_reconstruction_seconds,
            integration_without_tangent_calls=(
                self._integration_without_tangent_calls
            ),
            integration_with_tangent_calls=self._integration_with_tangent_calls,
            tensor_reconstruction_calls=self._tensor_reconstruction_calls,
        )

    @property
    def committed_nonlocal_equivalent_plastic_strain(self) -> NDArray:
        """Return the committed external field without exposing mutable storage."""

        if self._committed_nonlocal_values is None:
            raise MFrontUnavailableError(
                f"{self._behaviour_name} does not expose NonlocalEquivalentPlasticStrain"
            )
        return self._committed_nonlocal_values.copy()

    def _apply_trial_nonlocal_values(self) -> None:
        if self._trial_nonlocal_values is None:
            return
        assert self._nonlocal_values_s0 is not None
        assert self._nonlocal_values_s1 is not None
        self._nonlocal_values_s0[:] = self._trial_nonlocal_values
        self._nonlocal_values_s1[:] = self._trial_nonlocal_values

    def set_nonlocal_equivalent_plastic_strain(self, values: ArrayLike) -> None:
        """Set the fixed external ``chi`` value for the next trial integration."""

        if self._trial_nonlocal_values is None:
            raise MFrontUnavailableError(
                f"{self._behaviour_name} does not expose NonlocalEquivalentPlasticStrain"
            )
        supplied = np.asarray(values, dtype=float)
        if supplied.shape == (self._point_count,):
            if not np.isfinite(supplied).all() or np.any(supplied < 0):
                raise ValueError(
                    "nonlocal_equivalent_plastic_strain must be finite and nonnegative"
                )
            trial = supplied
        else:
            trial = _broadcast_point_property(
                supplied,
                self._point_count,
                name="nonlocal_equivalent_plastic_strain",
                nonnegative=True,
            )
        if self._has_trial_state:
            self._mgis.revert(self._manager)
            self._has_trial_state = False
        self._trial_nonlocal_values[:] = trial
        self._apply_trial_nonlocal_values()

    def _integrate_trial(
        self,
        total_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool,
    ) -> NDArray:
        """Integrate one trial and return the reusable Kelvin gradient buffer."""

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
        self._apply_trial_nonlocal_values()

        conversion_started = time.perf_counter()
        total_kelvin = engineering_strain_to_kelvin(
            strain,
            out=self._total_kelvin_buffer,
        )
        self._manager.s1.gradients[:, :] = total_kelvin
        self._kelvin_conversion_seconds += time.perf_counter() - conversion_started
        integration_type = (
            self._mgis.IntegrationType.IntegrationWithConsistentTangentOperator
            if consistent_tangent
            else self._mgis.IntegrationType.IntegrationWithoutTangentOperator
        )
        integration_started = time.perf_counter()
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
        integration_seconds = time.perf_counter() - integration_started
        if consistent_tangent:
            self._integration_with_tangent_seconds += integration_seconds
            self._integration_with_tangent_calls += 1
        else:
            self._integration_without_tangent_seconds += integration_seconds
            self._integration_without_tangent_calls += 1
        if status != 1:
            self._mgis.revert(self._manager)
            self._has_trial_state = False
            raise MFrontIntegrationError(f"MFront integration failed with status {status}")
        self._has_trial_state = True
        return total_kelvin

    def evaluate_equivalent_plastic_strain(
        self,
        total_strain: ArrayLike,
        *,
        time_increment: float = 1.0,
    ) -> NDArray:
        """Integrate without tangent and expose only the ephemeral PEEQ view."""

        self._integrate_trial(
            total_strain,
            time_increment=time_increment,
            consistent_tangent=False,
        )
        return self._manager.s1.internal_state_variables[
            :, self._equivalent_plastic_strain_offset
        ]

    def evaluate(
        self,
        total_strain: ArrayLike,
        *,
        time_increment: float = 1.0,
        consistent_tangent: bool = True,
        commit: bool = False,
    ) -> MFrontIntegrationResult:
        """Integrate a trial total strain from the last committed state."""

        total_kelvin = self._integrate_trial(
            total_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
        )

        conversion_started = time.perf_counter()
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
        self._kelvin_conversion_seconds += time.perf_counter() - conversion_started
        result = MFrontIntegrationResult(
            stress_mpa=stress,
            plastic_strain=plastic_strain,
            equivalent_plastic_strain=equivalent_plastic_strain,
            yield_surface_radius_mpa=yield_surface_radius,
            consistent_tangent_mpa=tangent,
        )
        if commit:
            self.commit()
        return result

    def current_full_tensor_state(self) -> FullTensorState | None:
        """Extract the latest successful native MFront trial without committing it."""

        if not self._has_trial_state:
            raise RuntimeError("no successful MFront trial state is available")
        if self._axial_strain_offset is None:
            return None
        reconstruction_started = time.perf_counter()
        total_kelvin = self._manager.s1.gradients.copy()
        total_kelvin[:, 2] = self._manager.s1.internal_state_variables[:, self._axial_strain_offset]
        elastic_offset = self._elastic_strain_offset
        elastic_kelvin = self._manager.s1.internal_state_variables[
            :, elastic_offset : elastic_offset + 4
        ].copy()
        stress_kelvin = self._manager.s1.thermodynamic_forces.copy()
        state = reconstruct_native_plane_stress_state(
            total_kelvin,
            elastic_kelvin,
            stress_kelvin,
        )
        self._tensor_reconstruction_seconds += (
            time.perf_counter() - reconstruction_started
        )
        self._tensor_reconstruction_calls += 1
        return state

    def commit(self) -> None:
        """Commit the latest successful trial state."""

        if not self._has_trial_state:
            raise RuntimeError("no successful MFront trial state to commit")
        self._mgis.update(self._manager)
        if self._committed_nonlocal_values is not None:
            assert self._trial_nonlocal_values is not None
            self._committed_nonlocal_values[:] = self._trial_nonlocal_values
            self._apply_trial_nonlocal_values()
        self._has_trial_state = False

    def revert(self) -> None:
        """Discard the latest trial and restore the committed state."""

        self._mgis.revert(self._manager)
        if self._committed_nonlocal_values is not None:
            assert self._trial_nonlocal_values is not None
            self._trial_nonlocal_values[:] = self._committed_nonlocal_values
            self._apply_trial_nonlocal_values()
        self._has_trial_state = False


class MFrontNativePlaneStressBatch:
    """Common-contract adapter for MFront's native ``PlaneStress`` behaviour."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._bridge = MFrontMaterialPointBatch(*args, **kwargs)
        if not self._bridge.has_native_plane_stress_state:
            raise MFrontUnavailableError(
                "PixelLudwikJ2Plasticity must expose AxialStrain; analytical completion "
                "is not a generic MFront fallback"
            )
        self._maximum_residual = 0.0

    @property
    def point_count(self) -> int:
        return self._bridge.point_count

    @property
    def backend_name(self) -> str:
        if self._bridge.supports_nonlocal_equivalent_plastic_strain:
            return "mfront-native-plane-stress-micromorphic"
        return "mfront-native-plane-stress"

    @property
    def completion_strategy(self) -> str:
        return "mfront_native_plane_stress"

    @property
    def linear_system_matrix_type(self) -> LinearSystemMatrixType:
        """Use symmetry only for behaviours explicitly verified by this project."""

        return self._bridge.linear_system_matrix_type

    @property
    def statistics(self) -> PlaneStressBatchStatistics:
        return PlaneStressBatchStatistics(
            maximum_gauss_point_plane_stress_residual_mpa=self._maximum_residual
        )

    @property
    def timing_statistics(self) -> MFrontTimingStatistics:
        return self._bridge.timing_statistics

    def evaluate_equivalent_plastic_strain(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
    ) -> NDArray:
        return self._bridge.evaluate_equivalent_plastic_strain(
            in_plane_strain,
            time_increment=time_increment,
        )

    def evaluate_in_plane(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> InPlaneConstitutiveTrial:
        result = self._bridge.evaluate(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
        )
        return InPlaneConstitutiveTrial(
            stress_in_plane_mpa=result.stress_mpa,
            tangent_in_plane_mpa=result.consistent_tangent_mpa,
            observables={
                "plastic_strain_2d": result.plastic_strain,
                "equivalent_plastic_strain": result.equivalent_plastic_strain,
                "yield_surface_radius_mpa": result.yield_surface_radius_mpa,
            },
        )

    def complete_trial(self, trial: InPlaneConstitutiveTrial) -> ConstitutiveTrial:
        full = self._bridge.current_full_tensor_state()
        if full is None:
            raise MFrontUnavailableError("native MFront plane-stress state is unavailable")
        self._maximum_residual = max(
            self._maximum_residual,
            float(np.max(np.abs(full.plane_stress_residual_vector_mpa))),
        )
        return ConstitutiveTrial(
            stress_in_plane_mpa=trial.stress_in_plane_mpa,
            tangent_in_plane_mpa=trial.tangent_in_plane_mpa,
            full_stress_tensor_mpa=full.stress_tensor_mpa,
            full_strain_tensor=full.total_strain_tensor,
            elastic_strain_tensor=full.elastic_strain_tensor,
            plastic_strain_tensor=full.plastic_strain_tensor,
            plane_stress_residual_mpa=full.plane_stress_residual_vector_mpa,
            observables=trial.observables,
            local_plane_stress_iterations=trial.local_plane_stress_iterations,
            cbb_condition_number=trial.cbb_condition_number,
        )

    def evaluate(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> ConstitutiveTrial:
        trial = self.evaluate_in_plane(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
        )
        return self.complete_trial(trial)

    def commit(self) -> None:
        self._bridge.commit()

    def revert(self) -> None:
        self._bridge.revert()

    def set_nonlocal_equivalent_plastic_strain(self, values: ArrayLike) -> None:
        self._bridge.set_nonlocal_equivalent_plastic_strain(values)

    @property
    def committed_nonlocal_equivalent_plastic_strain(self) -> NDArray:
        return self._bridge.committed_nonlocal_equivalent_plastic_strain


@dataclass(frozen=True, slots=True)
class _MFront3DTrial:
    total_strain_kelvin: NDArray
    stress_kelvin_mpa: NDArray
    elastic_strain_kelvin: NDArray
    equivalent_plastic_strain: NDArray
    yield_surface_radius_mpa: NDArray
    consistent_tangent_kelvin_mpa: NDArray


class _MFront3DMaterialPointBatch:
    """Raw transaction-safe MGIS bridge for the tridimensional J2 behaviour."""

    def __init__(
        self,
        library_path: str | Path,
        initial_yield_stress_mpa: ArrayLike,
        hardening_coefficient_mpa: ArrayLike,
        hardening_exponent: ArrayLike,
        *,
        temperature_k: float = 293.15,
        thread_count: int = 1,
        behaviour_name: str = "PixelLudwikJ2Plasticity3D",
        micromorphic_coupling_modulus_mpa: ArrayLike | None = None,
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
        hypothesis = mgis.Hypothesis.Tridimensional
        behaviour = mgis.load(
            str(library.resolve()),
            behaviour_name,
            hypothesis,
        )
        _variable_offset(mgis, behaviour.gradients, "Strain", hypothesis, expected_size=6)
        _variable_offset(
            mgis,
            behaviour.thermodynamic_forces,
            "Stress",
            hypothesis,
            expected_size=6,
        )
        elastic_offset = _variable_offset(
            mgis,
            behaviour.isvs,
            "ElasticStrain",
            hypothesis,
            expected_size=6,
        )
        peeq_offset = _variable_offset(
            mgis,
            behaviour.isvs,
            "EquivalentPlasticStrain",
            hypothesis,
            expected_size=1,
        )
        radius_offset = _variable_offset(
            mgis,
            behaviour.isvs,
            "YieldSurfaceRadius",
            hypothesis,
            expected_size=1,
        )
        assert elastic_offset is not None
        assert peeq_offset is not None
        assert radius_offset is not None
        manager = mgis.MaterialDataManager(behaviour, yield_stress.size)
        material_values = {
            "InitialYieldStress": yield_stress,
            "HardeningCoefficient": coefficient,
            "HardeningExponent": exponent,
        }
        nonlocal_values_s0: NDArray | None = None
        nonlocal_values_s1: NDArray | None = None
        committed_nonlocal_values: NDArray | None = None
        trial_nonlocal_values: NDArray | None = None
        if micromorphic_coupling_modulus_mpa is not None:
            coupling = _broadcast_point_property(
                micromorphic_coupling_modulus_mpa,
                yield_stress.size,
                name="micromorphic_coupling_modulus_mpa",
                nonnegative=True,
            )
            _variable_offset(
                mgis,
                behaviour.mps,
                "MicromorphicCouplingModulus",
                hypothesis,
                expected_size=1,
            )
            _variable_offset(
                mgis,
                behaviour.esvs,
                "NonlocalEquivalentPlasticStrain",
                hypothesis,
                expected_size=1,
            )
            material_values["MicromorphicCouplingModulus"] = coupling
            nonlocal_values_s0 = np.zeros(yield_stress.size)
            nonlocal_values_s1 = np.zeros(yield_stress.size)
            committed_nonlocal_values = np.zeros(yield_stress.size)
            trial_nonlocal_values = np.zeros(yield_stress.size)
        temperature_values = np.full(yield_stress.size, temperature_k)
        storage_mode = mgis.MaterialStateManagerStorageMode.ExternalStorage
        for state in (manager.s0, manager.s1):
            for name, values in material_values.items():
                mgis.setMaterialProperty(state, name, values, storage_mode)
            mgis.setExternalStateVariable(
                state,
                "Temperature",
                temperature_values,
                storage_mode,
            )
        if nonlocal_values_s0 is not None and nonlocal_values_s1 is not None:
            mgis.setExternalStateVariable(
                manager.s0,
                "NonlocalEquivalentPlasticStrain",
                nonlocal_values_s0,
                storage_mode,
            )
            mgis.setExternalStateVariable(
                manager.s1,
                "NonlocalEquivalentPlasticStrain",
                nonlocal_values_s1,
                storage_mode,
            )
        self._mgis = mgis
        self._behaviour = behaviour
        self._manager = manager
        self._point_count = yield_stress.size
        self._material_values = material_values
        self._temperature_values = temperature_values
        self._behaviour_name = behaviour_name
        self._nonlocal_values_s0 = nonlocal_values_s0
        self._nonlocal_values_s1 = nonlocal_values_s1
        self._committed_nonlocal_values = committed_nonlocal_values
        self._trial_nonlocal_values = trial_nonlocal_values
        self._elastic_offset = elastic_offset
        self._peeq_offset = peeq_offset
        self._radius_offset = radius_offset
        self._thread_pool = _load_mgis_root().ThreadPool(thread_count) if thread_count > 1 else None
        self._has_trial_state = False

    @property
    def point_count(self) -> int:
        return self._point_count

    @property
    def behaviour_name(self) -> str:
        """Return the exact MFront behaviour selected by this bridge."""

        return self._behaviour_name

    @property
    def linear_system_matrix_type(self) -> LinearSystemMatrixType:
        """Return the verified matrix capability of the selected behaviour."""

        if self._behaviour_name in _SYMMETRIC_POSITIVE_DEFINITE_J2_BEHAVIOURS:
            return "symmetric_positive_definite"
        return "nonsymmetric"

    @property
    def committed_transverse_strain_kelvin(self) -> NDArray:
        return self._manager.s0.gradients[:, _TRANSVERSE_COMPONENTS_3D].copy()

    @property
    def supports_nonlocal_equivalent_plastic_strain(self) -> bool:
        return self._trial_nonlocal_values is not None

    @property
    def committed_nonlocal_equivalent_plastic_strain(self) -> NDArray:
        if self._committed_nonlocal_values is None:
            raise MFrontUnavailableError(
                f"{self._behaviour_name} does not expose NonlocalEquivalentPlasticStrain"
            )
        return self._committed_nonlocal_values.copy()

    def _apply_trial_nonlocal_values(self) -> None:
        if self._trial_nonlocal_values is None:
            return
        assert self._nonlocal_values_s0 is not None
        assert self._nonlocal_values_s1 is not None
        self._nonlocal_values_s0[:] = self._trial_nonlocal_values
        self._nonlocal_values_s1[:] = self._trial_nonlocal_values

    def set_nonlocal_equivalent_plastic_strain(self, values: ArrayLike) -> None:
        if self._trial_nonlocal_values is None:
            raise MFrontUnavailableError(
                f"{self._behaviour_name} does not expose NonlocalEquivalentPlasticStrain"
            )
        trial = _broadcast_point_property(
            values,
            self._point_count,
            name="nonlocal_equivalent_plastic_strain",
            nonnegative=True,
        )
        if self._has_trial_state:
            self._mgis.revert(self._manager)
            self._has_trial_state = False
        self._trial_nonlocal_values[:] = trial
        self._apply_trial_nonlocal_values()

    def evaluate(self, total_strain_kelvin: ArrayLike, *, time_increment: float) -> _MFront3DTrial:
        strain = np.asarray(total_strain_kelvin, dtype=float)
        if strain.shape != (self._point_count, 6):
            raise ValueError(f"total_strain_kelvin must have shape {(self._point_count, 6)}")
        if not np.isfinite(strain).all():
            raise ValueError("total_strain_kelvin must be finite")
        if not np.isfinite(time_increment) or time_increment <= 0:
            raise ValueError("time_increment must be finite and positive")
        if self._has_trial_state:
            self._mgis.revert(self._manager)
            self._has_trial_state = False
        self._apply_trial_nonlocal_values()
        self._manager.s1.gradients[:, :] = strain
        integration_type = self._mgis.IntegrationType.IntegrationWithConsistentTangentOperator
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
            self.revert()
            raise MFrontIntegrationError(f"3D MFront integration failed with status {status}")
        self._has_trial_state = True
        elastic = self._manager.s1.internal_state_variables[
            :, self._elastic_offset : self._elastic_offset + 6
        ].copy()
        return _MFront3DTrial(
            total_strain_kelvin=strain.copy(),
            stress_kelvin_mpa=self._manager.s1.thermodynamic_forces.copy(),
            elastic_strain_kelvin=elastic,
            equivalent_plastic_strain=self._manager.s1.internal_state_variables[
                :, self._peeq_offset
            ].copy(),
            yield_surface_radius_mpa=self._manager.s1.internal_state_variables[
                :, self._radius_offset
            ].copy(),
            consistent_tangent_kelvin_mpa=self._manager.K.copy(),
        )

    def commit(self) -> None:
        if not self._has_trial_state:
            raise RuntimeError("no successful 3D MFront trial state to commit")
        self._mgis.update(self._manager)
        if self._committed_nonlocal_values is not None:
            assert self._trial_nonlocal_values is not None
            self._committed_nonlocal_values[:] = self._trial_nonlocal_values
            self._apply_trial_nonlocal_values()
        self._has_trial_state = False

    def revert(self) -> None:
        self._mgis.revert(self._manager)
        if self._committed_nonlocal_values is not None:
            assert self._trial_nonlocal_values is not None
            self._trial_nonlocal_values[:] = self._committed_nonlocal_values
            self._apply_trial_nonlocal_values()
        self._has_trial_state = False


def condense_kelvin_tangent_to_engineering(tangent: ArrayLike) -> tuple[NDArray, NDArray]:
    """Return the plane-stress Schur complement and ``Cbb`` condition numbers."""

    values = np.asarray(tangent, dtype=float)
    if values.ndim < 2 or values.shape[-2:] != (6, 6):
        raise ValueError("3D Kelvin tangent must have trailing dimensions (6, 6)")
    if not np.isfinite(values).all():
        raise ValueError("3D Kelvin tangent must be finite")
    caa = np.take(
        np.take(values, _PLANE_STRESS_COMPONENTS, axis=-2), _PLANE_STRESS_COMPONENTS, axis=-1
    )
    cab = np.take(
        np.take(values, _PLANE_STRESS_COMPONENTS, axis=-2), _TRANSVERSE_COMPONENTS_3D, axis=-1
    )
    cba = np.take(
        np.take(values, _TRANSVERSE_COMPONENTS_3D, axis=-2), _PLANE_STRESS_COMPONENTS, axis=-1
    )
    cbb = np.take(
        np.take(values, _TRANSVERSE_COMPONENTS_3D, axis=-2), _TRANSVERSE_COMPONENTS_3D, axis=-1
    )
    condition = np.linalg.cond(cbb)
    condensed_kelvin = caa - cab @ np.linalg.solve(cbb, cba)
    condensed_engineering = (
        condensed_kelvin
        * _KELVIN_TO_ENGINEERING_STRESS_SCALE[:, None]
        * _ENGINEERING_TO_KELVIN_STRAIN_SCALE[None, :]
    )
    return condensed_engineering, condition


class MFront3DCondensedPlaneStressBatch:
    """Impose three plane-stress constraints on a 3D MFront behaviour."""

    def __init__(
        self,
        *args: Any,
        local_tolerance_mpa: float = 1e-8,
        local_relative_tolerance: float = 1e-10,
        maximum_local_iterations: int = 15,
        maximum_cbb_condition_number: float = 1e12,
        **kwargs: Any,
    ) -> None:
        if not np.isfinite(local_tolerance_mpa) or local_tolerance_mpa <= 0:
            raise ValueError("local_tolerance_mpa must be finite and positive")
        if not np.isfinite(local_relative_tolerance) or local_relative_tolerance <= 0:
            raise ValueError("local_relative_tolerance must be finite and positive")
        if maximum_local_iterations < 1:
            raise ValueError("maximum_local_iterations must be positive")
        if not np.isfinite(maximum_cbb_condition_number) or maximum_cbb_condition_number <= 1:
            raise ValueError("maximum_cbb_condition_number must be finite and greater than one")
        self._bridge = _MFront3DMaterialPointBatch(*args, **kwargs)
        self._absolute_tolerance = float(local_tolerance_mpa)
        self._relative_tolerance = float(local_relative_tolerance)
        self._maximum_iterations = maximum_local_iterations
        self._maximum_condition = float(maximum_cbb_condition_number)
        self._maximum_residual = 0.0
        self._maximum_iterations_observed = 0
        self._iteration_sum = 0
        self._iteration_count = 0
        self._failures = 0
        self._maximum_condition_observed = 0.0

    @property
    def point_count(self) -> int:
        return self._bridge.point_count

    @property
    def backend_name(self) -> str:
        if self._bridge.supports_nonlocal_equivalent_plastic_strain:
            return "mfront-3d-condensed-plane-stress-micromorphic"
        return "mfront-3d-condensed-plane-stress"

    @property
    def completion_strategy(self) -> str:
        return "mfront_3d_local_condensation"

    @property
    def linear_system_matrix_type(self) -> LinearSystemMatrixType:
        """Use symmetry only for behaviours explicitly verified by this project."""

        return self._bridge.linear_system_matrix_type

    @property
    def statistics(self) -> PlaneStressBatchStatistics:
        mean = self._iteration_sum / self._iteration_count if self._iteration_count else 0.0
        return PlaneStressBatchStatistics(
            maximum_gauss_point_plane_stress_residual_mpa=self._maximum_residual,
            maximum_local_plane_stress_iterations=self._maximum_iterations_observed,
            mean_local_plane_stress_iterations=mean,
            local_plane_stress_failures=self._failures,
            maximum_cbb_condition_number=self._maximum_condition_observed,
        )

    def _fail(self, message: str) -> None:
        self._failures += 1
        self._bridge.revert()
        raise LocalPlaneStressConvergenceError(message)

    def evaluate(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> ConstitutiveTrial:
        in_plane = np.asarray(in_plane_strain, dtype=float)
        if in_plane.shape != (self.point_count, 3):
            raise ValueError(f"in_plane_strain must have shape {(self.point_count, 3)}")
        if not np.isfinite(in_plane).all():
            raise ValueError("in_plane_strain must be finite")
        total_kelvin = np.zeros((self.point_count, 6), dtype=float)
        total_kelvin[:, _PLANE_STRESS_COMPONENTS] = in_plane * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
        total_kelvin[:, _TRANSVERSE_COMPONENTS_3D] = self._bridge.committed_transverse_strain_kelvin
        first_converged = np.zeros(self.point_count, dtype=int)
        final: _MFront3DTrial | None = None
        final_condition: NDArray | None = None
        for iteration in range(1, self._maximum_iterations + 1):
            final = self._bridge.evaluate(total_kelvin, time_increment=time_increment)
            stress_b = final.stress_kelvin_mpa[:, _TRANSVERSE_COMPONENTS_3D]
            if not np.isfinite(stress_b).all():
                self._fail("local plane-stress residual is non-finite")
            tangent = final.consistent_tangent_kelvin_mpa
            cbb = np.take(
                np.take(tangent, _TRANSVERSE_COMPONENTS_3D, axis=-2),
                _TRANSVERSE_COMPONENTS_3D,
                axis=-1,
            )
            condition = np.linalg.cond(cbb)
            if not np.isfinite(condition).all():
                self._fail("Cbb condition number is non-finite")
            maximum_condition = float(np.max(condition))
            self._maximum_condition_observed = max(
                self._maximum_condition_observed,
                maximum_condition,
            )
            if maximum_condition > self._maximum_condition:
                self._fail(
                    f"Cbb condition number {maximum_condition:.3e} exceeds "
                    f"{self._maximum_condition:.3e}"
                )
            stress_scale = np.maximum(
                1.0,
                np.max(np.abs(final.stress_kelvin_mpa), axis=1),
            )
            residual_norm = np.max(np.abs(stress_b), axis=1)
            converged = residual_norm <= (
                self._absolute_tolerance + self._relative_tolerance * stress_scale
            )
            first_converged[(first_converged == 0) & converged] = iteration
            if np.all(converged):
                final_condition = condition
                break
            try:
                correction = np.linalg.solve(cbb, -stress_b[..., None])[..., 0]
            except np.linalg.LinAlgError as error:
                self._fail(f"failed to solve local Cbb system: {error}")
            correction[converged] = 0.0
            if not np.isfinite(correction).all():
                self._fail("local plane-stress correction is non-finite")
            total_kelvin[:, _TRANSVERSE_COMPONENTS_3D] += correction
        else:
            maximum_residual = float(np.max(np.abs(stress_b)))
            self._fail(
                f"local plane-stress Newton did not converge in "
                f"{self._maximum_iterations} iterations; residual={maximum_residual:.3e} MPa"
            )
        assert final is not None
        assert final_condition is not None
        full_stress = kelvin_3d_to_tensor(final.stress_kelvin_mpa, quantity="stress")
        full_strain = kelvin_3d_to_tensor(final.total_strain_kelvin, quantity="strain")
        elastic_strain = kelvin_3d_to_tensor(final.elastic_strain_kelvin, quantity="strain")
        plastic_strain = full_strain - elastic_strain
        residual = np.stack(
            (
                full_stress[:, 2, 2],
                full_stress[:, 0, 2],
                full_stress[:, 1, 2],
            ),
            axis=-1,
        )
        self._maximum_residual = max(
            self._maximum_residual,
            float(np.max(np.abs(residual))),
        )
        self._maximum_iterations_observed = max(
            self._maximum_iterations_observed,
            int(np.max(first_converged)),
        )
        self._iteration_sum += int(np.sum(first_converged))
        self._iteration_count += self.point_count
        tangent_engineering, condition = condense_kelvin_tangent_to_engineering(
            final.consistent_tangent_kelvin_mpa
        )
        return ConstitutiveTrial(
            stress_in_plane_mpa=tensor_to_engineering_stress_2d(full_stress),
            tangent_in_plane_mpa=tangent_engineering if consistent_tangent else None,
            full_stress_tensor_mpa=full_stress,
            full_strain_tensor=full_strain,
            elastic_strain_tensor=elastic_strain,
            plastic_strain_tensor=plastic_strain,
            plane_stress_residual_mpa=residual,
            observables={
                "plastic_strain_2d": tensor_to_engineering_strain_2d(plastic_strain),
                "equivalent_plastic_strain": final.equivalent_plastic_strain,
                "yield_surface_radius_mpa": final.yield_surface_radius_mpa,
            },
            local_plane_stress_iterations=first_converged,
            cbb_condition_number=condition,
        )

    def evaluate_in_plane(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> InPlaneConstitutiveTrial:
        return self.evaluate(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
        )

    def evaluate_equivalent_plastic_strain(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
    ) -> NDArray:
        trial = self.evaluate_in_plane(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=False,
        )
        return trial.observables["equivalent_plastic_strain"]

    def complete_trial(self, trial: InPlaneConstitutiveTrial) -> ConstitutiveTrial:
        if not isinstance(trial, ConstitutiveTrial):
            raise TypeError("3D condensed trial is missing its reconstructed state")
        return trial

    def commit(self) -> None:
        self._bridge.commit()

    def revert(self) -> None:
        self._bridge.revert()

    def set_nonlocal_equivalent_plastic_strain(self, values: ArrayLike) -> None:
        self._bridge.set_nonlocal_equivalent_plastic_strain(values)

    @property
    def committed_nonlocal_equivalent_plastic_strain(self) -> NDArray:
        return self._bridge.committed_nonlocal_equivalent_plastic_strain
