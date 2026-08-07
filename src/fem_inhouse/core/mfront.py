"""Optional MFront/MGIS constitutive backend for plane-stress material points."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.crystal_orientation import mgis_rotation_argument, validate_rotations
from fem_inhouse.core.linear_solver import LinearSystemMatrixType
from fem_inhouse.core.mfront_behaviours import MFrontBehaviourSpec
from fem_inhouse.core.plane_stress_material import (
    ConstitutiveIntegrationError,
    ConstitutiveTrial,
    InPlaneConstitutiveTrial,
    LocalPlaneStressConvergenceError,
    PlaneStressBatchStatistics,
    ResponseLevel,
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

LocalConditionCheckMode = Literal["always", "on_failure", "diagnostic_sample"]


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
class MFrontMaterialStateSnapshot:
    """Committed MGIS state captured for an isolated adaptive branch."""

    gradients_s0: NDArray
    internal_state_variables_s0: NDArray
    thermodynamic_forces_s0: NDArray
    committed_global_strain: NDArray
    committed_nonlocal_values: NDArray | None


@dataclass(frozen=True, slots=True)
class MFrontCondensedStateSnapshot:
    """Committed state of the 3-D bridge and plane-stress predictor."""

    bridge: MFrontMaterialStateSnapshot
    accepted_transverse: NDArray
    latest_transverse: NDArray | None
    has_accepted_global_trial: bool
    last_in_plane: NDArray | None
    last_time_increment: float | None
    accepted_in_plane: NDArray | None = None
    accepted_cbb: NDArray | None = None
    accepted_cba: NDArray | None = None


@dataclass(frozen=True, slots=True)
class MFrontCondensedBlocksStateSnapshot:
    """Committed state of a block-partitioned condensed material."""

    blocks: tuple[MFrontCondensedStateSnapshot, ...]


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
    rotation_to_material_seconds: float = 0.0
    integration_seconds: float = 0.0
    rotation_to_global_seconds: float = 0.0
    condensation_seconds: float = 0.0
    condition_check_seconds: float = 0.0
    local_solve_seconds: float = 0.0
    reconstruction_seconds: float = 0.0
    observable_seconds: float = 0.0
    evaluate_calls: int = 0
    condition_checks: int = 0
    local_condensation_evaluations: int = 0
    full_batch_integration_calls: int = 0
    equivalent_active_point_integrations: int = 0
    material_point_integrations: int = 0
    material_point_integrations_with_tangent: int = 0
    material_point_integrations_without_tangent: int = 0
    material_block_integration_calls: int = 0
    material_block_count: int = 1
    native_batch_calls: int = 0
    native_material_points: int = 0
    native_internal_integrations: int = 0
    native_total_local_iterations: int = 0
    native_thread_count: int = 1
    local_iteration_histogram: tuple[int, ...] = ()


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
        # The experimental generalized-plane-stress binding is intentionally
        # shipped as a standalone extension while it is being qualified.  It
        # exposes the same behaviour API and is accepted only as an explicit
        # opt-in backend below.
        try:
            return import_module("behaviour")
        except (ImportError, OSError):
            raise MFrontUnavailableError(
                "MGIS Python bindings are unavailable; source the TFEL environment "
                "before starting Python"
            ) from error


def _load_mgis_root() -> Any:
    try:
        return import_module("mgis")
    except (ImportError, OSError) as error:
        try:
            return import_module("behaviour")
        except (ImportError, OSError):
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


def _declared_internal_slices(
    mgis: Any,
    behaviour: Any,
    hypothesis: Any,
    specification: MFrontBehaviourSpec | None,
) -> dict[str, slice]:
    """Locate each declared internal-state family in the flat MGIS array.

    The offset is resolved from the first member of the family and the extent
    from the declared `component_count`, checked against the sizes MGIS
    reports. Names such as `PlasticSlip[7]` are never parsed: MFront indexes
    array variables with a suffix, but relying on that spelling would break on
    any behaviour that names its members differently.
    """

    if specification is None:
        return {}
    sizes = {
        variable.name: int(mgis.getVariableSize(variable, hypothesis))
        for variable in behaviour.isvs
    }
    offsets: dict[str, int] = {}
    running = 0
    for variable in behaviour.isvs:
        offsets[variable.name] = running
        running += sizes[variable.name]

    slices: dict[str, slice] = {}
    for declared in specification.internal_state_variables:
        head = declared.entry_name
        if head not in offsets:
            # MFront suffixes the members of an array variable.
            head = f"{declared.entry_name}[0]"
        if head not in offsets:
            if declared.required:
                raise MFrontUnavailableError(
                    f"MFront behaviour does not expose internal state "
                    f"{declared.entry_name!r} declared by the catalogue"
                )
            continue
        start = offsets[head]
        stop = start + declared.component_count
        if stop > running:
            raise MFrontUnavailableError(
                f"internal state {declared.entry_name!r} declares "
                f"{declared.component_count} components but only {running - start} "
                "remain in the behaviour"
            )
        slices[declared.canonical_name] = slice(start, stop)
    return slices


def _apply_behaviour_parameters(
    mgis: Any,
    behaviour: Any,
    values: Mapping[str, float] | None,
    behaviour_name: str,
) -> None:
    """Override MFront `@Parameter` values on a loaded behaviour.

    This is how a registered parameter set reaches the law without editing and
    recompiling the `.mfront` source.

    **`mgis.load` does not give you a private behaviour.** Two `load` calls for
    the same library, name and hypothesis return handles onto the *same*
    underlying object, so a `setParameter` through one is visible through the
    other, process-wide. Measured, not assumed: setting `tau0 = 999` on one
    handle moved the plateau of a run driven through a second, untouched handle.

    The consequence is that applying a set once at construction is not enough.
    Two batches with different sets alive in the same process would silently
    share whichever set was applied last -- and the wrong one would be the
    plausible-looking one. `_reassert_behaviour_parameters` therefore re-applies
    the batch's own values immediately before every integration, which is exact
    under any interleaving and costs a handful of `setParameter` calls against
    the integration of every point in the batch.

    An unknown name is refused rather than ignored. MGIS `setParameter` on a
    name the behaviour does not export raises, but only when it is called, and
    a silently mistyped parameter would otherwise look like a run that simply
    used the defaults -- which is exactly the failure that must not be possible
    to publish. The check happens here, before the first integration.
    """

    if not values:
        return
    declared = set(behaviour.parameters)
    unknown = sorted(name for name in values if name not in declared)
    if unknown:
        raise ValueError(
            f"MFront behaviour {behaviour_name!r} does not expose the parameter(s) "
            f"{', '.join(unknown)}; it exposes: {', '.join(sorted(declared))}"
        )
    for name, value in values.items():
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"parameter {name!r} must be finite, got {value!r}")
        mgis.setParameter(behaviour, name, numeric)


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
        #: The native plane-stress J2 law exposes no selectable parameter set,
        #: so there is nothing to re-assert; the attribute exists so the shared
        #: guard against MGIS's shared behaviour handles applies uniformly.
        self._behaviour_parameters: dict[str, float] = {}
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
            material_point_integrations=(
                self._point_count
                * (self._integration_with_tangent_calls + self._integration_without_tangent_calls)
            ),
            material_point_integrations_with_tangent=(
                self._point_count * self._integration_with_tangent_calls
            ),
            material_point_integrations_without_tangent=(
                self._point_count * self._integration_without_tangent_calls
            ),
            material_block_integration_calls=(
                self._integration_with_tangent_calls + self._integration_without_tangent_calls
            ),
            material_block_count=1,
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

    def _reassert_behaviour_parameters(self) -> None:
        """No-op for this law, kept so the integration path is uniform."""

        for name, value in self._behaviour_parameters.items():
            self._mgis.setParameter(self._behaviour, name, value)

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
        self._reassert_behaviour_parameters()
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

    def evaluate_nonlocal_state(
        self,
        total_strain: ArrayLike,
        *,
        time_increment: float = 1.0,
    ) -> tuple[NDArray, NDArray]:
        """Integrate without tangent and expose PEEQ and the yield radius.

        Both arrays are ephemeral MGIS views valid until the next trial
        integration.  This keeps positivity diagnostics inside the lightweight
        micromorphic fixed-point path.
        """

        self._integrate_trial(
            total_strain,
            time_increment=time_increment,
            consistent_tangent=False,
        )
        state = self._manager.s1.internal_state_variables
        return (
            state[:, self._equivalent_plastic_strain_offset],
            state[:, self._yield_surface_radius_offset],
        )

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

    def evaluate_nonlocal_state(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
    ) -> tuple[NDArray, NDArray]:
        return self._bridge.evaluate_nonlocal_state(
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

    def reference_in_plane_tangent_mpa(self) -> NDArray:
        """Elastic plane-stress tangent, measured the same way as the 3D bridge.

        See MFront3DCondensedPlaneStressBatch.reference_in_plane_tangent_mpa;
        the reasoning is identical and the probe is reverted here too.
        """

        probe = self.evaluate(
            np.zeros((self.point_count, 3)), time_increment=1.0, consistent_tangent=True
        )
        self.revert()
        tangent = probe.tangent_in_plane_mpa
        if tangent is None:  # pragma: no cover - requested explicitly above
            raise MFrontUnavailableError("behaviour returned no consistent tangent")
        return np.asarray(tangent[0], dtype=float)

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
    #: Everything the behaviour spec declares, in the global frame for tensors
    #: and in the material frame for per-slip-system quantities.
    observables: dict[str, NDArray] = field(default_factory=dict)


#: The profile whose J2 conventions the bridge is allowed to assume.
#:
#: Every reference to InitialYieldStress, EquivalentPlasticStrain or
#: YieldSurfaceRadius lives behind a check against this constant. A crystal has
#: twelve slips and twelve critical resolved shear stresses, not a scalar pair,
#: so those names must never be required of a behaviour that does not declare
#: them.
_LUDWIK_J2_PROFILE = "ludwik_j2_v1"


class MFront3DMaterialPointBatch:
    """Transaction-safe MGIS bridge for any tridimensional MFront behaviour.

    The bridge is driven by an `MFrontBehaviourSpec`: material properties come
    from a generic mapping, internal state variables are read at the offsets the
    catalogue declares, and nothing here knows what a yield surface is.

    Orientations are optional. When supplied they are one
    `Q_global_to_material` per material point, so an EBSD map is a different
    provider and not a different bridge. Strains are rotated into the crystal
    frame before integration, and stresses and the consistent tangent are
    rotated back, so everything crossing this boundary is in the global frame.
    """

    def __init__(
        self,
        library_path: str | Path,
        initial_yield_stress_mpa: ArrayLike | None = None,
        hardening_coefficient_mpa: ArrayLike | None = None,
        hardening_exponent: ArrayLike | None = None,
        *,
        behaviour_spec: MFrontBehaviourSpec | None = None,
        point_count: int | None = None,
        material_property_values: Mapping[str, float | ArrayLike] | None = None,
        rotation_global_to_material: ArrayLike | None = None,
        temperature_k: float = 293.15,
        thread_count: int = 1,
        behaviour_name: str = "PixelLudwikJ2Plasticity3D",
        micromorphic_coupling_modulus_mpa: ArrayLike | None = None,
        behaviour_parameters: Mapping[str, float] | None = None,
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
        profile = behaviour_spec.bridge_profile if behaviour_spec is not None else (
            _LUDWIK_J2_PROFILE
        )
        is_j2 = profile == _LUDWIK_J2_PROFILE
        if is_j2:
            missing = [
                name
                for name, value in (
                    ("initial_yield_stress_mpa", initial_yield_stress_mpa),
                    ("hardening_coefficient_mpa", hardening_coefficient_mpa),
                    ("hardening_exponent", hardening_exponent),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    f"behaviour profile {_LUDWIK_J2_PROFILE!r} requires "
                    f"{', '.join(missing)}"
                )
            assert initial_yield_stress_mpa is not None
            assert hardening_coefficient_mpa is not None
            assert hardening_exponent is not None
            yield_stress, coefficient, exponent = _broadcast_material_properties(
                initial_yield_stress_mpa,
                hardening_coefficient_mpa,
                hardening_exponent,
            )
            resolved_point_count = int(yield_stress.size)
        else:
            if point_count is None:
                raise ValueError(
                    f"behaviour profile {profile!r} carries no J2 material properties, "
                    "so point_count must be given explicitly"
                )
            if isinstance(point_count, bool) or not isinstance(point_count, int):
                raise TypeError("point_count must be an integer")
            if point_count < 1:
                raise ValueError("point_count must be at least 1")
            if any(
                value is not None
                for value in (
                    initial_yield_stress_mpa,
                    hardening_coefficient_mpa,
                    hardening_exponent,
                )
            ):
                raise ValueError(
                    f"behaviour profile {profile!r} does not accept the J2 hardening "
                    "properties; pass material_property_values instead"
                )
            resolved_point_count = point_count
        mgis = _load_mgis()
        hypothesis = mgis.Hypothesis.Tridimensional
        behaviour = mgis.load(
            str(library.resolve()),
            behaviour_name,
            hypothesis,
        )
        _apply_behaviour_parameters(mgis, behaviour, behaviour_parameters, behaviour_name)
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
        assert elastic_offset is not None
        peeq_offset: int | None = None
        radius_offset: int | None = None
        if is_j2:
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
            assert peeq_offset is not None
            assert radius_offset is not None
        observable_slices = _declared_internal_slices(mgis, behaviour, hypothesis, behaviour_spec)
        manager = mgis.MaterialDataManager(behaviour, resolved_point_count)
        material_values: dict[str, NDArray] = {}
        if is_j2:
            material_values.update(
                {
                    "InitialYieldStress": yield_stress,
                    "HardeningCoefficient": coefficient,
                    "HardeningExponent": exponent,
                }
            )
        for name, value in (material_property_values or {}).items():
            material_values[name] = _broadcast_point_property(
                value, resolved_point_count, name=name
            )
        nonlocal_values_s0: NDArray | None = None
        nonlocal_values_s1: NDArray | None = None
        committed_nonlocal_values: NDArray | None = None
        trial_nonlocal_values: NDArray | None = None
        if micromorphic_coupling_modulus_mpa is not None:
            coupling = _broadcast_point_property(
                micromorphic_coupling_modulus_mpa,
                resolved_point_count,
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
            nonlocal_values_s0 = np.zeros(resolved_point_count)
            nonlocal_values_s1 = np.zeros(resolved_point_count)
            committed_nonlocal_values = np.zeros(resolved_point_count)
            trial_nonlocal_values = np.zeros(resolved_point_count)
        temperature_values = np.full(resolved_point_count, temperature_k)
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
        self._point_count = resolved_point_count
        #: Kept so they can be re-applied before every integration; see
        #: `_apply_behaviour_parameters` for why once is not enough.
        self._behaviour_parameters = dict(behaviour_parameters or {})
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
        self._specification = behaviour_spec
        self._profile = profile
        self._thread_count = thread_count
        self._observable_slices = observable_slices
        if rotation_global_to_material is None:
            self._rotations: NDArray | None = None
            self._mgis_rotations: NDArray | None = None
        else:
            self._rotations = validate_rotations(
                rotation_global_to_material, point_count=resolved_point_count
            )
            self._mgis_rotations = mgis_rotation_argument(self._rotations)
        self._thread_pool = _load_mgis_root().ThreadPool(thread_count) if thread_count > 1 else None
        self._has_trial_state = False
        #: Committed total strain in the global frame; see
        #: committed_transverse_strain_kelvin for why it is not read from s0.
        self._committed_global_strain = np.zeros((resolved_point_count, 6))
        self._trial_global_strain = np.zeros((resolved_point_count, 6))
        self._rotation_to_material_seconds = 0.0
        self._integration_seconds = 0.0
        self._rotation_to_global_seconds = 0.0
        self._evaluate_calls = 0

    @property
    def point_count(self) -> int:
        return self._point_count

    @property
    def behaviour_name(self) -> str:
        """Return the exact MFront behaviour selected by this bridge."""

        return self._behaviour_name

    @property
    def thread_count(self) -> int:
        return self._thread_count

    @property
    def is_oriented(self) -> bool:
        """Whether a crystallographic orientation is applied to each point."""

        return self._rotations is not None

    @property
    def rotations_global_to_material(self) -> NDArray | None:
        return None if self._rotations is None else self._rotations.copy()

    @property
    def linear_system_matrix_type(self) -> LinearSystemMatrixType:
        """Return the verified matrix capability of the selected behaviour."""

        if self._specification is not None:
            return self._specification.linear_system_matrix_type
        if self._behaviour_name in _SYMMETRIC_POSITIVE_DEFINITE_J2_BEHAVIOURS:
            return "symmetric_positive_definite"
        return "nonsymmetric"

    @property
    def committed_transverse_strain_kelvin(self) -> NDArray:
        """Out-of-plane components of the committed strain, in the GLOBAL frame.

        The condensation seeds each increment with this, then drives the global
        out-of-plane stresses to zero. Reading it from `s0.gradients` would
        return crystal-frame components, which the condensation would then treat
        as global ones: harmless at the identity orientation, where the two
        frames coincide, and silently wrong at every other. The global strain is
        therefore tracked separately rather than recovered by rotating back.
        """

        return self._committed_global_strain[:, _TRANSVERSE_COMPONENTS_3D].copy()

    @property
    def timing_statistics(self) -> MFrontTimingStatistics:
        return MFrontTimingStatistics(
            rotation_to_material_seconds=self._rotation_to_material_seconds,
            integration_seconds=self._integration_seconds,
            rotation_to_global_seconds=self._rotation_to_global_seconds,
            evaluate_calls=self._evaluate_calls,
            material_point_integrations=self._point_count * self._evaluate_calls,
            material_point_integrations_with_tangent=(
                self._point_count * self._evaluate_calls
            ),
            material_point_integrations_without_tangent=0,
            material_block_integration_calls=self._evaluate_calls,
            material_block_count=1,
        )

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

    def _reassert_behaviour_parameters(self) -> None:
        """Re-apply this batch's parameter values before integrating.

        `mgis.load` hands out shared handles, so another batch in the same
        process can have overwritten them since the last call. Cheap: a dozen
        scalar writes against the integration of every point in the batch.
        """

        for name, value in self._behaviour_parameters.items():
            self._mgis.setParameter(self._behaviour, name, value)

    def evaluate(
        self,
        total_strain_kelvin: ArrayLike,
        *,
        time_increment: float,
        collect_observables: bool = True,
    ) -> _MFront3DTrial:
        self._evaluate_calls += 1
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
        self._reassert_behaviour_parameters()
        self._apply_trial_nonlocal_values()
        # Step 3 of the plane-stress ordering: the strain arrives in the global
        # frame, complete with the current transverse components, and is turned
        # into the crystal frame here. The rotation call writes in place, so it
        # is given a copy and never the caller's array.
        rotation_started = time.perf_counter()
        if self._mgis_rotations is None:
            self._manager.s1.gradients[:, :] = strain
        else:
            crystal_strain = np.ascontiguousarray(strain.reshape(-1).copy())
            self._mgis.rotateGradients(
                crystal_strain, self._behaviour, self._mgis_rotations
            )
            self._manager.s1.gradients[:, :] = crystal_strain.reshape(self._point_count, 6)
        self._rotation_to_material_seconds += time.perf_counter() - rotation_started
        integration_type = self._mgis.IntegrationType.IntegrationWithConsistentTangentOperator
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
        self._integration_seconds += time.perf_counter() - integration_started
        if status != 1:
            self.revert()
            raise MFrontIntegrationError(f"3D MFront integration failed with status {status}")
        self._has_trial_state = True
        self._trial_global_strain[:, :] = strain
        state = self._manager.s1.internal_state_variables
        elastic = state[:, self._elastic_offset : self._elastic_offset + 6].copy()
        stress = self._manager.s1.thermodynamic_forces.copy()
        tangent = self._manager.K.copy()
        # Steps 5 and 6: bring the stress and the consistent tangent back to the
        # global frame. The elastic strain is a crystal-frame quantity but is
        # rotated too, because the solver subtracts it from the global total
        # strain to obtain the plastic part.
        rotation_started = time.perf_counter()
        if self._mgis_rotations is not None:
            for tensor in (stress, elastic):
                flat = np.ascontiguousarray(tensor.reshape(-1))
                self._mgis.rotateThermodynamicForces(
                    flat, self._behaviour, self._mgis_rotations
                )
                tensor[:, :] = flat.reshape(self._point_count, 6)
            flat_tangent = np.ascontiguousarray(tangent.reshape(-1))
            self._mgis.rotateTangentOperatorBlocks(
                flat_tangent, self._behaviour, self._mgis_rotations
            )
            tangent = flat_tangent.reshape(tangent.shape)
        self._rotation_to_global_seconds += time.perf_counter() - rotation_started

        observables = (
            {
                name: state[:, position].copy()
                for name, position in self._observable_slices.items()
            }
            if collect_observables
            else {}
        )
        if collect_observables and "equivalent_plastic_slip" in observables:
            # Not a J2 equivalent plastic strain and deliberately not named like
            # one: the sum of the twelve accumulated slips is a different scalar
            # with a different meaning.
            observables["accumulated_slip"] = observables["equivalent_plastic_slip"].sum(axis=1)

        empty = np.empty(0)
        return _MFront3DTrial(
            total_strain_kelvin=strain.copy(),
            stress_kelvin_mpa=stress,
            elastic_strain_kelvin=elastic,
            equivalent_plastic_strain=(
                state[:, self._peeq_offset].copy() if self._peeq_offset is not None else empty
            ),
            yield_surface_radius_mpa=(
                state[:, self._radius_offset].copy() if self._radius_offset is not None else empty
            ),
            consistent_tangent_kelvin_mpa=tangent,
            observables=observables,
        )

    def commit(self) -> None:
        if not self._has_trial_state:
            raise RuntimeError("no successful 3D MFront trial state to commit")
        self._mgis.update(self._manager)
        self._committed_global_strain[:, :] = self._trial_global_strain
        if self._committed_nonlocal_values is not None:
            assert self._trial_nonlocal_values is not None
            self._committed_nonlocal_values[:] = self._trial_nonlocal_values
            self._apply_trial_nonlocal_values()
        self._has_trial_state = False

    def snapshot_state(self) -> MFrontMaterialStateSnapshot:
        """Capture the committed state without changing MGIS ownership."""

        if self._has_trial_state:
            raise RuntimeError("snapshot_state requires a committed MGIS state")
        return MFrontMaterialStateSnapshot(
            gradients_s0=np.asarray(self._manager.s0.gradients, dtype=float).copy(),
            internal_state_variables_s0=np.asarray(
                self._manager.s0.internal_state_variables, dtype=float
            ).copy(),
            thermodynamic_forces_s0=np.asarray(
                self._manager.s0.thermodynamic_forces, dtype=float
            ).copy(),
            committed_global_strain=self._committed_global_strain.copy(),
            committed_nonlocal_values=(
                None
                if self._committed_nonlocal_values is None
                else self._committed_nonlocal_values.copy()
            ),
        )

    def restore_state(self, snapshot: MFrontMaterialStateSnapshot) -> None:
        """Restore a committed state and clear any active MGIS trial."""

        if self._has_trial_state:
            self._mgis.revert(self._manager)
            self._has_trial_state = False
        if snapshot.gradients_s0.shape != self._manager.s0.gradients.shape:
            raise ValueError("incompatible MGIS snapshot gradient shape")
        if snapshot.internal_state_variables_s0.shape != (
            self._manager.s0.internal_state_variables.shape
        ):
            raise ValueError("incompatible MGIS snapshot state-variable shape")
        for state in (self._manager.s0, self._manager.s1):
            state.gradients[:, :] = snapshot.gradients_s0
            state.internal_state_variables[:, :] = snapshot.internal_state_variables_s0
            state.thermodynamic_forces[:, :] = snapshot.thermodynamic_forces_s0
        self._committed_global_strain[:, :] = snapshot.committed_global_strain
        if self._committed_nonlocal_values is not None:
            if snapshot.committed_nonlocal_values is None:
                raise ValueError("snapshot is missing committed nonlocal state")
            self._committed_nonlocal_values[:] = snapshot.committed_nonlocal_values
            assert self._trial_nonlocal_values is not None
            self._trial_nonlocal_values[:] = snapshot.committed_nonlocal_values
            self._apply_trial_nonlocal_values()
        self._trial_global_strain[:, :] = self._committed_global_strain

    def revert(self) -> None:
        self._mgis.revert(self._manager)
        if self._committed_nonlocal_values is not None:
            assert self._trial_nonlocal_values is not None
            self._trial_nonlocal_values[:] = self._committed_nonlocal_values
            self._apply_trial_nonlocal_values()
        self._has_trial_state = False


def condense_kelvin_tangent_to_engineering(
    tangent: ArrayLike,
    *,
    check_condition: bool = True,
) -> tuple[NDArray, NDArray | None]:
    """Return the plane-stress Schur complement and optional ``Cbb`` checks."""

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
    condition = np.linalg.cond(cbb) if check_condition else None
    condensed_kelvin = caa - cab @ np.linalg.solve(cbb, cba)
    condensed_engineering = (
        condensed_kelvin
        * _KELVIN_TO_ENGINEERING_STRESS_SCALE[:, None]
        * _ENGINEERING_TO_KELVIN_STRAIN_SCALE[None, :]
    )
    return condensed_engineering, condition


#: Retained so that existing imports of the private name keep working.
_MFront3DMaterialPointBatch = MFront3DMaterialPointBatch


class MFront3DCondensedPlaneStressBatch:
    """Impose three plane-stress constraints on a 3D MFront behaviour."""

    def __init__(
        self,
        *args: Any,
        local_tolerance_mpa: float = 1e-8,
        local_relative_tolerance: float = 1e-10,
        maximum_local_iterations: int = 15,
        maximum_cbb_condition_number: float = 1e12,
        local_condition_check_mode: LocalConditionCheckMode = "always",
        local_transverse_predictor: Literal["committed", "tangent"] = "committed",
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
        if local_condition_check_mode not in {"always", "on_failure", "diagnostic_sample"}:
            raise ValueError(
                "local_condition_check_mode must be 'always', 'on_failure', "
                "or 'diagnostic_sample'"
            )
        if local_transverse_predictor not in {"committed", "tangent"}:
            raise ValueError("local_transverse_predictor must be 'committed' or 'tangent'")
        self._bridge = MFront3DMaterialPointBatch(*args, **kwargs)
        self._absolute_tolerance = float(local_tolerance_mpa)
        self._relative_tolerance = float(local_relative_tolerance)
        self._maximum_iterations = maximum_local_iterations
        self._maximum_condition = float(maximum_cbb_condition_number)
        self._condition_check_mode = local_condition_check_mode
        self._local_transverse_predictor = local_transverse_predictor
        self._maximum_residual = 0.0
        self._maximum_iterations_observed = 0
        self._iteration_sum = 0
        self._iteration_count = 0
        self._failures = 0
        self._maximum_condition_observed = 0.0
        self._condensation_seconds = 0.0
        self._condition_check_seconds = 0.0
        self._local_solve_seconds = 0.0
        self._reconstruction_seconds = 0.0
        self._observable_seconds = 0.0
        self._condition_checks = 0
        self._local_condensation_evaluations = 0
        self._full_batch_integration_calls = 0
        self._equivalent_active_point_integrations = 0
        self._local_iteration_histogram = np.zeros(maximum_local_iterations + 1, dtype=np.int64)
        self._accepted_transverse = self._bridge.committed_transverse_strain_kelvin.copy()
        self._latest_transverse: NDArray | None = None
        self._has_accepted_global_trial = False
        self._warm_start_uses = 0
        self._warm_start_resets = 0
        self._last_in_plane: NDArray | None = None
        self._last_time_increment: float | None = None
        self._accepted_in_plane: NDArray | None = None
        self._accepted_cbb: NDArray | None = None
        self._accepted_cba: NDArray | None = None
        self._latest_cbb: NDArray | None = None
        self._latest_cba: NDArray | None = None

    @property
    def point_count(self) -> int:
        return self._bridge.point_count

    @property
    def backend_name(self) -> str:
        if self._bridge.supports_nonlocal_equivalent_plastic_strain:
            return "mfront-3d-condensed-plane-stress-micromorphic"
        return "mfront-3d-condensed-plane-stress"

    @property
    def thread_count(self) -> int:
        return self._bridge.thread_count

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

    @property
    def timing_statistics(self) -> MFrontTimingStatistics:
        bridge_timing = self._bridge.timing_statistics
        return MFrontTimingStatistics(
            rotation_to_material_seconds=bridge_timing.rotation_to_material_seconds,
            integration_seconds=bridge_timing.integration_seconds,
            rotation_to_global_seconds=bridge_timing.rotation_to_global_seconds,
            condensation_seconds=self._condensation_seconds,
            condition_check_seconds=self._condition_check_seconds,
            local_solve_seconds=self._local_solve_seconds,
            reconstruction_seconds=self._reconstruction_seconds,
            observable_seconds=self._observable_seconds,
            material_point_integrations=bridge_timing.material_point_integrations,
            material_point_integrations_with_tangent=(
                bridge_timing.material_point_integrations_with_tangent
            ),
            material_point_integrations_without_tangent=(
                bridge_timing.material_point_integrations_without_tangent
            ),
            material_block_integration_calls=(
                bridge_timing.material_block_integration_calls
            ),
            material_block_count=1,
            evaluate_calls=bridge_timing.evaluate_calls,
            condition_checks=self._condition_checks,
            local_condensation_evaluations=self._local_condensation_evaluations,
            full_batch_integration_calls=self._full_batch_integration_calls,
            equivalent_active_point_integrations=self._equivalent_active_point_integrations,
            local_iteration_histogram=tuple(
                int(value) for value in self._local_iteration_histogram
            ),
        )

    @property
    def local_condition_check_mode(self) -> LocalConditionCheckMode:
        return self._condition_check_mode

    @property
    def local_transverse_predictor(self) -> str:
        return self._local_transverse_predictor

    @property
    def warm_start_uses(self) -> int:
        return self._warm_start_uses

    @property
    def warm_start_resets(self) -> int:
        return self._warm_start_resets

    def accept_global_trial(self) -> None:
        """Accept the latest local condensation as the next Newton predictor."""

        if self._latest_transverse is not None:
            self._accepted_transverse = self._latest_transverse.copy()
            if self._last_in_plane is not None:
                self._accepted_in_plane = self._last_in_plane.copy()
            if self._latest_cbb is not None:
                self._accepted_cbb = self._latest_cbb.copy()
            if self._latest_cba is not None:
                self._accepted_cba = self._latest_cba.copy()
            self._has_accepted_global_trial = True

    def _reset_global_trial_predictor(self) -> None:
        self._accepted_transverse = self._bridge.committed_transverse_strain_kelvin.copy()
        self._latest_transverse = None
        self._has_accepted_global_trial = False
        self._accepted_in_plane = None
        self._accepted_cbb = None
        self._accepted_cba = None
        self._latest_cbb = None
        self._latest_cba = None
        self._warm_start_resets += 1

    def _fail(self, message: str) -> None:
        self._failures += 1
        self._bridge.revert()
        self._reset_global_trial_predictor()
        raise LocalPlaneStressConvergenceError(message)

    def reference_in_plane_tangent_mpa(self) -> NDArray:
        """Condensed elastic tangent in the GLOBAL frame, for hourglass control.

        Measured rather than reconstructed. A zero strain increment from the
        committed state leaves every behaviour in its elastic branch -- the
        crystal laws take their guarded no-slip branch, a J2 law has not
        yielded -- so the condensed tangent the bridge returns IS the elastic
        plane-stress operator, already rotated into the global frame by whatever
        orientation this batch carries.

        Rebuilding it instead from C11, C12 and C44 would mean restating the
        elasticity that already lives inside the MFront behaviour, and keeping
        the two in step by hand. The difference is not academic: for a crystal
        at 30 degrees the isotropic matrix gets the hourglass stiffness wrong by
        more than 10 percent, and nothing downstream would say so.

        The batch is left exactly as it was found: the probe is reverted.
        """

        probe = self.evaluate(
            np.zeros((self.point_count, 3)), time_increment=1.0, consistent_tangent=True
        )
        self.revert()
        tangent = probe.tangent_in_plane_mpa
        if tangent is None:  # pragma: no cover - requested explicitly above
            raise MFrontUnavailableError(
                f"{self._bridge.behaviour_name} returned no consistent tangent"
            )
        # One matrix per point today; the element needs one per element, and a
        # homogeneous orientation makes every point identical. A per-element
        # orientation will return the full stack instead.
        first = np.asarray(tangent[0], dtype=float)
        spread = float(np.abs(tangent - first).max())
        if spread > 1e-8 * float(np.abs(first).max()):
            raise ValueError(
                "the elastic reference tangent is not homogeneous over the batch "
                f"(spread {spread:.3e}); a per-element hourglass reference is needed"
            )
        return first

    def reference_full_tangent_kelvin_mpa(self) -> NDArray:
        """Measure the unloaded elastic 3D tangent in the global frame."""

        probe = self._bridge.evaluate(
            np.zeros((self.point_count, 6)), time_increment=1.0
        )
        self._bridge.revert()
        tangent = np.asarray(probe.consistent_tangent_kelvin_mpa, dtype=float).copy()
        tangent.setflags(write=False)
        return tangent

    def _check_cbb_condition(self, cbb: NDArray) -> NDArray:
        started = time.perf_counter()
        self._condition_checks += 1
        condition = np.linalg.cond(cbb)
        self._condition_check_seconds += time.perf_counter() - started
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
        return condition

    def evaluate(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> ConstitutiveTrial:
        result = self._evaluate_response(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
            response_level="complete",
        )
        if not isinstance(result, ConstitutiveTrial):
            raise RuntimeError("complete MFront response unexpectedly returned a light trial")
        return result

    def _evaluate_response(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
        response_level: ResponseLevel = "complete",
    ) -> InPlaneConstitutiveTrial | ConstitutiveTrial:
        in_plane = np.asarray(in_plane_strain, dtype=float)
        if in_plane.shape != (self.point_count, 3):
            raise ValueError(f"in_plane_strain must have shape {(self.point_count, 3)}")
        if not np.isfinite(in_plane).all():
            raise ValueError("in_plane_strain must be finite")
        if response_level not in {"residual", "tangent", "complete"}:
            raise ValueError("response_level must be 'residual', 'tangent', or 'complete'")
        self._last_in_plane = in_plane.copy()
        self._last_time_increment = float(time_increment)
        condensation_started = time.perf_counter()
        total_kelvin = np.zeros((self.point_count, 6), dtype=float)
        total_kelvin[:, _PLANE_STRESS_COMPONENTS] = in_plane * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
        transverse_initial = self._accepted_transverse
        if (
            self._local_transverse_predictor == "tangent"
            and self._accepted_in_plane is not None
            and self._accepted_cbb is not None
            and self._accepted_cba is not None
        ):
            delta_in_plane = (
                in_plane - self._accepted_in_plane
            ) * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
            try:
                transverse_correction = np.linalg.solve(
                    self._accepted_cbb,
                    -np.einsum("nij,nj->ni", self._accepted_cba, delta_in_plane)[..., None],
                )[..., 0]
                if np.isfinite(transverse_correction).all():
                    transverse_initial = self._accepted_transverse + transverse_correction
            except np.linalg.LinAlgError:
                transverse_initial = self._accepted_transverse
        if self._has_accepted_global_trial:
            self._warm_start_uses += 1
        total_kelvin[:, _TRANSVERSE_COMPONENTS_3D] = transverse_initial
        first_converged = np.zeros(self.point_count, dtype=int)
        final: _MFront3DTrial | None = None
        final_condition: NDArray | None = None
        previous_residual = np.full(self.point_count, np.inf)
        sample_condition = (
            self._condition_check_mode == "diagnostic_sample"
            and self._bridge.timing_statistics.evaluate_calls % 16 == 0
        )
        for iteration in range(1, self._maximum_iterations + 1):
            self._local_condensation_evaluations += 1
            self._full_batch_integration_calls += 1
            final = self._bridge.evaluate(
                total_kelvin,
                time_increment=time_increment,
                collect_observables=response_level == "complete",
            )
            stress_b = final.stress_kelvin_mpa[:, _TRANSVERSE_COMPONENTS_3D]
            if not np.isfinite(stress_b).all():
                self._fail("local plane-stress residual is non-finite")
            tangent = final.consistent_tangent_kelvin_mpa
            cbb = np.take(
                np.take(tangent, _TRANSVERSE_COMPONENTS_3D, axis=-2),
                _TRANSVERSE_COMPONENTS_3D,
                axis=-1,
            )
            stress_scale = np.maximum(
                1.0,
                np.max(np.abs(final.stress_kelvin_mpa), axis=1),
            )
            residual_norm = np.max(np.abs(stress_b), axis=1)
            converged = residual_norm <= (
                self._absolute_tolerance + self._relative_tolerance * stress_scale
            )
            self._equivalent_active_point_integrations += int(
                np.count_nonzero(~converged)
            )
            condition: NDArray | None = None
            if (
                self._condition_check_mode == "always"
                or sample_condition
                or np.any(residual_norm > previous_residual * (1.0 + 1.0e-12))
            ):
                condition = self._check_cbb_condition(cbb)
            first_converged[(first_converged == 0) & converged] = iteration
            if np.all(converged):
                final_condition = condition
                break
            previous_residual = residual_norm
            try:
                solve_started = time.perf_counter()
                correction = np.linalg.solve(cbb, -stress_b[..., None])[..., 0]
            except np.linalg.LinAlgError as error:
                self._check_cbb_condition(cbb)
                self._fail(f"failed to solve local Cbb system: {error}")
            self._local_solve_seconds += time.perf_counter() - solve_started
            correction[converged] = 0.0
            if not np.isfinite(correction).all():
                self._check_cbb_condition(cbb)
                self._fail("local plane-stress correction is non-finite")
            correction_limit = 1.0e6 * max(
                1.0,
                float(np.max(np.abs(total_kelvin[:, _TRANSVERSE_COMPONENTS_3D]))),
            )
            if float(np.max(np.abs(correction))) > correction_limit:
                self._check_cbb_condition(cbb)
                self._fail("local plane-stress correction is too large")
            total_kelvin[:, _TRANSVERSE_COMPONENTS_3D] += correction
        else:
            maximum_residual = float(np.max(np.abs(stress_b)))
            self._fail(
                f"local plane-stress Newton did not converge in "
                f"{self._maximum_iterations} iterations; residual={maximum_residual:.3e} MPa"
            )
        assert final is not None
        self._latest_cbb = np.take(
            np.take(final.consistent_tangent_kelvin_mpa, _TRANSVERSE_COMPONENTS_3D, axis=-2),
            _TRANSVERSE_COMPONENTS_3D,
            axis=-1,
        ).copy()
        self._latest_cba = np.take(
            np.take(final.consistent_tangent_kelvin_mpa, _TRANSVERSE_COMPONENTS_3D, axis=-2),
            _PLANE_STRESS_COMPONENTS,
            axis=-1,
        ).copy()
        first_iteration_values, first_iteration_counts = np.unique(
            first_converged,
            return_counts=True,
        )
        for iteration_value, count in zip(
            first_iteration_values,
            first_iteration_counts,
            strict=True,
        ):
            if 0 <= iteration_value < self._local_iteration_histogram.size:
                self._local_iteration_histogram[int(iteration_value)] += int(count)
        self._maximum_residual = max(
            self._maximum_residual,
            float(np.max(np.abs(stress_b))),
        )
        self._maximum_iterations_observed = max(
            self._maximum_iterations_observed,
            int(np.max(first_converged)),
        )
        self._iteration_sum += int(np.sum(first_converged))
        self._iteration_count += self.point_count
        tangent_engineering: NDArray | None = None
        if response_level != "residual" and consistent_tangent:
            tangent_engineering, _ = condense_kelvin_tangent_to_engineering(
                final.consistent_tangent_kelvin_mpa,
                check_condition=False,
            )
        if response_level != "complete":
            self._condensation_seconds += time.perf_counter() - condensation_started
            self._latest_transverse = total_kelvin[:, _TRANSVERSE_COMPONENTS_3D].copy()
            return InPlaneConstitutiveTrial(
                stress_in_plane_mpa=(
                    final.stress_kelvin_mpa[:, _PLANE_STRESS_COMPONENTS]
                    * _KELVIN_TO_ENGINEERING_STRESS_SCALE
                ),
                tangent_in_plane_mpa=(
                    tangent_engineering if response_level == "tangent" else None
                ),
                local_plane_stress_iterations=first_converged,
                cbb_condition_number=final_condition,
            )
        reconstruction_started = time.perf_counter()
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
        self._reconstruction_seconds += time.perf_counter() - reconstruction_started
        observable_started = time.perf_counter()
        observables = {
            "plastic_strain_2d": tensor_to_engineering_strain_2d(plastic_strain),
            # J2 scalars only when the behaviour actually has them. A
            # crystal law exposes twelve slips instead, and inventing a
            # scalar equivalent would let a consumer that needs a genuine
            # PEEQ silently accept a different quantity.
            **(
                {
                    "equivalent_plastic_strain": final.equivalent_plastic_strain,
                    "yield_surface_radius_mpa": final.yield_surface_radius_mpa,
                }
                if final.equivalent_plastic_strain.size
                else {}
            ),
            **final.observables,
        }
        self._observable_seconds += time.perf_counter() - observable_started
        self._condensation_seconds += time.perf_counter() - condensation_started
        self._latest_transverse = total_kelvin[:, _TRANSVERSE_COMPONENTS_3D].copy()
        return ConstitutiveTrial(
            stress_in_plane_mpa=tensor_to_engineering_stress_2d(full_stress),
            tangent_in_plane_mpa=tangent_engineering if consistent_tangent else None,
            full_stress_tensor_mpa=full_stress,
            full_strain_tensor=full_strain,
            elastic_strain_tensor=elastic_strain,
            plastic_strain_tensor=plastic_strain,
            plane_stress_residual_mpa=residual,
            observables=observables,
            local_plane_stress_iterations=first_converged,
            cbb_condition_number=final_condition,
        )

    def evaluate_in_plane(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> InPlaneConstitutiveTrial:
        return self._evaluate_response(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
            response_level="tangent",
        )

    def evaluate_in_plane_response(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        response_level: ResponseLevel,
        consistent_tangent: bool = True,
    ) -> InPlaneConstitutiveTrial:
        result = self._evaluate_response(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
            response_level=response_level,
        )
        return result

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

    def evaluate_nonlocal_state(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
    ) -> tuple[NDArray, NDArray]:
        trial = self.evaluate_in_plane(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=False,
        )
        return (
            trial.observables["equivalent_plastic_strain"],
            trial.observables["yield_surface_radius_mpa"],
        )

    def complete_trial(self, trial: InPlaneConstitutiveTrial) -> ConstitutiveTrial:
        if isinstance(trial, ConstitutiveTrial):
            return trial
        if self._last_in_plane is None or self._last_time_increment is None:
            raise TypeError("3D condensed trial is missing its reconstructed state")
        result = self._evaluate_response(
            self._last_in_plane,
            time_increment=self._last_time_increment,
            response_level="complete",
        )
        if not isinstance(result, ConstitutiveTrial):
            raise RuntimeError("complete MFront response unexpectedly returned a light trial")
        return result

    def commit(self) -> None:
        self._bridge.commit()
        self.accept_global_trial()
        self._latest_transverse = None
        self._has_accepted_global_trial = False

    def snapshot_state(self) -> MFrontCondensedStateSnapshot:
        """Capture committed MGIS and plane-stress predictor state."""

        if self._has_accepted_global_trial or self._latest_transverse is not None:
            raise RuntimeError("snapshot_state requires no active condensed trial")
        return MFrontCondensedStateSnapshot(
            bridge=self._bridge.snapshot_state(),
            accepted_transverse=self._accepted_transverse.copy(),
            latest_transverse=None,
            has_accepted_global_trial=False,
            last_in_plane=None if self._last_in_plane is None else self._last_in_plane.copy(),
            last_time_increment=self._last_time_increment,
            accepted_in_plane=(
                None if self._accepted_in_plane is None else self._accepted_in_plane.copy()
            ),
            accepted_cbb=None if self._accepted_cbb is None else self._accepted_cbb.copy(),
            accepted_cba=None if self._accepted_cba is None else self._accepted_cba.copy(),
        )

    def restore_state(self, snapshot: MFrontCondensedStateSnapshot) -> None:
        """Restore committed MGIS and plane-stress predictor state."""

        self._bridge.restore_state(snapshot.bridge)
        self._accepted_transverse = snapshot.accepted_transverse.copy()
        self._latest_transverse = (
            None if snapshot.latest_transverse is None else snapshot.latest_transverse.copy()
        )
        self._has_accepted_global_trial = snapshot.has_accepted_global_trial
        self._last_in_plane = (
            None if snapshot.last_in_plane is None else snapshot.last_in_plane.copy()
        )
        self._last_time_increment = snapshot.last_time_increment
        self._accepted_in_plane = (
            None if snapshot.accepted_in_plane is None else snapshot.accepted_in_plane.copy()
        )
        self._accepted_cbb = None if snapshot.accepted_cbb is None else snapshot.accepted_cbb.copy()
        self._accepted_cba = None if snapshot.accepted_cba is None else snapshot.accepted_cba.copy()
        self._latest_cbb = None
        self._latest_cba = None

    def revert(self) -> None:
        self._bridge.revert()
        self._reset_global_trial_predictor()

    def set_nonlocal_equivalent_plastic_strain(self, values: ArrayLike) -> None:
        self._bridge.set_nonlocal_equivalent_plastic_strain(values)

    @property
    def committed_nonlocal_equivalent_plastic_strain(self) -> NDArray:
        return self._bridge.committed_nonlocal_equivalent_plastic_strain


class MFrontNativeGeneralisedPlaneStressBatch:
    """Passive-bridge generalized-plane-stress crystal adapter (UMAT closure).

    This is deliberately separate from :class:`MFront3DCondensedPlaneStressBatch`.
    The latter remains the production/reference path.  The adapter runs
    :class:`Fcc316LForestRubinSrixGps`, whose local Newton carries the three
    plane-stress closure unknowns in the GLOBAL frame:

    - the nine components of ``Q_global_to_material`` are passed as per-point
      material properties, so the law can rotate the imposed gradient itself
      and enforce ``(Q^T sigma Q)_zz = (Q^T sigma Q)_xz = (Q^T sigma Q)_yz =
      0``;
    - the bridge applies NO gradient rotation (the law owns the rotation) and
      NO Python closure Newton;
    - the bridge rotates the returned stress, elastic strain and tangent back
      to the global frame, post-multiplies the tangent by the in-plane
      rotation operator (the DSL's automatic tangent differentiates the
      system as if every imposed gradient component entered the elastic
      residual, which only holds for the in-plane ones here), and discards
      the out-of-plane block.

    Transaction semantics (commit, revert, snapshot) are those of the other
    MGIS batches. The local transverse predictor options are accepted for
    interface compatibility but inert: the law's own Newton warm-starts from
    its committed closure state.
    """

    def __init__(
        self,
        library_path: str | Path,
        *,
        behaviour_spec: MFrontBehaviourSpec,
        point_count: int,
        rotation_global_to_material: ArrayLike | None = None,
        thread_count: int = 1,
        behaviour_name: str,
        behaviour_parameters: Mapping[str, float] | None = None,
        temperature_k: float = 293.15,
        maximum_local_iterations: int = 25,
        local_relative_tolerance: float = 1.0e-10,
        local_tolerance_mpa: float = 1.0e-8,
        local_transverse_predictor: str = "committed",
        local_condition_check_mode: str = "on_failure",
        shadow_tangent: bool = False,
        shadow_behaviour_name: str = "Fcc316LForestRubinSrix",
    ) -> None:
        if point_count < 1:
            raise ValueError("point_count must be positive")
        if thread_count < 1:
            raise ValueError("thread_count must be positive")
        self._mgis = _load_mgis()
        self._behaviour = self._mgis.load(
            str(Path(library_path).resolve()),
            behaviour_name,
            self._mgis.Hypothesis.Tridimensional,
        )
        _apply_behaviour_parameters(
            self._mgis, self._behaviour, behaviour_parameters, behaviour_name
        )
        self._specification = behaviour_spec
        self._behaviour_name = behaviour_name
        self._point_count = point_count
        self._thread_count = int(thread_count)
        self._parameters = dict(behaviour_parameters or {})
        self._temperature = float(temperature_k)
        self._maximum_iterations = int(maximum_local_iterations)
        self._relative_tolerance = float(local_relative_tolerance)
        self._absolute_tolerance = float(local_tolerance_mpa)
        if local_transverse_predictor not in {"committed", "tangent"}:
            raise ValueError("local_transverse_predictor must be 'committed' or 'tangent'")
        self._local_transverse_predictor = local_transverse_predictor
        self._rotations = (
            None
            if rotation_global_to_material is None
            else validate_rotations(rotation_global_to_material, point_count=point_count)
        )
        self._mgis_rotations = (
            None if self._rotations is None else mgis_rotation_argument(self._rotations)
        )
        self._manager = self._mgis.MaterialDataManager(self._behaviour, point_count)
        # The nine global-to-material rotation components are per-point
        # material properties: the law owns the strain rotation and needs Q to
        # assemble the closure residuals in the global frame.
        rotations = (
            np.broadcast_to(np.eye(3), (point_count, 3, 3)).copy()
            if self._rotations is None
            else self._rotations
        )
        storage_mode = self._mgis.MaterialStateManagerStorageMode.ExternalStorage
        for row in range(3):
            for column in range(3):
                name = f"Q{row + 1}{column + 1}"
                values = np.asarray(rotations[:, row, column], dtype=float)
                self._mgis.setMaterialProperty(self._manager.s0, name, values, storage_mode)
                self._mgis.setMaterialProperty(self._manager.s1, name, values, storage_mode)
        temperature_values = np.full(point_count, self._temperature)
        self._mgis.setExternalStateVariable(
            self._manager.s0, "Temperature", temperature_values, storage_mode
        )
        self._mgis.setExternalStateVariable(
            self._manager.s1, "Temperature", temperature_values, storage_mode
        )
        self._observable_slices = _declared_internal_slices(
            self._mgis,
            self._behaviour,
            self._mgis.Hypothesis.Tridimensional,
            behaviour_spec,
        )
        elastic_offset = _variable_offset(
            self._mgis,
            self._behaviour.isvs,
            "ElasticStrain",
            self._mgis.Hypothesis.Tridimensional,
            expected_size=6,
        )
        assert elastic_offset is not None
        self._elastic_offset = elastic_offset
        # Closure state variables: the global transverse strains solved by the
        # law's own Newton, read back to assemble the global total strain.
        closure_offsets = {
            name: _variable_offset(
                self._mgis,
                self._behaviour.isvs,
                name,
                self._mgis.Hypothesis.Tridimensional,
                expected_size=1,
            )
            for name in ("ezz", "eyz", "exz")
        }
        assert all(offset is not None for offset in closure_offsets.values())
        self._ezz_offset = closure_offsets["ezz"]
        self._eyz_offset = closure_offsets["eyz"]
        self._exz_offset = closure_offsets["exz"]
        self._committed_strain = np.zeros((point_count, 6), dtype=float)
        self._latest_trial: ConstitutiveTrial | None = None
        self._latest_total_kelvin: NDArray | None = None
        self._latest_in_plane: NDArray | None = None
        self._latest_dt: float | None = None
        self._evaluate_calls = 0
        self._internal_integrations = 0
        self._integration_seconds = 0.0
        self._local_iterations = np.zeros(point_count, dtype=np.int64)
        self._accepted_transverse = np.zeros((point_count, 3), dtype=float)
        self._accepted_in_plane: NDArray | None = None
        self._accepted_cbb: NDArray | None = None
        self._accepted_cba: NDArray | None = None
        self._latest_transverse: NDArray | None = None
        self._latest_cbb: NDArray | None = None
        self._latest_cba: NDArray | None = None
        self._warm_start_uses = 0
        self._warm_start_resets = 0
        self._maximum_residual = 0.0
        self._last_local_failure: dict[str, object] | None = None
        # Sub-stepping bound. Eight halvings take a full increment down to
        # 1/256 of itself; beyond that a failure is not a step-size problem.
        self._maximum_substeps = 256
        self._substep_uses = 0
        self._rerun_uses = 0
        self._rerun_failures = 0
        self._substep_divisions_max = 0
        # Route 2 of the handoff, and the reason it is needed. Sub-stepping is
        # what makes the joint Newton reach the deep plastic states at all, but
        # the matrix it leaves behind is the LAST sub-step's, not the whole
        # increment's, and A6 then fails by a factor of two to five. Route 1
        # (rerun the full increment from the located root, through a predictor
        # external state variable) was implemented and measured to leave the
        # tangent BIT-IDENTICAL -- the mechanism never fires -- so the
        # conclusion drawn from it, that the DSL tangent machinery is at fault,
        # is not supported. Measured instead: the RAW 3D law's consistent
        # tangent is exact to 1e-10 at every depth of the frozen history, and
        # the GPS tangent is exact wherever no sub-stepping happens (7e-08 at
        # increment 1, 1e-07 at 2, 1.8e-06 at 3). The chain is right; only the
        # sub-stepped matrix is wrong.
        #
        # Route 2 was then implemented on that basis and MEASURED TO FAIL, so
        # it is off by default and kept only as the record. The premise was
        # that imposing the located transverse strain turns the problem back
        # into the plain 18-unknown one, whose tangent is the qualified one.
        # It does not: driven in lockstep from the same committed state with
        # exactly the GPS transverse strain imposed, the raw law converges to
        # the OTHER root -- sigma_zz of -152 MPa at increment 2, growing to
        # -4221 by increment 8, against the 0 the closure enforces. That is the
        # multiple-root structure of
        # `validation/srix_plane_stress_branch_diagnostic.md` reappearing
        # inside the 18-unknown problem: imposing the strain does not select
        # the branch. Any future route must carry the branch, not just the
        # strain.
        self._shadow: MFront3DMaterialPointBatch | None = None
        self._shadow_failures = 0
        self._shadow_has_trial = False
        # tr(eel) = tr(eps_total) is the trace of the elastic residual: the
        # Schmid tensors are deviatoric and a rotation preserves the trace, so
        # a converged state that violates it is not a solution of the system
        # the law says it solved. Cheap enough to check on every evaluation,
        # and it is the invariant that exposed the overwrite defect.
        self._maximum_kinematic_defect = 0.0
        if shadow_tangent:
            self._shadow = MFront3DMaterialPointBatch(
                library_path,
                behaviour_spec=behaviour_spec,
                point_count=point_count,
                rotation_global_to_material=self._rotations,
                thread_count=thread_count,
                behaviour_name=shadow_behaviour_name,
                behaviour_parameters=behaviour_parameters,
                temperature_k=temperature_k,
            )

    @property
    def substep_uses(self) -> int:
        """How many increments needed halving. Zero is the healthy case."""

        return self._substep_uses

    @property
    def substep_divisions_max(self) -> int:
        return self._substep_divisions_max

    @property
    def point_count(self) -> int:
        return self._point_count

    @property
    def backend_name(self) -> str:
        return "mfront-native-generalised-plane-stress"

    @property
    def completion_strategy(self) -> str:
        return "mfront_native_generalised_plane_stress"

    @property
    def linear_system_matrix_type(self) -> LinearSystemMatrixType:
        return self._specification.linear_system_matrix_type

    @property
    def thread_count(self) -> int:
        return self._thread_count

    @property
    def statistics(self) -> PlaneStressBatchStatistics:
        return PlaneStressBatchStatistics(
            maximum_gauss_point_plane_stress_residual_mpa=self._maximum_residual,
            maximum_local_plane_stress_iterations=int(self._local_iterations.max()),
            mean_local_plane_stress_iterations=float(self._local_iterations.mean()),
        )

    @property
    def timing_statistics(self) -> MFrontTimingStatistics:
        return MFrontTimingStatistics(
            integration_seconds=self._integration_seconds,
            evaluate_calls=self._evaluate_calls,
            material_point_integrations=self._internal_integrations,
            material_point_integrations_with_tangent=self._internal_integrations,
            material_block_integration_calls=self._evaluate_calls,
            material_block_count=1,
            native_batch_calls=self._evaluate_calls,
            native_material_points=self._evaluate_calls * self._point_count,
            native_internal_integrations=self._internal_integrations,
            native_total_local_iterations=int(np.sum(self._local_iterations)),
            native_thread_count=self._thread_count,
        )

    @property
    def committed_transverse_strain_kelvin(self) -> NDArray:
        return self._committed_strain[:, _TRANSVERSE_COMPONENTS_3D].copy()

    @property
    def local_transverse_predictor(self) -> str:
        return self._local_transverse_predictor

    @property
    def warm_start_uses(self) -> int:
        return self._warm_start_uses

    @property
    def warm_start_resets(self) -> int:
        return self._warm_start_resets

    @property
    def last_local_failure(self) -> dict[str, object] | None:
        return None if self._last_local_failure is None else dict(self._last_local_failure)

    def _set_parameters(self) -> None:
        _apply_behaviour_parameters(
            self._mgis, self._behaviour, self._parameters, self._behaviour_name
        )

    def _integrate_once(self, in_plane_kelvin: NDArray, time_increment: float) -> int:
        """One integration, with the transverse strain kept in the gradient.

        The law adds its closure unknowns to the gradient it is given rather
        than replacing three of its components, so the transverse strain the
        bridge supplies IS part of the imposed gradient. On success the
        increment the closure found is folded back into `s1.gradients`, which
        makes `s1.gradients` the true total strain at every point of the
        transaction: `mgis.update` then carries it into the committed state,
        `mgis.revert` discards it with everything else, and every sub-step
        starts from the transverse strain the previous one reached.

        This is the whole point of the change of 2026-08-07. While the law
        overwrote the transverse components instead, MGIS held a gradient that
        was fictitious on three components; nothing could cross-check the
        state, and the recorded transverse strain drifted from the integrated
        one by a factor equal to the increment number.
        """

        # `in_plane_kelvin` is the TOTAL in-plane strain, not an increment --
        # that is the contract of `evaluate` for every plane-stress batch in
        # this module, and the reference sets the gradient absolutely. This
        # bridge used to write `s0.gradients + gradient`, so it applied the
        # total as if it were an increment and the imposed strain accumulated
        # as 1+2+3+... instead of 1,2,3: at increment 2 the in-plane trace was
        # 3.0e-3 where 2.0e-3 was asked. Increment 1 was unaffected -- total
        # and increment coincide there -- which is exactly why every
        # comparison against the reference agreed at increment 1 and diverged
        # from increment 2.
        gradients = np.asarray(self._manager.s0.gradients).copy()
        gradients[:, _PLANE_STRESS_COMPONENTS] = in_plane_kelvin
        self._manager.s1.gradients[:, :] = gradients
        status = int(
            self._mgis.integrate(
                self._manager,
                self._mgis.IntegrationType.IntegrationWithConsistentTangentOperator,
                float(time_increment),
                0,
                self._point_count,
            )
        )
        if status == 1:
            closure = np.array(
                [self._ezz_offset, self._exz_offset, self._eyz_offset], dtype=int
            )
            increment = (
                np.asarray(self._manager.s1.internal_state_variables)[:, closure]
                - np.asarray(self._manager.s0.internal_state_variables)[:, closure]
            )
            self._manager.s1.gradients[:, _TRANSVERSE_COMPONENTS_3D] += increment
        return status

    def _set_closure_predictor(self, values: NDArray) -> None:
        """Set the closure-root predictor external state variables (n, 3).

        Zero means "no location available" and lets the law's physics-based
        guess apply.
        """

        storage = self._mgis.MaterialStateManagerStorageMode.ExternalStorage
        for index, name in enumerate(
            ("GpsPredictorEzz", "GpsPredictorEyz", "GpsPredictorExz")
        ):
            column = np.ascontiguousarray(np.asarray(values[:, index], dtype=float))
            self._mgis.setExternalStateVariable(self._manager.s0, name, column, storage)
            self._mgis.setExternalStateVariable(self._manager.s1, name, column, storage)

    def _rerun_full_increment_from_located_root(
        self, in_plane_kelvin: NDArray, time_increment: float
    ) -> int:
        """Route 1 of the handoff: rerun the full increment from the root the
        sub-stepping located.

        The sub-stepped result is exact in the stress, the closure and the
        internal variables, but its Newton matrix is that of the LAST
        sub-step, not of the whole increment. Rerunning the full increment
        with the located closure increments imposed through the predictor
        external state variables gives the exact consistent tangent. If the
        rerun fails, the sub-stepping is redone (deterministic) and its
        result kept, restoring the previous behaviour.
        """

        closure_columns = np.array(
            [self._ezz_offset, self._eyz_offset, self._exz_offset], dtype=int
        )
        located = (
            np.asarray(self._manager.s1.internal_state_variables)[:, closure_columns]
            - np.asarray(self._manager.s0.internal_state_variables)[:, closure_columns]
        )
        snapshot = self._committed_snapshot()
        self._restore_committed(snapshot)
        self._mgis.revert(self._manager)
        self._set_closure_predictor(located)
        status = self._integrate_once(in_plane_kelvin, time_increment)
        if status == 1:
            return 1
        # Fallback: deterministic redo of the sub-stepping.
        self._restore_committed(snapshot)
        self._mgis.revert(self._manager)
        self._set_closure_predictor(np.zeros((self._point_count, 3)))
        status, _ = self._integrate_with_substepping(in_plane_kelvin, time_increment)
        return status

    def _shadow_condensed_tangent(
        self,
        in_plane: NDArray,
        internal_state_variables: NDArray,
        time_increment: float,
    ) -> NDArray | None:
        """Exact plane-stress tangent of the whole increment, or `None`.

        The closure has just located the transverse strain. Imposing it turns
        the problem back into the plain 18-unknown one the raw law solves, and
        that law's consistent tangent is exact (measured to `1e-10` at every
        depth of the frozen history). Condensing it with the Schur complement
        the reference already uses gives the plane-stress tangent of the whole
        increment -- which is precisely what sub-stepping cannot supply.

        The transverse components are read from the closure state variables in
        the STORAGE the law itself used: it writes them straight into Kelvin
        slots 2, 4 and 5 of the gradient, so they are Kelvin and take no
        `sqrt(2)`. (The reconstruction of the reported total strain further
        down divides by `sqrt(2)`; the two cannot both be right, and the law's
        own substitution is the authority. That inconsistency is real and is
        recorded rather than silently aligned.)

        Returns `None` on any shadow failure, leaving the sub-stepped tangent
        in place: an approximate Newton matrix is worse than an exact one and
        better than an exception.
        """

        if self._shadow is None:
            return None
        total = np.zeros((self._point_count, 6), dtype=float)
        total[:, _PLANE_STRESS_COMPONENTS] = (
            in_plane * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
        )
        total[:, _TRANSVERSE_COMPONENTS_3D] = internal_state_variables[
            :,
            np.array(
                [self._ezz_offset, self._exz_offset, self._eyz_offset], dtype=int
            ),
        ]
        try:
            trial = self._shadow.evaluate(total, time_increment=time_increment)
            engineering, _ = condense_kelvin_tangent_to_engineering(
                trial.consistent_tangent_kelvin_mpa, check_condition=False
            )
        except Exception:
            self._shadow_failures += 1
            self._shadow_has_trial = False
            return None
        if not np.isfinite(engineering).all():
            self._shadow_failures += 1
            self._shadow.revert()
            self._shadow_has_trial = False
            return None
        self._shadow_has_trial = True
        return np.asarray(engineering, dtype=float)

    @property
    def shadow_failures(self) -> int:
        return self._shadow_failures

    @property
    def maximum_kinematic_defect(self) -> float:
        """Worst relative violation of `tr(eel) = tr(eps_total)` so far."""

        return self._maximum_kinematic_defect

    def _committed_snapshot(self) -> tuple[NDArray, NDArray, NDArray]:
        return (
            np.asarray(self._manager.s0.gradients).copy(),
            np.asarray(self._manager.s0.thermodynamic_forces).copy(),
            np.asarray(self._manager.s0.internal_state_variables).copy(),
        )

    def _restore_committed(self, snapshot: tuple[NDArray, NDArray, NDArray]) -> None:
        gradients, forces, internal = snapshot
        self._manager.s0.gradients[:, :] = gradients
        self._manager.s0.thermodynamic_forces[:, :] = forces
        self._manager.s0.internal_state_variables[:, :] = internal

    def _integrate_with_substepping(
        self, in_plane_kelvin: NDArray, time_increment: float
    ) -> tuple[int, int]:
        """Integrate one increment, halving it as often as needed.

        The 21-unknown joint Newton of the GPS law has no line search and no
        step control -- the Implicit DSL offers neither -- so it fails on large
        plastic increments even though the root is there. Measured by
        `scripts/diagnose_srix_closure_root_sweep.py`: the closure equation
        `sigma_zz(eps_zz) = 0` has **exactly one** root at every increment of
        the frozen history, increment 8 included, and the roots march at a
        regular `-0.001` per increment. There is no fold and no branch to lose;
        what fails is the iteration, not the problem. Halving the increment is
        the standard remedy and the only one that addresses the actual cause.

        Sub-stepping advances `s0` internally and then puts it back, so
        `commit()` and `revert()` keep their meaning for the caller: `s1` holds
        the end state, `s0` the true committed one.

        The returned tangent is that of the LAST sub-step, not of the whole
        increment. That is the usual compromise -- the stress, the closure and
        the internal variables are exact; only the Newton matrix is
        approximate, and the global iteration corrects for it.
        """

        status = self._integrate_once(in_plane_kelvin, time_increment)
        if status == 1 or self._maximum_substeps <= 1:
            return status, 1

        snapshot = self._committed_snapshot()
        committed_in_plane = snapshot[0][:, _PLANE_STRESS_COMPONENTS].copy()
        divisions = 2
        while divisions <= self._maximum_substeps:
            self._restore_committed(snapshot)
            self._mgis.revert(self._manager)
            step = time_increment / divisions
            succeeded = True
            for index in range(divisions):
                # Interpolate the TOTAL in-plane strain between the committed
                # state and the target; a fraction of a total is not a strain.
                weight = (index + 1) / divisions
                stage = committed_in_plane + weight * (
                    in_plane_kelvin - committed_in_plane
                )
                status = self._integrate_once(stage, step)
                if status != 1:
                    succeeded = False
                    break
                if index < divisions - 1:
                    self._mgis.update(self._manager)
            if succeeded:
                # `s1` is the end state; `s0` goes back to the real committed
                # state so the caller's transaction is unchanged.
                self._restore_committed(snapshot)
                self._substep_uses += 1
                self._substep_divisions_max = max(self._substep_divisions_max, divisions)
                return 1, divisions
            divisions *= 2

        self._restore_committed(snapshot)
        self._mgis.revert(self._manager)
        return status, divisions // 2

    def evaluate_in_plane(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> InPlaneConstitutiveTrial:
        full = self.evaluate(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
        )
        return InPlaneConstitutiveTrial(
            stress_in_plane_mpa=full.stress_in_plane_mpa,
            tangent_in_plane_mpa=full.tangent_in_plane_mpa,
            observables=full.observables,
            local_plane_stress_iterations=full.local_plane_stress_iterations,
        )

    def evaluate(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> ConstitutiveTrial:
        in_plane = np.asarray(in_plane_strain, dtype=float)
        if in_plane.shape != (self._point_count, 3):
            raise ValueError(f"in_plane_strain must have shape {(self._point_count, 3)}")
        # The closure state variables own the transverse strains: the law
        # overwrites the transverse gradient components with (dezz, deyz,
        # dexz) before rotating. The bridge passes the in-plane gradient in
        # Kelvin storage (the repository convention for MGIS arrays) and zero
        # transverse values, which the law never reads.
        in_plane_kelvin = in_plane * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
        self._set_parameters()
        started = time.perf_counter()
        # d(deto_m)/ddeto restricted to the in-plane columns, in the MGIS
        # storage: the rotation operator of the stock gradient rotation,
        # with the transverse columns zeroed. The DSL's automatic tangent
        # differentiates the local system as if every imposed gradient
        # component entered the elastic residual; here only the in-plane
        # ones do, so the returned tangent is post-multiplied by this
        # operator.
        in_plane_operator: NDArray
        if self._mgis_rotations is None:
            in_plane_operator = np.eye(6, dtype=float)
            in_plane_operator[:, _TRANSVERSE_COMPONENTS_3D] = 0.0
        else:
            identity_block = np.tile(np.eye(6, dtype=float), (self._point_count, 1))
            rotated_block = np.ascontiguousarray(identity_block.reshape(-1).copy())
            # Each virtual point (one per unit column) carries the rotation of
            # its parent material point: repeat per row, never element-wise.
            rotations_expanded = np.repeat(
                self._mgis_rotations.reshape(self._point_count, 9), 6, axis=0
            ).reshape(-1)
            self._mgis.rotateGradients(rotated_block, self._behaviour, rotations_expanded)
            # The MGIS output rows are the rotated unit vectors; the
            # derivative operator d(deto_m)/ddeto has them as COLUMNS, so the
            # output is transposed. At the identity this is a no-op, which is
            # why the error only showed up on rotated orientations.
            in_plane_operator = rotated_block.reshape(self._point_count, 6, 6).transpose(
                0, 2, 1
            )
            in_plane_operator[:, :, _TRANSVERSE_COMPONENTS_3D] = 0.0
        self._last_local_failure = None
        # The closure-root predictor is reset at every evaluation; the
        # sub-stepping path sets it to the located root (route 1).
        self._set_closure_predictor(np.zeros((self._point_count, 3)))
        status, divisions = self._integrate_with_substepping(
            in_plane_kelvin, time_increment
        )
        if status != 1:
            self._last_local_failure = {
                "status": int(status),
                "substep_divisions": int(divisions),
            }
            self.revert()
            raise MFrontIntegrationError(
                f"GPS UMAT integration failed with status {status} after "
                f"sub-stepping down to 1/{divisions} of the increment"
            )
        if divisions > 1:
            # Sub-stepping located the closure root; rerun the full increment
            # from it so the returned tangent is the exact one of the whole
            # increment, not of the last sub-step.
            self._rerun_uses += 1
            status = self._rerun_full_increment_from_located_root(
                in_plane_kelvin, time_increment
            )
            if status != 1:
                self._rerun_failures += 1
                self._last_local_failure = {
                    "status": int(status),
                    "substep_divisions": int(divisions),
                    "note": "full-increment rerun from the located root failed; "
                    "kept the sub-stepped result with its last-sub-step tangent",
                }
        self._integration_seconds += time.perf_counter() - started
        self._evaluate_calls += 1
        internal_state_variables = np.asarray(
            self._manager.s1.internal_state_variables
        ).copy()
        stress = np.asarray(self._manager.s1.thermodynamic_forces).copy()
        elastic = internal_state_variables[
            :, self._elastic_offset : self._elastic_offset + 6
        ].copy()
        tangent = np.asarray(self._manager.K).copy()
        # Output rotations: stress and elastic strain back to the global frame,
        # the tangent one-sided (its strain indices already are global), then
        # the in-plane operator correction.
        tangent = tangent @ in_plane_operator
        if self._mgis_rotations is not None:
            for tensor in (stress, elastic):
                flat = np.ascontiguousarray(tensor.reshape(-1))
                self._mgis.rotateThermodynamicForces(
                    flat, self._behaviour, self._mgis_rotations
                )
                tensor[:, :] = flat.reshape(self._point_count, 6)
            for column in range(6):
                flat = np.ascontiguousarray(tangent[:, :, column].reshape(-1))
                self._mgis.rotateThermodynamicForces(
                    flat, self._behaviour, self._mgis_rotations
                )
                tangent[:, :, column] = flat.reshape(self._point_count, 6)
        shadow_tangent_engineering = self._shadow_condensed_tangent(
            in_plane, internal_state_variables, time_increment
        )
        # Global total strain: read straight off the gradient, which now
        # carries the transverse strain too. There is no reconstruction from
        # the closure state variables any more, and with it goes the
        # engineering-versus-Kelvin ambiguity that reconstruction carried --
        # the gradient has one storage convention and MGIS owns it.
        total = np.asarray(self._manager.s1.gradients).copy()
        transverse = total[:, _TRANSVERSE_COMPONENTS_3D].copy()
        kinematic_defect = np.abs(
            elastic[:, :3].sum(axis=1) - total[:, :3].sum(axis=1)
        ) / np.maximum(np.abs(total[:, :3].sum(axis=1)), 1.0e-30)
        self._maximum_kinematic_defect = max(
            self._maximum_kinematic_defect, float(kinematic_defect.max())
        )
        observables = {
            name: internal_state_variables[:, item].copy()
            for name, item in self._observable_slices.items()
        }
        if "equivalent_plastic_slip" in observables:
            observables["accumulated_slip"] = observables["equivalent_plastic_slip"].sum(axis=1)
        total_tensor = kelvin_3d_to_tensor(total, quantity="strain")
        elastic_tensor = kelvin_3d_to_tensor(elastic, quantity="strain")
        stress_tensor = kelvin_3d_to_tensor(stress, quantity="stress")
        plastic_tensor = total_tensor - elastic_tensor
        condensed_tangent = tangent[:, _PLANE_STRESS_COMPONENTS][
            :, :, _PLANE_STRESS_COMPONENTS
        ].copy()
        in_tangent = condensed_tangent.copy()
        in_tangent = in_tangent * _KELVIN_TO_ENGINEERING_STRESS_SCALE[None, :, None]
        in_tangent = in_tangent * _ENGINEERING_TO_KELVIN_STRAIN_SCALE[None, None, :]
        if shadow_tangent_engineering is not None:
            in_tangent = shadow_tangent_engineering
        in_stress = stress[:, _PLANE_STRESS_COMPONENTS] * _KELVIN_TO_ENGINEERING_STRESS_SCALE
        residual = np.stack(
            (stress_tensor[:, 2, 2], stress_tensor[:, 0, 2], stress_tensor[:, 1, 2]),
            axis=-1,
        )
        self._maximum_residual = max(
            self._maximum_residual, float(np.max(np.abs(residual)))
        )
        self._latest_in_plane = in_plane.copy()
        self._latest_dt = float(time_increment)
        self._latest_total_kelvin = total.copy()
        self._latest_transverse = transverse.copy()
        self._latest_cbb = tangent[:, _TRANSVERSE_COMPONENTS_3D][
            :, :, _TRANSVERSE_COMPONENTS_3D
        ].copy()
        self._latest_cba = tangent[:, _TRANSVERSE_COMPONENTS_3D][
            :, :, _PLANE_STRESS_COMPONENTS
        ].copy()
        # The law's internal Newton is not observable through stock MGIS; the
        # closure is part of it, so the reported count is the law's own
        # integration (one per point), documented as such.
        self._local_iterations[:] = 1
        self._internal_integrations += self._point_count
        self._latest_trial = ConstitutiveTrial(
            stress_in_plane_mpa=in_stress,
            tangent_in_plane_mpa=in_tangent if consistent_tangent else None,
            full_stress_tensor_mpa=stress_tensor,
            full_strain_tensor=total_tensor,
            elastic_strain_tensor=elastic_tensor,
            plastic_strain_tensor=plastic_tensor,
            plane_stress_residual_mpa=residual,
            observables=observables,
            local_plane_stress_iterations=self._local_iterations.astype(float),
        )
        return self._latest_trial

    def commit(self) -> None:
        self.accept_global_trial()
        self._mgis.update(self._manager)
        if self._shadow is not None and self._shadow_has_trial:
            # The shadow tracks the SAME branch, so its committed state must
            # advance with the closure's or its tangent would be evaluated
            # from a state the law never reached. A shadow that failed has no
            # trial to commit; the closure's own state is unaffected either
            # way, so the increment is still accepted.
            self._shadow.commit()
            self._shadow_has_trial = False
        if self._latest_total_kelvin is not None:
            self._committed_strain[:, :] = self._latest_total_kelvin
        self._latest_trial = None
        self._latest_total_kelvin = None
        self._latest_transverse = None
        self._latest_cbb = None
        self._latest_cba = None

    def revert(self) -> None:
        self._mgis.revert(self._manager)
        if self._shadow is not None and self._shadow_has_trial:
            self._shadow.revert()
            self._shadow_has_trial = False
        self._latest_trial = None
        self._latest_total_kelvin = None
        self._latest_transverse = None
        self._latest_cbb = None
        self._latest_cba = None
        self._accepted_transverse = self._committed_strain[:, _TRANSVERSE_COMPONENTS_3D].copy()
        self._accepted_in_plane = None
        self._accepted_cbb = None
        self._accepted_cba = None
        self._warm_start_resets += 1

    def accept_global_trial(self) -> None:
        if self._latest_transverse is not None:
            self._accepted_transverse = self._latest_transverse.copy()
            self._accepted_in_plane = (
                None if self._latest_in_plane is None else self._latest_in_plane.copy()
            )
            self._accepted_cbb = None if self._latest_cbb is None else self._latest_cbb.copy()
            self._accepted_cba = None if self._latest_cba is None else self._latest_cba.copy()

    def snapshot_state(self) -> tuple[Any, ...]:
        return (
            np.asarray(self._manager.s0.gradients, dtype=float).copy(),
            np.asarray(self._manager.s0.internal_state_variables, dtype=float).copy(),
            np.asarray(self._manager.s0.thermodynamic_forces, dtype=float).copy(),
            self._committed_strain.copy(),
            self._accepted_transverse.copy(),
            None if self._accepted_in_plane is None else self._accepted_in_plane.copy(),
            None if self._accepted_cbb is None else self._accepted_cbb.copy(),
            None if self._accepted_cba is None else self._accepted_cba.copy(),
        )

    def restore_state(self, snapshot: Any) -> None:
        self.revert()
        for state in (self._manager.s0, self._manager.s1):
            state.gradients[:, :] = snapshot[0]
            state.internal_state_variables[:, :] = snapshot[1]
            state.thermodynamic_forces[:, :] = snapshot[2]
        self._committed_strain[:, :] = snapshot[3]
        self._accepted_transverse[:, :] = snapshot[4]
        self._accepted_in_plane = (
            None if snapshot[5] is None else np.asarray(snapshot[5]).copy()
        )
        self._accepted_cbb = None if snapshot[6] is None else np.asarray(snapshot[6]).copy()
        self._accepted_cba = None if snapshot[7] is None else np.asarray(snapshot[7]).copy()
        self._latest_trial = None
        self._latest_transverse = None
        self._latest_cbb = None
        self._latest_cba = None

    def complete_trial(self, trial: InPlaneConstitutiveTrial) -> ConstitutiveTrial:
        if self._latest_trial is None:
            raise RuntimeError("no native generalized plane-stress trial is available")
        return self._latest_trial
class MFront3DCondensedPlaneStressBlockBatch:
    """Partition a condensed MFront batch into independently converged blocks.

    The constitutive law is unchanged.  Each block owns its MGIS manager, so a
    difficult grain only keeps the points in its own block active during later
    local plane-stress iterations.  Blocks are evaluated sequentially; this
    first implementation therefore prioritizes transaction safety and a
    measurable active-block baseline over parallel scheduling.
    """

    def __init__(
        self,
        *args: Any,
        condensation_block_size: int = 2500,
        **kwargs: Any,
    ) -> None:
        if isinstance(condensation_block_size, bool) or condensation_block_size < 1:
            raise ValueError("condensation_block_size must be a positive integer")
        point_count = kwargs.get("point_count")
        if point_count is None:
            for name in (
                "initial_yield_stress_mpa",
                "hardening_coefficient_mpa",
                "hardening_exponent",
            ):
                value = kwargs.get(name)
                if value is not None and np.asarray(value).ndim > 0:
                    point_count = int(np.asarray(value).size)
                    break
        if point_count is None:
            for value in args[1:4]:
                if np.asarray(value).ndim > 0:
                    point_count = int(np.asarray(value).size)
                    break
        if point_count is None:
            raise ValueError("block condensation requires an explicit point_count")
        self._point_count = int(point_count)
        if self._point_count < 1:
            raise ValueError("point_count must be positive")
        self._block_size = int(condensation_block_size)
        self._slices = tuple(
            slice(start, min(start + self._block_size, self._point_count))
            for start in range(0, self._point_count, self._block_size)
        )
        self._blocks: list[MFront3DCondensedPlaneStressBatch] = []
        self._last_light_trials: list[InPlaneConstitutiveTrial] | None = None
        self._last_light_trial: InPlaneConstitutiveTrial | None = None

        rotation = kwargs.get("rotation_global_to_material")
        material_properties = kwargs.get("material_property_values")
        coupling = kwargs.get("micromorphic_coupling_modulus_mpa")
        for block_slice in self._slices:
            block_args = list(args)
            for index in range(1, min(4, len(block_args))):
                block_args[index] = self._slice_point_value(block_args[index], block_slice)
            block_kwargs = dict(kwargs)
            block_kwargs["point_count"] = block_slice.stop - block_slice.start
            for name in (
                "initial_yield_stress_mpa",
                "hardening_coefficient_mpa",
                "hardening_exponent",
            ):
                if name in block_kwargs:
                    block_kwargs[name] = self._slice_point_value(
                        block_kwargs[name], block_slice
                    )
            if rotation is not None:
                block_kwargs["rotation_global_to_material"] = np.asarray(rotation)[block_slice]
            if material_properties is not None:
                block_kwargs["material_property_values"] = {
                    name: self._slice_point_value(value, block_slice)
                    for name, value in material_properties.items()
                }
            if coupling is not None:
                block_kwargs["micromorphic_coupling_modulus_mpa"] = self._slice_point_value(
                    coupling, block_slice
                )
            self._blocks.append(
                MFront3DCondensedPlaneStressBatch(*block_args, **block_kwargs)
            )

    @staticmethod
    def _slice_point_value(value: Any, block_slice: slice) -> Any:
        values = np.asarray(value)
        if values.ndim == 0:
            return value
        return values[block_slice]

    @property
    def point_count(self) -> int:
        return self._point_count

    @property
    def backend_name(self) -> str:
        return f"{self._blocks[0].backend_name}-blocks"

    @property
    def thread_count(self) -> int:
        return self._blocks[0].thread_count

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def block_count(self) -> int:
        return len(self._blocks)

    @property
    def completion_strategy(self) -> str:
        return "mfront_3d_local_condensation_blocks"

    @property
    def linear_system_matrix_type(self) -> LinearSystemMatrixType:
        return self._blocks[0].linear_system_matrix_type

    @property
    def local_condition_check_mode(self) -> LocalConditionCheckMode:
        return self._blocks[0].local_condition_check_mode

    @property
    def local_transverse_predictor(self) -> str:
        return self._blocks[0].local_transverse_predictor

    @property
    def warm_start_uses(self) -> int:
        return sum(block.warm_start_uses for block in self._blocks)

    @property
    def warm_start_resets(self) -> int:
        return sum(block.warm_start_resets for block in self._blocks)

    @property
    def statistics(self) -> PlaneStressBatchStatistics:
        statistics = [block.statistics for block in self._blocks]
        total_points = float(self._point_count)
        weighted_mean = sum(
            stat.mean_local_plane_stress_iterations * block.point_count
            for block, stat in zip(self._blocks, statistics, strict=True)
        ) / total_points
        return PlaneStressBatchStatistics(
            maximum_gauss_point_plane_stress_residual_mpa=max(
                stat.maximum_gauss_point_plane_stress_residual_mpa for stat in statistics
            ),
            maximum_local_plane_stress_iterations=max(
                stat.maximum_local_plane_stress_iterations for stat in statistics
            ),
            mean_local_plane_stress_iterations=weighted_mean,
            local_plane_stress_failures=sum(
                stat.local_plane_stress_failures for stat in statistics
            ),
            maximum_cbb_condition_number=max(
                stat.maximum_cbb_condition_number for stat in statistics
            ),
        )

    @property
    def timing_statistics(self) -> MFrontTimingStatistics:
        timings = [block.timing_statistics for block in self._blocks]
        histogram_size = max((len(t.local_iteration_histogram) for t in timings), default=0)
        histogram = [0] * histogram_size
        for timing in timings:
            for index, count in enumerate(timing.local_iteration_histogram):
                histogram[index] += count
        return MFrontTimingStatistics(
            rotation_to_material_seconds=sum(t.rotation_to_material_seconds for t in timings),
            integration_seconds=sum(t.integration_seconds for t in timings),
            rotation_to_global_seconds=sum(t.rotation_to_global_seconds for t in timings),
            condensation_seconds=sum(t.condensation_seconds for t in timings),
            condition_check_seconds=sum(t.condition_check_seconds for t in timings),
            local_solve_seconds=sum(t.local_solve_seconds for t in timings),
            reconstruction_seconds=sum(t.reconstruction_seconds for t in timings),
            observable_seconds=sum(t.observable_seconds for t in timings),
            material_point_integrations=sum(
                t.material_point_integrations for t in timings
            ),
            material_point_integrations_with_tangent=sum(
                t.material_point_integrations_with_tangent for t in timings
            ),
            material_point_integrations_without_tangent=sum(
                t.material_point_integrations_without_tangent for t in timings
            ),
            material_block_integration_calls=sum(
                t.material_block_integration_calls for t in timings
            ),
            material_block_count=sum(t.material_block_count for t in timings),
            evaluate_calls=sum(t.evaluate_calls for t in timings),
            condition_checks=sum(t.condition_checks for t in timings),
            local_condensation_evaluations=sum(
                t.local_condensation_evaluations for t in timings
            ),
            full_batch_integration_calls=sum(t.full_batch_integration_calls for t in timings),
            equivalent_active_point_integrations=sum(
                t.equivalent_active_point_integrations for t in timings
            ),
            local_iteration_histogram=tuple(histogram),
        )

    @staticmethod
    def _concat_optional(
        values: list[NDArray | None],
    ) -> NDArray | None:
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise RuntimeError("inconsistent optional constitutive block response")
        return np.concatenate([value for value in values if value is not None], axis=0)

    @staticmethod
    def _concat_trials(
        trials: Sequence[InPlaneConstitutiveTrial],
    ) -> InPlaneConstitutiveTrial:
        observable_names = set().union(*(trial.observables for trial in trials))
        observables = {
            name: np.concatenate([trial.observables[name] for trial in trials], axis=0)
            for name in observable_names
        }
        return InPlaneConstitutiveTrial(
            stress_in_plane_mpa=np.concatenate(
                [trial.stress_in_plane_mpa for trial in trials], axis=0
            ),
            tangent_in_plane_mpa=MFront3DCondensedPlaneStressBlockBatch._concat_optional(
                [trial.tangent_in_plane_mpa for trial in trials]
            ),
            observables=observables,
            local_plane_stress_iterations=MFront3DCondensedPlaneStressBlockBatch._concat_optional(
                [trial.local_plane_stress_iterations for trial in trials]
            ),
            cbb_condition_number=MFront3DCondensedPlaneStressBlockBatch._concat_optional(
                [trial.cbb_condition_number for trial in trials]
            ),
        )

    @staticmethod
    def _concat_complete_trials(
        trials: Sequence[ConstitutiveTrial],
    ) -> ConstitutiveTrial:
        light = MFront3DCondensedPlaneStressBlockBatch._concat_trials(trials)
        return ConstitutiveTrial(
            stress_in_plane_mpa=light.stress_in_plane_mpa,
            tangent_in_plane_mpa=light.tangent_in_plane_mpa,
            observables=light.observables,
            local_plane_stress_iterations=light.local_plane_stress_iterations,
            cbb_condition_number=light.cbb_condition_number,
            full_stress_tensor_mpa=np.concatenate(
                [trial.full_stress_tensor_mpa for trial in trials], axis=0
            ),
            full_strain_tensor=np.concatenate(
                [trial.full_strain_tensor for trial in trials], axis=0
            ),
            elastic_strain_tensor=np.concatenate(
                [trial.elastic_strain_tensor for trial in trials], axis=0
            ),
            plastic_strain_tensor=np.concatenate(
                [trial.plastic_strain_tensor for trial in trials], axis=0
            ),
            plane_stress_residual_mpa=np.concatenate(
                [trial.plane_stress_residual_mpa for trial in trials], axis=0
            ),
        )

    def evaluate_in_plane(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> InPlaneConstitutiveTrial:
        values = np.asarray(in_plane_strain, dtype=float)
        if values.shape != (self._point_count, 3):
            raise ValueError(f"in_plane_strain must have shape {(self._point_count, 3)}")
        self._last_light_trials = None
        self._last_light_trial = None
        try:
            trials = [
                block.evaluate_in_plane(
                    values[block_slice],
                    time_increment=time_increment,
                    consistent_tangent=consistent_tangent,
                )
                for block, block_slice in zip(self._blocks, self._slices, strict=True)
            ]
        except Exception:
            self.revert()
            raise
        self._last_light_trials = trials
        self._last_light_trial = self._concat_trials(trials)
        return self._last_light_trial

    def evaluate_in_plane_response(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        response_level: ResponseLevel,
        consistent_tangent: bool = True,
    ) -> InPlaneConstitutiveTrial:
        if response_level == "complete":
            return self.evaluate(
                in_plane_strain,
                time_increment=time_increment,
                consistent_tangent=consistent_tangent,
            )
        return self.evaluate_in_plane(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
        )

    def evaluate(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> ConstitutiveTrial:
        values = np.asarray(in_plane_strain, dtype=float)
        if values.shape != (self._point_count, 3):
            raise ValueError(f"in_plane_strain must have shape {(self._point_count, 3)}")
        try:
            trials = [
                block.evaluate(
                    values[block_slice],
                    time_increment=time_increment,
                    consistent_tangent=consistent_tangent,
                )
                for block, block_slice in zip(self._blocks, self._slices, strict=True)
            ]
        except Exception:
            self.revert()
            raise
        self._last_light_trials = None
        self._last_light_trial = None
        return self._concat_complete_trials(trials)

    def complete_trial(self, trial: InPlaneConstitutiveTrial) -> ConstitutiveTrial:
        if isinstance(trial, ConstitutiveTrial):
            return trial
        if trial is not self._last_light_trial or self._last_light_trials is None:
            raise TypeError("block condensed trial is not the latest light trial")
        try:
            complete = [
                block.complete_trial(block_trial)
                for block, block_trial in zip(
                    self._blocks, self._last_light_trials, strict=True
                )
            ]
        except Exception:
            self.revert()
            raise
        return self._concat_complete_trials(complete)

    def commit(self) -> None:
        for block in self._blocks:
            block.commit()
        self._last_light_trials = None
        self._last_light_trial = None

    def revert(self) -> None:
        for block in self._blocks:
            block.revert()
        self._last_light_trials = None
        self._last_light_trial = None

    def snapshot_state(self) -> MFrontCondensedBlocksStateSnapshot:
        return MFrontCondensedBlocksStateSnapshot(
            blocks=tuple(block.snapshot_state() for block in self._blocks)
        )

    def restore_state(self, snapshot: MFrontCondensedBlocksStateSnapshot) -> None:
        if len(snapshot.blocks) != len(self._blocks):
            raise ValueError("incompatible block-condensed snapshot")
        for block, block_snapshot in zip(self._blocks, snapshot.blocks, strict=True):
            block.restore_state(block_snapshot)
        self._last_light_trials = None
        self._last_light_trial = None

    def set_nonlocal_equivalent_plastic_strain(self, values: ArrayLike) -> None:
        values_array = np.asarray(values, dtype=float)
        if values_array.shape != (self._point_count,):
            raise ValueError(
                f"values must have shape {(self._point_count,)}"
            )
        for block, block_slice in zip(self._blocks, self._slices, strict=True):
            block.set_nonlocal_equivalent_plastic_strain(values_array[block_slice])

    @property
    def committed_nonlocal_equivalent_plastic_strain(self) -> NDArray:
        return np.concatenate(
            [block.committed_nonlocal_equivalent_plastic_strain for block in self._blocks],
            axis=0,
        )
