"""Native 2D MFront plane-stress bridge."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.linear_solver import LinearSystemMatrixType
from fem_inhouse.core.mfront_runtime import (
    MFrontIntegrationError,
    MFrontIntegrationResult,
    MFrontUnavailableError,
    _broadcast_material_properties,
    _broadcast_point_property,
    _load_mgis,
    _load_mgis_root,
    _variable_offset,
    engineering_strain_to_kelvin,
    kelvin_strain_to_engineering,
    kelvin_stress_to_engineering,
    kelvin_tangent_to_engineering,
)
from fem_inhouse.core.mfront_state import MFrontTimingStatistics
from fem_inhouse.core.plane_stress_material import (
    ConstitutiveTrial,
    InPlaneConstitutiveTrial,
    PlaneStressBatchStatistics,
)
from fem_inhouse.core.tensor_reconstruction import (
    FullTensorState,
    reconstruct_native_plane_stress_state,
)

_SYMMETRIC_POSITIVE_DEFINITE_J2_BEHAVIOURS = frozenset(
    {
        "PixelLudwikJ2Plasticity",
        "PixelMicromorphicLudwikJ2Plasticity",
        "PixelLudwikJ2Plasticity3D",
        "PixelMicromorphicLudwikJ2Plasticity3D",
    }
)

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
