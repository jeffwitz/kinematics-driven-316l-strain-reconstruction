"""Generic 3D MFront bridge with explicit global/crystal rotations."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.crystal_orientation import mgis_rotation_argument, validate_rotations
from fem_inhouse.core.linear_solver import LinearSystemMatrixType
from fem_inhouse.core.mfront_behaviours import MFrontBehaviourSpec
from fem_inhouse.core.mfront_runtime import (
    MFrontIntegrationError,
    MFrontUnavailableError,
    _apply_behaviour_parameters,
    _broadcast_material_properties,
    _broadcast_point_property,
    _declared_internal_slices,
    _load_mgis,
    _load_mgis_root,
    _variable_offset,
)
from fem_inhouse.core.mfront_state import MFrontMaterialStateSnapshot, MFrontTimingStatistics

_TRANSVERSE_COMPONENTS_3D = np.array([2, 4, 5])
_SYMMETRIC_POSITIVE_DEFINITE_J2_BEHAVIOURS = frozenset(
    {
        "PixelLudwikJ2Plasticity",
        "PixelMicromorphicLudwikJ2Plasticity",
        "PixelLudwikJ2Plasticity3D",
        "PixelMicromorphicLudwikJ2Plasticity3D",
    }
)

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
        # Micromorphic crystal extensions keep their coupling modulus as a
        # required MFront property so the same compiled law supports local and
        # non-local runs.  A direct local material-point user must still obtain
        # the historical law without having to know that implementation detail.
        if behaviour_spec is not None:
            for variable in behaviour_spec.material_properties:
                if (
                    variable.canonical_name == "coupling_modulus_mpa"
                    and variable.entry_name not in material_values
                ):
                    material_values[variable.entry_name] = np.zeros(resolved_point_count)
        nonlocal_values_s0: NDArray | None = None
        nonlocal_values_s1: NDArray | None = None
        committed_nonlocal_values: NDArray | None = None
        trial_nonlocal_values: NDArray | None = None
        declares_nonlocal_field = bool(
            behaviour_spec is not None
            and any(
                variable.canonical_name == "nonlocal_equivalent_plastic_strain"
                for variable in behaviour_spec.external_state_variables
            )
        )
        if micromorphic_coupling_modulus_mpa is not None or declares_nonlocal_field:
            coupling = (
                _broadcast_point_property(
                    micromorphic_coupling_modulus_mpa,
                    resolved_point_count,
                    name="micromorphic_coupling_modulus_mpa",
                    nonnegative=True,
                )
                if micromorphic_coupling_modulus_mpa is not None
                else np.zeros(resolved_point_count)
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
