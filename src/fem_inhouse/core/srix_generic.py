"""Raw MGIS bridge for the validation SRIX GenericBehaviour formulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.crystal_orientation import mgis_rotation_argument, validate_rotations
from fem_inhouse.core.linear_solver import LinearSystemMatrixType
from fem_inhouse.core.mfront_condensation import condense_kelvin_tangent_blocks
from fem_inhouse.core.mfront_runtime import (
    _ENGINEERING_TO_KELVIN_STRAIN_SCALE,
    _KELVIN_TO_ENGINEERING_STRESS_SCALE,
    MFrontIntegrationError,
    _broadcast_point_property,
    _load_mgis,
    _variable_offset,
)

FloatArray = NDArray[np.float64]
_PLANE = np.array([0, 1, 3])
_TRANSVERSE = np.array([2, 4, 5])


@dataclass(frozen=True, slots=True)
class SrixGeneric3DTrial:
    """Non-committed response of the seven-field GenericBehaviour."""

    total_strain_kelvin: FloatArray
    chi: FloatArray
    stress_kelvin_mpa: FloatArray
    accumulated_slip: FloatArray
    stress_strain_tangent: FloatArray
    stress_chi_tangent: FloatArray
    accumulated_slip_strain_tangent: FloatArray
    accumulated_slip_chi_tangent: FloatArray


class SrixGeneric3DMaterialPointBatch:
    """Transactional 3-D bridge for the tangent-enabled SRIX GenericBehaviour.

    This is deliberately a raw 3-D constitutive bridge. Plane-stress closure
    and global non-local coupling remain outside this first adapter; keeping
    those layers separate makes the GenericBehaviour contract independently
    testable against the historical SRIX law.
    """

    def __init__(
        self,
        library_path: str | Path,
        *,
        point_count: int,
        micromorphic_coupling_modulus_mpa: ArrayLike,
        temperature_k: float = 293.15,
        behaviour_name: str = "Fcc316LForestRubinSrixGeneric3D",
        rotation_global_to_material: ArrayLike | None = None,
    ) -> None:
        if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count < 1:
            raise ValueError("point_count must be a positive integer")
        if not np.isfinite(temperature_k) or temperature_k <= 0:
            raise ValueError("temperature_k must be finite and positive")
        library = Path(library_path)
        if not library.is_file():
            raise FileNotFoundError(f"MFront behaviour library not found: {library}")

        mgis = _load_mgis()
        behaviour = mgis.load(
            str(library.resolve()), behaviour_name, mgis.Hypothesis.Tridimensional
        )
        _variable_offset(
            mgis,
            behaviour.gradients,
            "Strain",
            mgis.Hypothesis.Tridimensional,
            expected_size=6,
        )
        _variable_offset(
            mgis,
            behaviour.gradients,
            "NonlocalEquivalentPlasticStrain",
            mgis.Hypothesis.Tridimensional,
            expected_size=1,
        )
        _variable_offset(
            mgis,
            behaviour.thermodynamic_forces,
            "Stress",
            mgis.Hypothesis.Tridimensional,
            expected_size=6,
        )
        _variable_offset(
            mgis,
            behaviour.thermodynamic_forces,
            "AccumulatedSlipOutput",
            mgis.Hypothesis.Tridimensional,
            expected_size=1,
        )
        coupling = _broadcast_point_property(
            micromorphic_coupling_modulus_mpa,
            point_count,
            name="micromorphic_coupling_modulus_mpa",
            nonnegative=True,
        )
        manager = mgis.MaterialDataManager(behaviour, point_count)
        storage_mode = mgis.MaterialStateManagerStorageMode.ExternalStorage
        for state in (manager.s0, manager.s1):
            mgis.setMaterialProperty(
                state, "MicromorphicCouplingModulus", coupling, storage_mode
            )
            mgis.setExternalStateVariable(
                state,
                "Temperature",
                np.full(point_count, temperature_k),
                storage_mode,
            )
        self._mgis = mgis
        self._behaviour = behaviour
        self._manager = manager
        self._point_count = point_count
        self._has_trial_state = False
        self._rotations = (
            None
            if rotation_global_to_material is None
            else validate_rotations(rotation_global_to_material, point_count=point_count)
        )
        self._mgis_rotations = (
            None if self._rotations is None else mgis_rotation_argument(self._rotations)
        )

    @property
    def point_count(self) -> int:
        return self._point_count

    @property
    def backend_name(self) -> str:
        return "srix-generic-3d"

    @property
    def completion_strategy(self) -> str:
        return "generic_3d_raw"

    @property
    def linear_system_matrix_type(self) -> LinearSystemMatrixType:
        return "nonsymmetric"

    @property
    def rotations_global_to_material(self) -> FloatArray | None:
        return None if self._rotations is None else self._rotations.copy()

    def evaluate(
        self,
        total_strain_kelvin: ArrayLike,
        chi: ArrayLike,
        *,
        time_increment: float,
    ) -> SrixGeneric3DTrial:
        strain = np.asarray(total_strain_kelvin, dtype=float)
        chi_values = _broadcast_point_property(chi, self._point_count, name="chi")
        if strain.shape != (self._point_count, 6):
            raise ValueError(f"total_strain_kelvin must have shape {(self._point_count, 6)}")
        if not np.isfinite(strain).all():
            raise ValueError("total_strain_kelvin must be finite")
        if not np.isfinite(time_increment) or time_increment <= 0:
            raise ValueError("time_increment must be finite and positive")
        if self._has_trial_state:
            self._mgis.revert(self._manager)
            self._has_trial_state = False
        gradients = self._manager.s1.gradients
        global_gradients = np.column_stack((strain, chi_values))
        if self._mgis_rotations is None:
            gradients[:, :] = global_gradients
        else:
            material_gradients = np.ascontiguousarray(global_gradients.reshape(-1).copy())
            self._mgis.rotateGradients(
                material_gradients, self._behaviour, self._mgis_rotations
            )
            gradients[:, :] = material_gradients.reshape(self._point_count, 7)
        status = self._mgis.integrate(
            self._manager,
            self._mgis.IntegrationType.IntegrationWithConsistentTangentOperator,
            float(time_increment),
            0,
            self._point_count,
        )
        if status != 1:
            self.revert()
            raise MFrontIntegrationError(
                f"SRIX Generic 3-D integration failed with status {status}"
            )
        self._has_trial_state = True
        forces = np.asarray(self._manager.s1.thermodynamic_forces, dtype=float).copy()
        tangent = np.asarray(self._manager.K, dtype=float).copy()
        if self._mgis_rotations is not None:
            flat_forces = np.ascontiguousarray(forces.reshape(-1))
            self._mgis.rotateThermodynamicForces(
                flat_forces, self._behaviour, self._mgis_rotations
            )
            forces = flat_forces.reshape(self._point_count, 7)
            flat_tangent = np.ascontiguousarray(tangent.reshape(-1))
            self._mgis.rotateTangentOperatorBlocks(
                flat_tangent, self._behaviour, self._mgis_rotations
            )
            tangent = flat_tangent.reshape(self._point_count, 49)
        if tangent.shape[-1] != 49:
            raise ValueError(f"expected 49 Generic tangent entries, got {tangent.shape}")
        stress_strain = tangent[:, :36].reshape(self._point_count, 6, 6)
        stress_chi = tangent[:, 36:42].reshape(self._point_count, 6, 1)
        accumulated_slip_strain = tangent[:, 42:48].reshape(self._point_count, 1, 6)
        accumulated_slip_chi = tangent[:, 48:].reshape(self._point_count, 1, 1)
        return SrixGeneric3DTrial(
            total_strain_kelvin=strain.copy(),
            chi=chi_values.copy(),
            stress_kelvin_mpa=forces[:, :6].copy(),
            accumulated_slip=forces[:, 6].copy(),
            stress_strain_tangent=stress_strain.copy(),
            stress_chi_tangent=stress_chi.copy(),
            accumulated_slip_strain_tangent=accumulated_slip_strain.copy(),
            accumulated_slip_chi_tangent=accumulated_slip_chi.copy(),
        )

    def commit(self) -> None:
        if not self._has_trial_state:
            raise RuntimeError("no successful Generic 3-D trial state to commit")
        self._mgis.update(self._manager)
        self._has_trial_state = False

    def revert(self) -> None:
        self._mgis.revert(self._manager)
        self._has_trial_state = False


@dataclass(frozen=True, slots=True)
class SrixGenericPlaneStressTrial:
    """Local plane-stress response of the Generic SRIX bridge."""

    stress_in_plane_mpa: FloatArray
    accumulated_slip: FloatArray
    tangent_in_plane_mpa: FloatArray
    transverse_strain_kelvin: FloatArray
    transverse_stress_mpa: FloatArray


class SrixGeneric3DCondensedPlaneStressBatch:
    """Plane-stress closure layered on ``SrixGeneric3DMaterialPointBatch``."""

    def __init__(
        self,
        bridge: SrixGeneric3DMaterialPointBatch,
        *,
        local_tolerance_mpa: float = 1e-8,
        maximum_local_iterations: int = 15,
    ) -> None:
        if local_tolerance_mpa <= 0 or not np.isfinite(local_tolerance_mpa):
            raise ValueError("local_tolerance_mpa must be finite and positive")
        if maximum_local_iterations < 1:
            raise ValueError("maximum_local_iterations must be positive")
        self._bridge = bridge
        self._tolerance = float(local_tolerance_mpa)
        self._maximum_iterations = int(maximum_local_iterations)
        self._committed_transverse = np.zeros((bridge.point_count, 3))
        self._trial_transverse: FloatArray | None = None
        self._has_trial = False

    @property
    def point_count(self) -> int:
        return self._bridge.point_count

    @property
    def backend_name(self) -> str:
        return "srix-generic-3d-condensed-plane-stress"

    def evaluate(
        self,
        in_plane_strain: ArrayLike,
        chi: ArrayLike,
        *,
        time_increment: float,
    ) -> SrixGenericPlaneStressTrial:
        in_plane = np.asarray(in_plane_strain, dtype=float)
        if in_plane.shape != (self.point_count, 3):
            raise ValueError(f"in_plane_strain must have shape {(self.point_count, 3)}")
        total = np.zeros((self.point_count, 6))
        total[:, _PLANE] = in_plane * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
        total[:, _TRANSVERSE] = self._committed_transverse
        final: SrixGeneric3DTrial | None = None
        for _ in range(self._maximum_iterations):
            final = self._bridge.evaluate(total, chi, time_increment=time_increment)
            residual = final.stress_kelvin_mpa[:, _TRANSVERSE]
            if np.max(np.abs(residual)) <= self._tolerance:
                break
            cbb = np.take(
                np.take(final.stress_strain_tangent, _TRANSVERSE, axis=-2),
                _TRANSVERSE,
                axis=-1,
            )
            correction = np.linalg.solve(cbb, -residual[..., None])[..., 0]
            total[:, _TRANSVERSE] += correction
        else:
            self.revert()
            raise RuntimeError("SRIX Generic plane-stress closure did not converge")
        assert final is not None
        c_ps, _, _, _, _ = condense_kelvin_tangent_blocks(
            final.stress_strain_tangent,
            final.stress_chi_tangent,
            final.accumulated_slip_strain_tangent,
            final.accumulated_slip_chi_tangent,
            check_condition=False,
        )
        tangent = (
            c_ps
            * _KELVIN_TO_ENGINEERING_STRESS_SCALE[None, :, None]
            * _ENGINEERING_TO_KELVIN_STRAIN_SCALE[None, None, :]
        )
        self._trial_transverse = total[:, _TRANSVERSE].copy()
        self._has_trial = True
        return SrixGenericPlaneStressTrial(
            stress_in_plane_mpa=(
                final.stress_kelvin_mpa[:, _PLANE]
                * _KELVIN_TO_ENGINEERING_STRESS_SCALE
            ),
            accumulated_slip=final.accumulated_slip.copy(),
            tangent_in_plane_mpa=tangent,
            transverse_strain_kelvin=self._trial_transverse.copy(),
            transverse_stress_mpa=residual.copy(),
        )

    def commit(self) -> None:
        if not self._has_trial or self._trial_transverse is None:
            raise RuntimeError("no successful Generic plane-stress trial to commit")
        self._bridge.commit()
        self._committed_transverse[:] = self._trial_transverse
        self._trial_transverse = None
        self._has_trial = False

    def revert(self) -> None:
        self._bridge.revert()
        self._trial_transverse = None
        self._has_trial = False
