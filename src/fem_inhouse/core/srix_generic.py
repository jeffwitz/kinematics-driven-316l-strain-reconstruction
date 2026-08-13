"""Raw MGIS bridge for the validation SRIX GenericBehaviour formulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.linear_solver import LinearSystemMatrixType
from fem_inhouse.core.mfront_runtime import (
    MFrontIntegrationError,
    _broadcast_point_property,
    _load_mgis,
    _variable_offset,
)

FloatArray = NDArray[np.float64]


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

    This is deliberately a raw 3-D constitutive bridge. Plane-stress closure,
    rotations and global non-local coupling remain outside this first adapter;
    keeping those layers separate makes the GenericBehaviour contract
    independently testable against the historical SRIX law.
    """

    def __init__(
        self,
        library_path: str | Path,
        *,
        point_count: int,
        micromorphic_coupling_modulus_mpa: ArrayLike,
        temperature_k: float = 293.15,
        behaviour_name: str = "Fcc316LForestRubinSrixGeneric3D",
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
        gradients[:, :6] = strain
        gradients[:, 6] = chi_values
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
        if tangent.shape[-1] != 49:
            raise ValueError(f"expected 49 Generic tangent entries, got {tangent.shape}")
        blocks = tangent.reshape(self._point_count, 7, 7)
        return SrixGeneric3DTrial(
            total_strain_kelvin=strain.copy(),
            chi=chi_values.copy(),
            stress_kelvin_mpa=forces[:, :6].copy(),
            accumulated_slip=forces[:, 6].copy(),
            stress_strain_tangent=blocks[:, :6, :6].copy(),
            stress_chi_tangent=blocks[:, :6, 6:7].copy(),
            accumulated_slip_strain_tangent=blocks[:, 6:7, :6].copy(),
            accumulated_slip_chi_tangent=blocks[:, 6:7, 6:7].copy(),
        )

    def commit(self) -> None:
        if not self._has_trial_state:
            raise RuntimeError("no successful Generic 3-D trial state to commit")
        self._mgis.update(self._manager)
        self._has_trial_state = False

    def revert(self) -> None:
        self._mgis.revert(self._manager)
        self._has_trial_state = False
