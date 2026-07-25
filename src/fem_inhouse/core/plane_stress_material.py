"""Common contract for constitutive batches used by the 2D plane-stress FEM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.constitutive import (
    PLANE_STRESS_VON_MISES_METRIC,
    make_hardening,
    return_mapping,
)
from fem_inhouse.core.constitutive import (
    consistent_tangent as python_consistent_tangent,
)
from fem_inhouse.core.element import plane_stress_elasticity
from fem_inhouse.core.tensor_reconstruction import reconstruct_python_plane_stress_state

FloatArray = NDArray[np.float64]


class ConstitutiveIntegrationError(RuntimeError):
    """A constitutive trial failed and must not be committed."""


class LocalPlaneStressConvergenceError(ConstitutiveIntegrationError):
    """The local elimination of transverse strains did not converge."""


@dataclass(frozen=True, slots=True, kw_only=True)
class InPlaneConstitutiveTrial:
    """Light non-committed response required by the global Newton loop."""

    stress_in_plane_mpa: FloatArray
    tangent_in_plane_mpa: FloatArray | None
    observables: dict[str, FloatArray] = field(default_factory=dict)
    local_plane_stress_iterations: FloatArray | None = None
    cbb_condition_number: FloatArray | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ConstitutiveTrial(InPlaneConstitutiveTrial):
    """In-plane response enriched with the complete three-dimensional state."""

    full_stress_tensor_mpa: FloatArray
    full_strain_tensor: FloatArray
    elastic_strain_tensor: FloatArray
    plastic_strain_tensor: FloatArray
    plane_stress_residual_mpa: FloatArray


@dataclass(frozen=True, slots=True)
class PlaneStressBatchStatistics:
    """Accumulated diagnostics for local plane-stress enforcement."""

    maximum_gauss_point_plane_stress_residual_mpa: float = 0.0
    maximum_local_plane_stress_iterations: int = 0
    mean_local_plane_stress_iterations: float = 0.0
    local_plane_stress_failures: int = 0
    maximum_cbb_condition_number: float = 0.0


@runtime_checkable
class PlaneStressMaterialBatch(Protocol):
    """Transaction-safe material integration contract seen by global Newton."""

    @property
    def point_count(self) -> int: ...

    @property
    def backend_name(self) -> str: ...

    @property
    def completion_strategy(self) -> str: ...

    @property
    def statistics(self) -> PlaneStressBatchStatistics: ...

    def evaluate(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> ConstitutiveTrial: ...

    def evaluate_in_plane(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> InPlaneConstitutiveTrial: ...

    def complete_trial(self, trial: InPlaneConstitutiveTrial) -> ConstitutiveTrial: ...

    def commit(self) -> None: ...

    def revert(self) -> None: ...


class PythonJ2PlaneStressBatch:
    """Transaction-safe adapter for the historical in-house J2 return mapping."""

    def __init__(
        self,
        initial_yield_stress_mpa: ArrayLike,
        hardening_coefficient_mpa: ArrayLike,
        hardening_exponent: float,
        *,
        young_modulus_mpa: float,
        poisson_ratio: float,
        hardening_mode: str = "ludwik",
        plastic_strain_max: float = 0.2,
        plastic_table_points: int = 1_000,
        first_positive_plastic_strain: float = 1e-6,
    ) -> None:
        yield_stress, coefficient = np.broadcast_arrays(
            np.asarray(initial_yield_stress_mpa, dtype=float),
            np.asarray(hardening_coefficient_mpa, dtype=float),
        )
        self._yield_stress = np.ravel(yield_stress).copy()
        self._coefficient = np.ravel(coefficient).copy()
        if self._yield_stress.size == 0:
            raise ValueError("at least one material point is required")
        if not np.isfinite(self._yield_stress).all() or np.any(self._yield_stress <= 0):
            raise ValueError("initial_yield_stress_mpa must be finite and positive")
        if not np.isfinite(self._coefficient).all() or np.any(self._coefficient < 0):
            raise ValueError("hardening_coefficient_mpa must be finite and non-negative")
        self._young = float(young_modulus_mpa)
        self._poisson = float(poisson_ratio)
        self._elasticity = plane_stress_elasticity(self._young, self._poisson)
        metric_product = self._elasticity @ PLANE_STRESS_VON_MISES_METRIC
        self._cm11 = float(metric_product[0, 0])
        self._cm12 = float(metric_product[0, 1])
        self._cm33 = float(metric_product[2, 2])
        self._hardening, self._hardening_derivative = make_hardening(
            hardening_exponent,
            hardening_mode,  # type: ignore[arg-type]
            plastic_strain_max,
            plastic_table_points,
            first_positive_plastic_strain,
        )
        self._plastic_strain = np.zeros((self.point_count, 3), dtype=float)
        self._peeq = np.zeros(self.point_count, dtype=float)
        self._trial_plastic: FloatArray | None = None
        self._trial_peeq: FloatArray | None = None

    @property
    def point_count(self) -> int:
        return self._yield_stress.size

    @property
    def backend_name(self) -> str:
        return "python-j2-plane-stress"

    @property
    def completion_strategy(self) -> str:
        return "j2_isotropic_analytical"

    @property
    def statistics(self) -> PlaneStressBatchStatistics:
        return PlaneStressBatchStatistics()

    def evaluate(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> ConstitutiveTrial:
        if not np.isfinite(time_increment) or time_increment <= 0:
            raise ValueError("time_increment must be finite and positive")
        total = np.asarray(in_plane_strain, dtype=float)
        if total.shape != (self.point_count, 3):
            raise ValueError(f"in_plane_strain must have shape {(self.point_count, 3)}")
        if not np.isfinite(total).all():
            raise ValueError("in_plane_strain must be finite")
        trial_stress = np.einsum(
            "ij,pj->pi",
            self._elasticity,
            total - self._plastic_strain,
        )
        stress, increment, peeq_increment = return_mapping(
            trial_stress,
            self._peeq,
            self._yield_stress,
            self._coefficient,
            self._hardening,
            self._cm11,
            self._cm12,
            self._cm33,
        )
        trial_plastic = self._plastic_strain + increment
        trial_peeq = self._peeq + peeq_increment
        tangent: FloatArray | None = None
        if consistent_tangent:
            tangent = np.broadcast_to(
                self._elasticity,
                (self.point_count, 3, 3),
            ).copy()
            plastic = peeq_increment > 0
            if np.any(plastic):
                tangent[plastic] = python_consistent_tangent(
                    stress[plastic],
                    peeq_increment[plastic],
                    self._peeq[plastic],
                    self._yield_stress[plastic],
                    self._coefficient[plastic],
                    self._hardening,
                    self._hardening_derivative,
                    self._elasticity,
                    self._cm11,
                    self._cm12,
                    self._cm33,
                )
        full = reconstruct_python_plane_stress_state(
            total,
            trial_plastic,
            stress,
            self._poisson,
        )
        self._trial_plastic = trial_plastic
        self._trial_peeq = trial_peeq
        return ConstitutiveTrial(
            stress_in_plane_mpa=stress,
            tangent_in_plane_mpa=tangent,
            full_stress_tensor_mpa=full.stress_tensor_mpa,
            full_strain_tensor=full.total_strain_tensor,
            elastic_strain_tensor=full.elastic_strain_tensor,
            plastic_strain_tensor=full.plastic_strain_tensor,
            plane_stress_residual_mpa=full.plane_stress_residual_vector_mpa,
            observables={
                "plastic_strain_2d": trial_plastic,
                "equivalent_plastic_strain": trial_peeq,
                "yield_surface_radius_mpa": (
                    self._yield_stress + self._coefficient * self._hardening(trial_peeq)
                ),
            },
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
    ) -> FloatArray:
        return self.evaluate(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=False,
        ).observables["equivalent_plastic_strain"]

    def complete_trial(self, trial: InPlaneConstitutiveTrial) -> ConstitutiveTrial:
        if not isinstance(trial, ConstitutiveTrial):
            raise TypeError("Python J2 in-plane trial is missing its reconstructed state")
        return trial

    def commit(self) -> None:
        if self._trial_plastic is None or self._trial_peeq is None:
            raise RuntimeError("no successful Python J2 trial state to commit")
        self._plastic_strain = self._trial_plastic
        self._peeq = self._trial_peeq
        self._trial_plastic = None
        self._trial_peeq = None

    def revert(self) -> None:
        self._trial_plastic = None
        self._trial_peeq = None


def create_plane_stress_material_batch(
    backend: str,
    initial_yield_stress_mpa: ArrayLike,
    hardening_coefficient_mpa: ArrayLike,
    hardening_exponent: float,
    *,
    young_modulus_mpa: float,
    poisson_ratio: float,
    hardening_mode: str,
    plastic_strain_max: float,
    plastic_table_points: int,
    first_positive_plastic_strain: float,
    mfront_library: str,
    mfront_threads: int,
    local_plane_stress_options: dict[str, Any] | None = None,
    nonlocal_coupling_modulus_mpa: float | None = None,
) -> PlaneStressMaterialBatch:
    """Construct a backend without exposing its implementation to global Newton."""

    if backend == "python":
        return PythonJ2PlaneStressBatch(
            initial_yield_stress_mpa,
            hardening_coefficient_mpa,
            hardening_exponent,
            young_modulus_mpa=young_modulus_mpa,
            poisson_ratio=poisson_ratio,
            hardening_mode=hardening_mode,
            plastic_strain_max=plastic_strain_max,
            plastic_table_points=plastic_table_points,
            first_positive_plastic_strain=first_positive_plastic_strain,
        )
    if backend in {
        "mfront",
        "mfront-native-plane-stress",
        "mfront-3d-condensed-plane-stress",
    }:
        if not np.isclose(young_modulus_mpa, 205_000.0) or not np.isclose(poisson_ratio, 0.3):
            raise ValueError("the compiled MFront behaviours require E=205000 MPa and nu=0.3")
        if not np.isclose(first_positive_plastic_strain, 1e-6):
            raise ValueError(
                "the compiled MFront behaviours require first_positive_plastic_strain=1e-6"
            )
        from fem_inhouse.core.mfront import (
            MFront3DCondensedPlaneStressBatch,
            MFrontNativePlaneStressBatch,
        )

        common = (
            mfront_library,
            initial_yield_stress_mpa,
            hardening_coefficient_mpa,
            np.full(np.asarray(initial_yield_stress_mpa).size, hardening_exponent),
        )
        micromorphic_options: dict[str, Any] = {}
        if nonlocal_coupling_modulus_mpa is not None:
            micromorphic_options = {
                "micromorphic_coupling_modulus_mpa": nonlocal_coupling_modulus_mpa,
            }
        if backend in {"mfront", "mfront-native-plane-stress"}:
            return MFrontNativePlaneStressBatch(
                *common,
                thread_count=mfront_threads,
                behaviour_name=(
                    "PixelMicromorphicLudwikJ2Plasticity"
                    if nonlocal_coupling_modulus_mpa is not None
                    else "PixelLudwikJ2Plasticity"
                ),
                **micromorphic_options,
            )
        return MFront3DCondensedPlaneStressBatch(
            *common,
            thread_count=mfront_threads,
            behaviour_name=(
                "PixelMicromorphicLudwikJ2Plasticity3D"
                if nonlocal_coupling_modulus_mpa is not None
                else "PixelLudwikJ2Plasticity3D"
            ),
            **micromorphic_options,
            **(local_plane_stress_options or {}),
        )
    raise ValueError(
        "constitutive_backend must be 'python', 'mfront-native-plane-stress', "
        "or 'mfront-3d-condensed-plane-stress'"
    )
