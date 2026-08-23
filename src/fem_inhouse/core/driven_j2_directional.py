"""Prototype driven-J2 material with a signed, prescribed flow-direction basis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from fem_inhouse.core.constitutive import PLANE_STRESS_VON_MISES_METRIC, von_mises
from fem_inhouse.core.driven_j2 import DrivenJ2Trial
from fem_inhouse.core.element import plane_stress_elasticity
from fem_inhouse.core.plane_stress_material import (
    ConstitutiveIntegrationError,
    PlaneStressBatchStatistics,
)
from fem_inhouse.core.tensor_reconstruction import reconstruct_python_plane_stress_state


@dataclass(frozen=True, slots=True)
class DirectionalDrivenJ2Trial(DrivenJ2Trial):
    """Marker subtype for the directional prototype."""


class DirectionalDrivenJ2PlaneStressBatch:
    """Transactional driven-J2 with fixed ``Delta p`` and signed direction modes.

    This is intentionally a validation backend: local tangents are obtained by
    controlled finite differences. It is not yet the production constitutive
    backend.
    """

    def __init__(
        self,
        point_count: int,
        direction_modes: ArrayLike,
        *,
        young_modulus_mpa: float,
        poisson_ratio: float,
    ) -> None:
        self._point_count = point_count
        self._elasticity = np.asarray(plane_stress_elasticity(young_modulus_mpa, poisson_ratio))
        modes = np.asarray(direction_modes, dtype=np.float64)
        if modes.ndim != 3 or modes.shape[0] != point_count or modes.shape[1] != 3:
            raise ValueError("direction_modes must have shape (point_count, 3, rank)")
        self._modes = modes
        self._coefficients = np.zeros(modes.shape[2], dtype=np.float64)
        self._committed_plastic_strain = np.zeros((point_count, 3), dtype=np.float64)
        self._committed_peeq = np.zeros(point_count, dtype=np.float64)
        self._trial_plastic_strain = None
        self._trial_peeq = None
        self._last_maximum_residual = 0.0
        self._last_maximum_iterations = 0

    @property
    def point_count(self) -> int:
        return self._point_count

    @property
    def statistics(self) -> PlaneStressBatchStatistics:
        return PlaneStressBatchStatistics(
            maximum_gauss_point_plane_stress_residual_mpa=self._last_maximum_residual,
            maximum_local_plane_stress_iterations=self._last_maximum_iterations,
        )

    def set_direction_coefficients(self, coefficients: ArrayLike) -> None:
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != self._coefficients.shape:
            raise ValueError("direction coefficient shape mismatch")
        self._coefficients = values.copy()

    def set_committed_state(
        self, plastic_strain: ArrayLike, equivalent_plastic_strain: ArrayLike
    ) -> None:
        """Load a committed b=0 state produced by the analytical Driven-J2 backend."""
        plastic = np.asarray(plastic_strain, dtype=np.float64)
        peeq = np.asarray(equivalent_plastic_strain, dtype=np.float64)
        if plastic.shape != (self._point_count, 3) or peeq.shape != (self._point_count,):
            raise ValueError("incompatible committed directional state")
        self._committed_plastic_strain = plastic.copy()
        self._committed_peeq = peeq.copy()
        self.revert()

    def _flow(self, stress: np.ndarray, point: int) -> tuple[np.ndarray, float]:
        q = float(von_mises(stress[None])[0])
        if q <= 1.0e-12:
            raise ConstitutiveIntegrationError("directional J2 requires non-zero stress")
        n = PLANE_STRESS_VON_MISES_METRIC @ stress / q
        raw = self._modes[point] @ self._coefficients
        projected = raw - n * (stress @ raw / q)
        vector = n + projected
        inverse_metric = np.linalg.inv(PLANE_STRESS_VON_MISES_METRIC)
        norm = float(np.sqrt(vector @ inverse_metric @ vector))
        if not np.isfinite(norm) or norm <= 1.0e-14:
            raise ConstitutiveIntegrationError("directional J2 normalization failed")
        return vector / norm, q

    def _residual(
        self, stress: np.ndarray, trial_stress: np.ndarray, increment: float, point: int
    ) -> np.ndarray:
        flow, _ = self._flow(stress, point)
        return stress - trial_stress + increment * (self._elasticity @ flow)

    def _solve_local(
        self,
        trial_stress: np.ndarray,
        increment: float,
        point: int,
        initial: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if increment == 0.0:
            return trial_stress.copy(), np.zeros(3), 0
        stress = trial_stress.copy() if initial is None else initial.copy()
        for iteration in range(1, 61):
            residual = self._residual(stress, trial_stress, increment, point)
            if np.linalg.norm(residual) <= 1.0e-8 + 1.0e-10 * max(np.linalg.norm(stress), 1.0):
                flow, _ = self._flow(stress, point)
                return stress, flow, iteration
            jacobian = np.empty((3, 3))
            step = 1.0e-7 * max(np.linalg.norm(stress), 1.0)
            for component in range(3):
                perturbation = np.zeros(3)
                perturbation[component] = step
                jacobian[:, component] = (
                    self._residual(stress + perturbation, trial_stress, increment, point)
                    - self._residual(stress - perturbation, trial_stress, increment, point)
                ) / (2.0 * step)
            correction = np.linalg.solve(jacobian, -residual)
            accepted = False
            base_norm = np.linalg.norm(residual)
            for reduction in range(20):
                candidate = stress + (0.5**reduction) * correction
                if (
                    np.linalg.norm(self._residual(candidate, trial_stress, increment, point))
                    < base_norm
                ):
                    stress = candidate
                    accepted = True
                    break
            if not accepted:
                raise ConstitutiveIntegrationError("directional J2 local line search failed")
        raise ConstitutiveIntegrationError("directional J2 local Newton did not converge")

    def evaluate(
        self,
        in_plane_strain: ArrayLike,
        equivalent_plastic_increment: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> DirectionalDrivenJ2Trial:
        del time_increment
        strain = np.asarray(in_plane_strain, dtype=np.float64)
        increment = np.broadcast_to(
            np.asarray(equivalent_plastic_increment), (self._point_count,)
        ).astype(float)
        trial_stress = np.einsum(
            "ij,pj->pi", self._elasticity, strain - self._committed_plastic_strain
        )
        stress = np.empty_like(trial_stress)
        flow = np.empty_like(stress)
        tangents = np.empty((self._point_count, 3, 3))
        for point in range(self._point_count):
            stress[point], flow[point], _ = self._solve_local(
                trial_stress[point], float(increment[point]), point
            )
            for component in range(3):
                h = 1.0e-7
                perturbation = self._elasticity[:, component] * h
                plus = self._solve_local(
                    trial_stress[point] + perturbation,
                    float(increment[point]),
                    point,
                    stress[point],
                )[0]
                minus = self._solve_local(
                    trial_stress[point] - perturbation,
                    float(increment[point]),
                    point,
                    stress[point],
                )[0]
                tangents[point, :, component] = (plus - minus) / (2.0 * h)
        plastic = self._committed_plastic_strain + increment[:, None] * flow
        full = reconstruct_python_plane_stress_state(strain, plastic, stress, 0.30)
        self._trial_plastic_strain = plastic
        trial_peeq = self._committed_peeq + increment
        self._trial_peeq = trial_peeq
        return DirectionalDrivenJ2Trial(
            stress_in_plane_mpa=stress,
            tangent_in_plane_mpa=tangents if consistent_tangent else None,
            stress_equivalent_plastic_increment_tangent_mpa=np.zeros_like(stress),
            local_residual_norm_mpa=np.zeros(self._point_count),
            local_iterations=np.zeros(self._point_count),
            full_stress_tensor_mpa=full.stress_tensor_mpa,
            full_strain_tensor=full.total_strain_tensor,
            elastic_strain_tensor=full.elastic_strain_tensor,
            plastic_strain_tensor=full.plastic_strain_tensor,
            plane_stress_residual_mpa=full.plane_stress_residual_vector_mpa,
            observables={
                "plastic_strain_2d": plastic,
                "equivalent_plastic_strain": trial_peeq,
                "equivalent_plastic_increment": increment,
                "flow_direction": flow,
            },
        )

    def commit(self) -> None:
        if self._trial_plastic_strain is None or self._trial_peeq is None:
            raise RuntimeError("no successful directional trial state to commit")
        self._committed_plastic_strain = self._trial_plastic_strain.copy()
        self._committed_peeq = self._trial_peeq.copy()
        self.revert()

    def revert(self) -> None:
        self._trial_plastic_strain = None
        self._trial_peeq = None
