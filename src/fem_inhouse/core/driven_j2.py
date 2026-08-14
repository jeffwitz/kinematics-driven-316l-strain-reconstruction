"""Associated plane-stress J2 response driven by a prescribed ``Delta p``.

This material is the local constitutive kernel of the experimental mechanical
oracle.  It deliberately contains no yield law: an outer optimiser prescribes
the non-negative equivalent plastic increment, while this module enforces the
J2 flow direction, plane-stress elasticity and transaction semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.constitutive import PLANE_STRESS_VON_MISES_METRIC, von_mises
from fem_inhouse.core.element import plane_stress_elasticity
from fem_inhouse.core.implicit_sensitivities import solve_implicit_sensitivities
from fem_inhouse.core.plane_stress_material import (
    ConstitutiveIntegrationError,
    ConstitutiveTrial,
    PlaneStressBatchStatistics,
)
from fem_inhouse.core.tensor_reconstruction import reconstruct_python_plane_stress_state

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True)
class DrivenJ2Trial(ConstitutiveTrial):
    """Non-committed driven-J2 response and both optimisation tangents."""

    stress_equivalent_plastic_increment_tangent_mpa: FloatArray
    local_residual_norm_mpa: FloatArray
    local_iterations: FloatArray

    def stress_direction_action(self, raw_direction: ArrayLike) -> FloatArray:
        """Return ``d sigma / d c`` for a tangent flow-direction perturbation.

        The raw perturbation is projected with ``sigma.T delta_n = 0`` so it
        changes the flow direction without changing the J2 plastic work at
        first order.
        """
        raw = np.asarray(raw_direction, dtype=np.float64)
        expected = self.stress_in_plane_mpa.shape
        if raw.shape != expected:
            raise ValueError(f"raw_direction must have shape {expected}")
        tangent_direction = self.project_direction(raw)
        return -self.observables["equivalent_plastic_increment"][:, None] * np.einsum(
            "pij,pj->pi", self.tangent_in_plane_mpa, tangent_direction
        )

    def project_direction(self, raw_direction: ArrayLike) -> FloatArray:
        """Project a raw direction field onto ``sigma.T delta_n = 0``."""
        raw = np.asarray(raw_direction, dtype=np.float64)
        expected = self.stress_in_plane_mpa.shape
        if raw.shape != expected:
            raise ValueError(f"raw_direction must have shape {expected}")
        stress = self.stress_in_plane_mpa
        q = von_mises(stress)
        direction = np.asarray(self.observables["flow_direction"], dtype=np.float64)
        safe = np.where(q > 0.0, q, 1.0)
        projected = raw - direction * (
            np.einsum("pi,pi->p", stress, raw) / safe
        )[:, None]
        projected[q <= 0.0] = 0.0
        return projected


@runtime_checkable
class DrivenJ2MaterialProtocol(Protocol):
    """Transactional material contract with ``Delta p`` as an explicit input."""

    @property
    def point_count(self) -> int: ...

    def evaluate(
        self,
        in_plane_strain: ArrayLike,
        equivalent_plastic_increment: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> DrivenJ2Trial: ...

    def commit(self) -> None: ...

    def revert(self) -> None: ...


def _flow_geometry(
    stress: FloatArray,
    *,
    equivalent_stress_floor_mpa: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``q``, associated direction ``n`` and ``dn/dsigma``."""

    metric_stress = np.einsum("ij,pj->pi", PLANE_STRESS_VON_MISES_METRIC, stress)
    equivalent = von_mises(stress)
    regular = equivalent > equivalent_stress_floor_mpa
    safe = np.where(regular, equivalent, 1.0)
    direction = metric_stress / safe[:, None]
    direction[~regular] = 0.0
    hessian = (
        PLANE_STRESS_VON_MISES_METRIC[None, :, :] / safe[:, None, None]
        - np.einsum("pi,pj->pij", direction, direction) / safe[:, None, None]
    )
    hessian[~regular] = 0.0
    return equivalent, direction, hessian


class DrivenJ2PlaneStressBatch:
    """Vectorised, transaction-safe associated J2 with prescribed ``Delta p``.

    The local equation is

    ``sigma - C:(epsilon-epsilon_p_n-Delta_p*n(sigma)) = 0``.

    The committed state is never mutated by :meth:`evaluate`.  Repeated trials
    therefore always start from the last accepted global increment.
    """

    def __init__(
        self,
        point_count: int,
        *,
        young_modulus_mpa: float,
        poisson_ratio: float,
        local_absolute_tolerance_mpa: float = 1.0e-10,
        local_relative_tolerance: float = 1.0e-12,
        maximum_local_iterations: int = 50,
        maximum_line_search_iterations: int = 16,
        equivalent_stress_floor_mpa: float = 1.0e-12,
    ) -> None:
        if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count < 1:
            raise ValueError("point_count must be a positive integer")
        if not np.isfinite(local_absolute_tolerance_mpa) or local_absolute_tolerance_mpa <= 0:
            raise ValueError("local_absolute_tolerance_mpa must be finite and positive")
        if not np.isfinite(local_relative_tolerance) or local_relative_tolerance <= 0:
            raise ValueError("local_relative_tolerance must be finite and positive")
        if maximum_local_iterations < 1 or maximum_line_search_iterations < 1:
            raise ValueError("local iteration limits must be positive")
        if not np.isfinite(equivalent_stress_floor_mpa) or equivalent_stress_floor_mpa <= 0:
            raise ValueError("equivalent_stress_floor_mpa must be finite and positive")

        self._point_count = point_count
        self._young = float(young_modulus_mpa)
        self._poisson = float(poisson_ratio)
        self._elasticity = np.asarray(
            plane_stress_elasticity(self._young, self._poisson), dtype=np.float64
        )
        self._absolute_tolerance = float(local_absolute_tolerance_mpa)
        self._relative_tolerance = float(local_relative_tolerance)
        self._maximum_iterations = int(maximum_local_iterations)
        self._maximum_line_search_iterations = int(maximum_line_search_iterations)
        self._stress_floor = float(equivalent_stress_floor_mpa)
        self._committed_plastic_strain = np.zeros((point_count, 3), dtype=np.float64)
        self._committed_peeq = np.zeros(point_count, dtype=np.float64)
        self._trial_plastic_strain: FloatArray | None = None
        self._trial_peeq: FloatArray | None = None
        self._last_maximum_iterations = 0
        self._last_maximum_residual = 0.0

    @property
    def point_count(self) -> int:
        return self._point_count

    @property
    def backend_name(self) -> str:
        return "driven-j2-plane-stress"

    @property
    def completion_strategy(self) -> str:
        return "prescribed-equivalent-plastic-increment-associated-j2"

    @property
    def statistics(self) -> PlaneStressBatchStatistics:
        return PlaneStressBatchStatistics(
            maximum_gauss_point_plane_stress_residual_mpa=self._last_maximum_residual,
            maximum_local_plane_stress_iterations=self._last_maximum_iterations,
        )

    @property
    def committed_plastic_strain(self) -> FloatArray:
        return self._committed_plastic_strain.copy()

    @property
    def committed_equivalent_plastic_strain(self) -> FloatArray:
        return self._committed_peeq.copy()

    @property
    def elastic_tangent_in_plane_mpa(self) -> FloatArray:
        return np.broadcast_to(self._elasticity, (self.point_count, 3, 3)).copy()

    def _residual_and_jacobian(
        self,
        stress: FloatArray,
        trial_stress: FloatArray,
        increment: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        _, direction, flow_hessian = _flow_geometry(
            stress,
            equivalent_stress_floor_mpa=self._stress_floor,
        )
        elastic_flow = np.einsum("ij,pj->pi", self._elasticity, direction)
        residual = stress - trial_stress + increment[:, None] * elastic_flow
        elastic_hessian = np.einsum("ij,pjk->pik", self._elasticity, flow_hessian)
        jacobian = np.eye(3)[None, :, :] + increment[:, None, None] * elastic_hessian
        return residual, jacobian, direction

    def _solve_stress(
        self,
        trial_stress: FloatArray,
        increment: FloatArray,
        initial_stress: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        stress = trial_stress.copy() if initial_stress is None else initial_stress.copy()
        trial_equivalent = von_mises(trial_stress)
        impossible = (increment > 0.0) & (trial_equivalent <= self._stress_floor)
        if np.any(impossible):
            first = int(np.flatnonzero(impossible)[0])
            raise ConstitutiveIntegrationError(
                "positive Delta p requires a non-zero trial J2 direction; "
                f"point {first} has q_trial={trial_equivalent[first]:.3e} MPa"
            )

        scale = np.maximum(np.linalg.norm(trial_stress, axis=1), 1.0)
        tolerance = self._absolute_tolerance + self._relative_tolerance * scale
        iterations = np.zeros(self.point_count, dtype=np.int64)
        converged = increment == 0.0

        for iteration in range(1, self._maximum_iterations + 1):
            residual, jacobian, _ = self._residual_and_jacobian(stress, trial_stress, increment)
            residual_norm = np.linalg.norm(residual, axis=1)
            converged |= residual_norm <= tolerance
            if np.all(converged):
                break
            active = ~converged
            try:
                correction = np.linalg.solve(jacobian[active], -residual[active, :, None])[..., 0]
            except np.linalg.LinAlgError as error:
                raise ConstitutiveIntegrationError(
                    "driven J2 local Jacobian is singular"
                ) from error

            active_indices = np.flatnonzero(active)
            accepted = np.zeros(active_indices.size, dtype=bool)
            step = np.ones(active_indices.size, dtype=np.float64)
            base_norm = residual_norm[active]
            candidate_stress = stress[active].copy()
            for _ in range(self._maximum_line_search_iterations):
                candidates = stress[active] + step[:, None] * correction
                candidate_residual, _, _ = self._residual_and_jacobian(
                    candidates,
                    trial_stress[active],
                    increment[active],
                )
                candidate_norm = np.linalg.norm(candidate_residual, axis=1)
                admissible = np.isfinite(candidate_norm) & (
                    candidate_norm <= (1.0 - 1.0e-4 * step) * base_norm
                )
                newly_accepted = admissible & ~accepted
                candidate_stress[newly_accepted] = candidates[newly_accepted]
                accepted |= admissible
                if np.all(accepted):
                    break
                step[~accepted] *= 0.5
            if not np.all(accepted):
                first = int(active_indices[np.flatnonzero(~accepted)[0]])
                local_index = int(np.flatnonzero(active_indices == first)[0])
                error = ConstitutiveIntegrationError(
                    f"driven J2 local line search failed at point {first}"
                )
                error.diagnostics = {
                    "failure_stage": "local_line_search",
                    "point": first,
                    "q_trial_mpa": float(von_mises(trial_stress[first][None])[0]),
                    "delta_p": float(increment[first]),
                    "newton_iteration": iteration,
                    "base_residual_mpa": float(base_norm[local_index]),
                    "jacobian_condition": float(np.linalg.cond(jacobian[local_index])),
                    "line_search_iterations": self._maximum_line_search_iterations,
                    "last_step": float(step[local_index]),
                    "last_residual_mpa": float(candidate_norm[local_index]),
                }
                raise error
            stress[active] = candidate_stress
            iterations[active] = iteration
        else:
            residual, _, _ = self._residual_and_jacobian(stress, trial_stress, increment)
            residual_norm = np.linalg.norm(residual, axis=1)
            first = int(np.argmax(residual_norm / tolerance))
            raise ConstitutiveIntegrationError(
                "driven J2 local Newton did not converge; "
                f"point {first}, residual={residual_norm[first]:.3e} MPa"
            )

        residual, jacobian, direction = self._residual_and_jacobian(stress, trial_stress, increment)
        residual_norm = np.linalg.norm(residual, axis=1)
        identity = np.broadcast_to(np.eye(3), (self.point_count, 3, 3))
        residual_parameters = np.empty((self.point_count, 3, 4), dtype=np.float64)
        residual_parameters[:, :, :3] = -self._elasticity[None, :, :]
        residual_parameters[:, :, 3] = np.einsum("ij,pj->pi", self._elasticity, direction)
        sensitivities = solve_implicit_sensitivities(
            jacobian,
            residual_parameters,
            identity,
        )
        return (
            stress,
            direction,
            sensitivities[:, :, :3],
            sensitivities[:, :, 3],
            np.column_stack((residual_norm, iterations.astype(np.float64))),
        )

    def _solve_stress_with_continuation(
        self,
        trial_stress: FloatArray,
        increment: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """Retry the same local problem through a non-committed Delta-p homotopy."""
        stress = trial_stress.copy()
        result: tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray] | None = None
        for scale in np.linspace(0.0, 1.0, 17):
            try:
                result = self._solve_stress(
                    trial_stress,
                    scale * increment,
                    initial_stress=stress,
                )
            except ConstitutiveIntegrationError as error:
                diagnostics = getattr(error, "diagnostics", {})
                diagnostics["continuation_scale"] = float(scale)
                error.diagnostics = diagnostics
                raise
            stress = result[0]
        if result is None:
            raise ConstitutiveIntegrationError("local Delta-p continuation produced no state")
        return result

    def evaluate(
        self,
        in_plane_strain: ArrayLike,
        equivalent_plastic_increment: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> DrivenJ2Trial:
        if not np.isfinite(time_increment) or time_increment <= 0:
            raise ValueError("time_increment must be finite and positive")
        strain = np.asarray(in_plane_strain, dtype=np.float64)
        if strain.shape != (self.point_count, 3):
            raise ValueError(f"in_plane_strain must have shape {(self.point_count, 3)}")
        if not np.isfinite(strain).all():
            raise ValueError("in_plane_strain must be finite")
        increment = np.asarray(equivalent_plastic_increment, dtype=np.float64)
        try:
            increment = np.broadcast_to(increment, (self.point_count,)).copy()
        except ValueError as error:
            raise ValueError(
                f"equivalent_plastic_increment must broadcast to {(self.point_count,)}"
            ) from error
        if not np.isfinite(increment).all() or np.any(increment < 0.0):
            raise ValueError("equivalent_plastic_increment must be finite and non-negative")

        trial_stress = np.einsum(
            "ij,pj->pi",
            self._elasticity,
            strain - self._committed_plastic_strain,
        )
        try:
            stress, direction, tangent, increment_tangent, diagnostics = self._solve_stress(
                trial_stress,
                increment,
            )
        except ConstitutiveIntegrationError:
            stress, direction, tangent, increment_tangent, diagnostics = (
                self._solve_stress_with_continuation(trial_stress, increment)
            )
        trial_plastic = self._committed_plastic_strain + increment[:, None] * direction
        trial_peeq = self._committed_peeq + increment
        full = reconstruct_python_plane_stress_state(
            strain,
            trial_plastic,
            stress,
            self._poisson,
        )
        self._trial_plastic_strain = trial_plastic
        self._trial_peeq = trial_peeq
        self._last_maximum_residual = float(np.max(diagnostics[:, 0]))
        self._last_maximum_iterations = int(np.max(diagnostics[:, 1]))
        return DrivenJ2Trial(
            stress_in_plane_mpa=stress,
            tangent_in_plane_mpa=tangent if consistent_tangent else None,
            stress_equivalent_plastic_increment_tangent_mpa=increment_tangent,
            local_residual_norm_mpa=diagnostics[:, 0],
            local_iterations=diagnostics[:, 1],
            full_stress_tensor_mpa=full.stress_tensor_mpa,
            full_strain_tensor=full.total_strain_tensor,
            elastic_strain_tensor=full.elastic_strain_tensor,
            plastic_strain_tensor=full.plastic_strain_tensor,
            plane_stress_residual_mpa=full.plane_stress_residual_vector_mpa,
            observables={
                "plastic_strain_2d": trial_plastic,
                "equivalent_plastic_strain": trial_peeq,
                "equivalent_plastic_increment": increment,
                "flow_direction": direction,
            },
        )

    def commit(self) -> None:
        if self._trial_plastic_strain is None or self._trial_peeq is None:
            raise RuntimeError("no successful driven J2 trial state to commit")
        self._committed_plastic_strain = self._trial_plastic_strain.copy()
        self._committed_peeq = self._trial_peeq.copy()
        self.revert()

    def revert(self) -> None:
        self._trial_plastic_strain = None
        self._trial_peeq = None
