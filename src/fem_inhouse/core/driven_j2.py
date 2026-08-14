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


#: Common eigenbasis of the plane-stress elasticity and the von Mises metric.
#:
#: In ``(S11, S22, S12)`` engineering-shear coordinates, ``C`` and ``M`` are
#: simultaneously diagonalised by the orthonormal rows below: the hydrostatic
#: in-plane mode, the deviatoric in-plane mode, and the shear mode. ``M`` has
#: the fixed eigenvalues 1/2, 3/2 and 3; the eigenvalues of ``C`` depend on
#: ``(E, nu)`` only. This is what collapses the 3x3 local system to a scalar.
_SQRT_HALF = float(np.sqrt(0.5))
_MODAL_BASIS = np.array(
    [
        [_SQRT_HALF, _SQRT_HALF, 0.0],
        [_SQRT_HALF, -_SQRT_HALF, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
_MODAL_METRIC_EIGENVALUES = np.array([0.5, 1.5, 3.0])

#: Iteration cap of the scalar return. Bisection alone halves the bracket each
#: time, so this covers a bracket spanning every order of magnitude float64 can
#: hold; the Newton steps make the usual count a handful.
_SCALAR_RETURN_MAXIMUM_ITERATIONS = 100


def _modal_elasticity_eigenvalues(young_modulus_mpa: float, poisson_ratio: float) -> FloatArray:
    """Eigenvalues of the plane-stress elasticity in :data:`_MODAL_BASIS`."""

    return np.array(
        [
            young_modulus_mpa / (1.0 - poisson_ratio),
            young_modulus_mpa / (1.0 + poisson_ratio),
            young_modulus_mpa / (2.0 * (1.0 + poisson_ratio)),
        ]
    )


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
        self._modal_elasticity = _modal_elasticity_eigenvalues(self._young, self._poisson)
        #: ``a_i / Delta_p`` of the scalar reduction, and the coefficients of the
        #: closed-form admissibility bound.
        self._modal_relaxation = self._modal_elasticity * _MODAL_METRIC_EIGENVALUES
        self._modal_bound_weights = 1.0 / (
            self._modal_elasticity**2 * _MODAL_METRIC_EIGENVALUES
        )
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

    def maximum_admissible_equivalent_plastic_increment(
        self, in_plane_strain: ArrayLike
    ) -> FloatArray:
        """Largest ``Delta p`` this trial state can absorb, per point.

        Associated J2 relaxes the deviatoric stress towards the origin as
        ``Delta p`` grows, and reaches it at a finite value: beyond that the
        local equation has no solution with ``q > 0``, because the flow would
        have to pass through the origin where the direction is undefined. The
        bound is closed-form in the modal basis,

        ``Delta p_max = sqrt(sum_i t_i^2 / (c_i^2 m_i))``,

        with ``t`` the modal trial stress. An optimiser driving ``Delta p``
        should project onto ``[0, Delta p_max)`` rather than discover the wall
        as an integration failure -- that wall is what stopped the directional
        replay at state 21, point 117.
        """

        strain = np.asarray(in_plane_strain, dtype=np.float64)
        if strain.shape != (self.point_count, 3):
            raise ValueError(f"in_plane_strain must have shape {(self.point_count, 3)}")
        trial_stress = np.einsum(
            "ij,pj->pi", self._elasticity, strain - self._committed_plastic_strain
        )
        modal_trial = trial_stress @ _MODAL_BASIS.T
        return np.sqrt(modal_trial**2 @ self._modal_bound_weights)

    def _solve_equivalent_stress(
        self, modal_trial: FloatArray, increment: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Solve ``sum_i m_i t_i^2 / (q + a_i)^2 = 1`` for every plastic point.

        Written in the common eigenbasis of ``C`` and ``M``, the 3x3 local
        system collapses to this one scalar equation, with
        ``a_i = Delta p * c_i * m_i``. The left-hand side is strictly
        decreasing and convex on ``[0, inf)``, so:

        * the root is unique and bracketed by ``[0, q_trial]``;
        * convexity puts every Newton iterate on the same side of it, so
          Newton started at ``q = 0`` increases monotonically to the root and
          can never overshoot.

        There is therefore no line search to fail, no 3x3 Jacobian to
        condition, and no branch to follow. Non-existence stops being a
        numerical accident and becomes the explicit test ``phi(0) > 1``.
        """

        weights = modal_trial**2 * _MODAL_METRIC_EIGENVALUES
        relaxation = increment[:, None] * self._modal_relaxation[None, :]
        # phi(0) > 1 is exactly Delta p < Delta p_max; below the bound the
        # solve is guaranteed, so this is the only place existence is decided.
        bound = np.sqrt(modal_trial**2 @ self._modal_bound_weights)
        # A point with Delta p = 0 is elastic and imposes nothing, including at
        # a virgin state where the bound itself is zero.
        inadmissible = (increment > 0.0) & (increment >= bound)
        if np.any(inadmissible):
            first = int(np.flatnonzero(inadmissible)[0])
            error = ConstitutiveIntegrationError(
                "prescribed Delta p exceeds what the trial state can absorb at "
                f"point {first}: Delta p={increment[first]:.6e} >= "
                f"Delta p_max={bound[first]:.6e}"
            )
            error.diagnostics = {
                "failure_stage": "delta_p_above_admissible_bound",
                "point": first,
                "delta_p": float(increment[first]),
                "delta_p_max": float(bound[first]),
                "q_trial_mpa": float(np.sqrt(weights[first].sum())),
            }
            raise error

        equivalent = np.zeros(increment.shape[0], dtype=np.float64)
        iterations = np.zeros(increment.shape[0], dtype=np.float64)
        active = increment > 0.0
        if not np.any(active):
            return np.sqrt(weights.sum(axis=1)), iterations
        w = weights[active]
        a = relaxation[active]
        reference = np.sqrt(w.sum(axis=1))
        # Starting point. When the two distinct relaxation values coincide the
        # equation collapses to the classical radial return `q = q_trial - a`,
        # which is exact; they never coincide here, but they stay within a
        # factor of three of each other, so the weighted-mean version of that
        # formula lands close to the root over the whole admissible range.
        #
        # This matters more than it looks. Newton from `q = 0` is monotone and
        # cannot overshoot, but far below the wall it advances by a factor of
        # about 1.5 per iteration from a first step of order `a/2`, so reaching
        # a root of order `q_trial` takes `log(q_trial / a)` iterations -- 50
        # was not enough at a continuation scale of 2.4e-4, which is exactly
        # how the replay failed at point 281.
        effective = np.sum(w * a, axis=1) / np.sum(w, axis=1)
        low = np.zeros(w.shape[0], dtype=np.float64)
        high = reference.copy()
        q = np.clip(reference - effective, 0.0, reference)
        # `phi` is dimensionless and equals 1 at the root, so the residual test
        # needs no scaling; the step test catches a root that has settled into
        # the last bits, which happens near the wall where the root approaches
        # zero. Both are per point: an all-or-nothing test lets one stagnating
        # point fail an otherwise converged batch.
        settled = np.zeros(w.shape[0], dtype=bool)
        for iteration in range(1, _SCALAR_RETURN_MAXIMUM_ITERATIONS + 1):
            shifted = q[:, None] + a
            phi = np.sum(w / shifted**2, axis=1)
            derivative = -2.0 * np.sum(w / shifted**3, axis=1)
            # phi is strictly decreasing, so this brackets the root without a
            # single extra function evaluation.
            above = phi > 1.0
            low = np.where(above & ~settled, q, low)
            high = np.where(~above & ~settled, q, high)
            candidate = q + (1.0 - phi) / derivative
            # Convexity keeps an exact Newton step inside the bracket; finite
            # precision does not, and a step that leaves it is replaced by a
            # bisection, which alone would converge in about sixty iterations
            # from any bracket this problem can produce.
            outside = ~((candidate > low) & (candidate < high))
            candidate = np.where(outside, 0.5 * (low + high), candidate)
            update = candidate - q
            q = np.where(settled, q, candidate)
            iterations[active] = iteration
            settled |= np.abs(update) <= 4.0 * np.finfo(float).eps * np.maximum(q, reference)
            if np.all(settled):
                break
        if not np.all(settled):
            shifted = q[:, None] + a
            worst = int(np.argmax(np.abs(np.sum(w / shifted**2, axis=1) - 1.0)))
            error = ConstitutiveIntegrationError(
                "driven J2 scalar return did not converge at point "
                f"{int(np.flatnonzero(active)[worst])}"
            )
            error.diagnostics = {
                "failure_stage": "scalar_return",
                "point": int(np.flatnonzero(active)[worst]),
                "phi_minus_one": float(np.sum(w[worst] / (q[worst] + a[worst]) ** 2) - 1.0),
                "equivalent_stress_mpa": float(q[worst]),
                "trial_equivalent_stress_mpa": float(reference[worst]),
                "bracket": [float(low[worst]), float(high[worst])],
                "iterations": _SCALAR_RETURN_MAXIMUM_ITERATIONS,
            }
            raise error
        equivalent[active] = q
        equivalent[~active] = np.sqrt(weights[~active].sum(axis=1))
        return equivalent, iterations

    def _solve_stress(
        self,
        trial_stress: FloatArray,
        increment: FloatArray,
        initial_stress: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        # `initial_stress` is accepted for interface compatibility with the
        # homotopy retry and deliberately ignored: the scalar return needs no
        # starting guess, so a warm start can only make the answer depend on
        # the path taken to it.
        del initial_stress
        trial_equivalent = von_mises(trial_stress)
        impossible = (increment > 0.0) & (trial_equivalent <= self._stress_floor)
        if np.any(impossible):
            first = int(np.flatnonzero(impossible)[0])
            raise ConstitutiveIntegrationError(
                "positive Delta p requires a non-zero trial J2 direction; "
                f"point {first} has q_trial={trial_equivalent[first]:.3e} MPa"
            )

        modal_trial = trial_stress @ _MODAL_BASIS.T
        equivalent, iterations = self._solve_equivalent_stress(modal_trial, increment)
        # sigma_i = t_i q / (q + a_i), exactly, once q is known. An elastic
        # point keeps its trial stress: at Delta p = 0 the ratio is 1 by
        # construction but degenerates to 0/0 at a virgin state.
        plastic = increment > 0.0
        shifted = equivalent[:, None] + increment[:, None] * self._modal_relaxation[None, :]
        modal_stress = modal_trial.copy()
        modal_stress[plastic] *= equivalent[plastic, None] / shifted[plastic]
        stress = modal_stress @ _MODAL_BASIS

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
        scale = 0.0
        step = 0.25
        attempts = 0
        while scale < 1.0 - 1.0e-14:
            target = min(1.0, scale + step)
            try:
                result = self._solve_stress(
                    trial_stress,
                    target * increment,
                    initial_stress=stress,
                )
            except ConstitutiveIntegrationError as error:
                diagnostics = getattr(error, "diagnostics", {})
                diagnostics["continuation_scale"] = float(target)
                diagnostics["continuation_step"] = float(step)
                error.diagnostics = diagnostics
                step *= 0.5
                attempts += 1
                if step < 2.0**-12 or attempts > 64:
                    raise
                continue
            stress = result[0]
            scale = target
            attempts = 0
            step = min(0.25, 1.0 - scale) if scale < 1.0 else 0.0
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
