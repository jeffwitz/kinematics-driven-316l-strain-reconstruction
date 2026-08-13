"""External 3D-to-plane-stress condensation bridges."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.linear_solver import LinearSystemMatrixType
from fem_inhouse.core.mfront_3d import MFront3DMaterialPointBatch, _MFront3DTrial
from fem_inhouse.core.mfront_runtime import (
    _ENGINEERING_TO_KELVIN_STRAIN_SCALE,
    _KELVIN_TO_ENGINEERING_STRESS_SCALE,
    MFrontUnavailableError,
)
from fem_inhouse.core.mfront_state import (
    MFrontCondensedBlocksStateSnapshot,
    MFrontCondensedStateSnapshot,
    MFrontTimingStatistics,
)
from fem_inhouse.core.plane_stress_material import (
    ConstitutiveTrial,
    InPlaneConstitutiveTrial,
    LocalPlaneStressConvergenceError,
    PlaneStressBatchStatistics,
    ResponseLevel,
)
from fem_inhouse.core.tensor_reconstruction import (
    kelvin_3d_to_tensor,
    tensor_to_engineering_strain_2d,
    tensor_to_engineering_stress_2d,
)

_SQRT_TWO = np.sqrt(2.0)
_PLANE_STRESS_COMPONENTS = np.array([0, 1, 3])
_TRANSVERSE_COMPONENTS_3D = np.array([2, 4, 5])
_MFront3DMaterialPointBatch: type[MFront3DMaterialPointBatch] = MFront3DMaterialPointBatch
LocalConditionCheckMode = Literal["always", "on_failure", "diagnostic_sample"]

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


def condense_kelvin_tangent_blocks(
    stress_strain: ArrayLike,
    stress_chi: ArrayLike,
    source_strain: ArrayLike,
    source_chi: ArrayLike,
    *,
    check_condition: bool = True,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray | None]:
    """Condense a four-block 3-D tangent through plane stress.

    The inputs describe the local map ``(strain, chi) -> (stress, source)``
    in Kelvin components.  The transverse strain components ``(zz, xz, yz)``
    are eliminated by the plane-stress constraints.  The returned blocks are
    ordered as ``Cps, stress_chi_ps, source_strain_ps, source_chi_ps`` and use
    the same Kelvin in-plane convention as the existing tangent helper.

    This is deliberately independent of MGIS and of the constitutive law: it
    is the reusable algebraic layer needed by a future GenericBehaviour
    bridge.
    """

    c_ee = np.asarray(stress_strain, dtype=float)
    s_chi = np.asarray(stress_chi, dtype=float)
    q_eps = np.asarray(source_strain, dtype=float)
    q_chi = np.asarray(source_chi, dtype=float)
    if c_ee.shape[-2:] != (6, 6):
        raise ValueError("stress_strain must have trailing shape (6, 6)")
    if s_chi.shape[-2:] != (6, 1):
        raise ValueError("stress_chi must have trailing shape (6, 1)")
    if q_eps.shape[-2:] != (1, 6):
        raise ValueError("source_strain must have trailing shape (1, 6)")
    if q_chi.shape[-2:] != (1, 1):
        raise ValueError("source_chi must have trailing shape (1, 1)")
    batch_shape = np.broadcast_shapes(
        c_ee.shape[:-2], s_chi.shape[:-2], q_eps.shape[:-2], q_chi.shape[:-2]
    )
    c_ee = np.broadcast_to(c_ee, (*batch_shape, 6, 6))
    s_chi = np.broadcast_to(s_chi, (*batch_shape, 6, 1))
    q_eps = np.broadcast_to(q_eps, (*batch_shape, 1, 6))
    q_chi = np.broadcast_to(q_chi, (*batch_shape, 1, 1))
    if not all(np.isfinite(values).all() for values in (c_ee, s_chi, q_eps, q_chi)):
        raise ValueError("four-block tangent inputs must be finite")
    caa = c_ee[..., _PLANE_STRESS_COMPONENTS, :][..., _PLANE_STRESS_COMPONENTS]
    cab = c_ee[..., _PLANE_STRESS_COMPONENTS, :][..., _TRANSVERSE_COMPONENTS_3D]
    cba = c_ee[..., _TRANSVERSE_COMPONENTS_3D, :][..., _PLANE_STRESS_COMPONENTS]
    cbb = c_ee[..., _TRANSVERSE_COMPONENTS_3D, :][..., _TRANSVERSE_COMPONENTS_3D]
    s_chi_a = s_chi[..., _PLANE_STRESS_COMPONENTS, :]
    s_chi_b = s_chi[..., _TRANSVERSE_COMPONENTS_3D, :]
    q_eps_a = q_eps[..., :, _PLANE_STRESS_COMPONENTS]
    q_eps_b = q_eps[..., :, _TRANSVERSE_COMPONENTS_3D]
    cbb_inv_cba = np.linalg.solve(cbb, cba)
    cbb_inv_s_chi_b = np.linalg.solve(cbb, s_chi_b)
    c_ps = caa - cab @ cbb_inv_cba
    s_chi_ps = s_chi_a - cab @ cbb_inv_s_chi_b
    q_eps_ps = q_eps_a - q_eps_b @ cbb_inv_cba
    q_chi_ps = q_chi - q_eps_b @ cbb_inv_s_chi_b
    condition = np.linalg.cond(cbb) if check_condition else None
    return c_ps, s_chi_ps, q_eps_ps, q_chi_ps, condition


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
            in_plane_strain, time_increment=time_increment, consistent_tangent=False
        )
        return trial.observables["equivalent_plastic_strain"]

    def evaluate_nonlocal_state(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
    ) -> tuple[NDArray, NDArray]:
        trial = self.evaluate_in_plane_response(
            in_plane_strain,
            time_increment=time_increment,
            response_level="complete",
            consistent_tangent=False,
        )
        if "equivalent_plastic_strain" in trial.observables:
            return (
                trial.observables["equivalent_plastic_strain"],
                trial.observables["yield_surface_radius_mpa"],
            )
        # Crystal plasticity has no scalar J2 yield radius.  The generic
        # non-local criterion uses the positive accumulated-slip source and a
        # strictly positive safety sentinel; SRIX admissibility is handled by
        # its local integration/plane-stress transaction.
        if "accumulated_slip" in trial.observables:
            source = np.asarray(trial.observables["accumulated_slip"], dtype=float)
            return source, np.ones_like(source)
        raise MFrontUnavailableError(
            "the behaviour exposes neither equivalent_plastic_strain nor "
            "accumulated_slip for non-local coupling"
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
