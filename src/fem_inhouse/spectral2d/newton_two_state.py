"""Newton-GMRES oracle using two independent SRIX states per TRI2 pixel."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import LinearOperator

from fem_inhouse.core.plane_stress_material import (
    InPlaneConstitutiveTrial,
    PlaneStressMaterialBatch,
    ResponseLevel,
    evaluate_in_plane_response,
)
from fem_inhouse.spectral2d.boundary import HarmonicDirichletExtension2D
from fem_inhouse.spectral2d.diagnostics import (
    JacobianActionDiagnostics,
    LinearSolveDiagnostics,
    LoadStepAttemptDiagnostics,
    PreconditionerActionDiagnostics,
    Spectral2DDiagnostics,
    collect_runtime_provenance,
    summarize_load_step_attempts,
)
from fem_inhouse.spectral2d.green import B0Green2D, project_isotropic_plane_stress_tangent
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.spectral2d.krylov import KrylovRecycleState, solve_nonsymmetric_krylov
from fem_inhouse.spectral2d.newton_ebi import (
    EBISpectralSolverConfig,
    pack_interior,
    pack_interior_into,
    unpack_interior,
    unpack_interior_into,
)
from fem_inhouse.spectral2d.nonlinear import _boundary_reactions, _equilibrium_metrics
from fem_inhouse.spectral2d.result import Spectral2DResult
from fem_inhouse.spectral2d.step_control import (
    AdaptiveLoadPath,
    LoadPathStep,
    LoadStepObservation,
    predictive_slip_error_ratio,
)
from fem_inhouse.spectral2d.step_doubling import (
    LoadStepAttempt,
    StepDoublingFailureError,
    StepObservables,
    estimate_step_error_by_doubling,
    step_error_to_record,
)
from fem_inhouse.spectral2d.transform_factory import create_full_dirichlet_dsti_plan
from fem_inhouse.spectral2d.transforms import BufferedTransformPlan2D, TransformPlan2D

FloatArray = NDArray[np.float64]
TangentKernel = Literal["einsum", "explicit"]
EvaluationKind = Literal["newton_tangent", "line_search", "verification"]


def _emit_progress_raw(config: EBISpectralSolverConfig, event: dict[str, object]) -> None:
    callback = config.progress_callback
    if callback is not None:
        callback(event)


@dataclass(slots=True)
class TwoStateJacobianWorkspace:
    """Persistent buffers for one two-state TRI2 Newton resolution."""

    nodal_increment: FloatArray
    sample_strain_increment: FloatArray
    sample_stress_increment: FloatArray
    nodal_force: FloatArray
    interior_force: FloatArray

    @classmethod
    def create(cls, grid: StructuredGrid2D) -> TwoStateJacobianWorkspace:
        return cls(
            nodal_increment=np.zeros((*grid.node_shape, 2), dtype=np.float64),
            sample_strain_increment=np.empty((*grid.pixel_shape, 2, 3), dtype=np.float64),
            sample_stress_increment=np.empty((*grid.pixel_shape, 2, 3), dtype=np.float64),
            nodal_force=np.empty((*grid.node_shape, 2), dtype=np.float64),
            interior_force=np.empty(2 * (grid.nx - 1) * (grid.ny - 1), dtype=np.float64),
        )


def apply_tangent_into(
    tangent: ArrayLike,
    strain: ArrayLike,
    destination: FloatArray,
    *,
    kernel: TangentKernel = "einsum",
) -> None:
    """Apply all 3x3 sample tangents into a reusable stress buffer."""

    values = np.asarray(tangent, dtype=np.float64)
    delta = np.asarray(strain, dtype=np.float64)
    if destination.shape != delta.shape or values.shape != (*delta.shape[:-1], 3, 3):
        raise ValueError("incompatible tangent action shapes")
    if kernel == "einsum":
        np.einsum("xyqij,xyqj->xyqi", values, delta, out=destination)
    elif kernel == "explicit":
        e0, e1, e2 = delta[..., 0], delta[..., 1], delta[..., 2]
        destination[..., 0] = (
            values[..., 0, 0] * e0 + values[..., 0, 1] * e1 + values[..., 0, 2] * e2
        )
        destination[..., 1] = (
            values[..., 1, 0] * e0 + values[..., 1, 1] * e1 + values[..., 1, 2] * e2
        )
        destination[..., 2] = (
            values[..., 2, 0] * e0 + values[..., 2, 1] * e1 + values[..., 2, 2] * e2
        )
    else:
        raise ValueError(f"unsupported tangent kernel: {kernel}")


def _accumulate_jacobian(
    target: JacobianActionDiagnostics,
    source: JacobianActionDiagnostics,
) -> None:
    target.calls += source.calls
    target.total_seconds += source.total_seconds
    target.unpack_seconds += source.unpack_seconds
    target.gradient_seconds += source.gradient_seconds
    target.tangent_seconds += source.tangent_seconds
    target.divergence_seconds += source.divergence_seconds
    target.pack_seconds += source.pack_seconds


def _accumulate_preconditioner(
    target: PreconditionerActionDiagnostics,
    source: PreconditionerActionDiagnostics,
) -> None:
    target.calls += source.calls
    target.total_seconds += source.total_seconds
    target.reshape_seconds += source.reshape_seconds
    target.forward_transform_seconds += source.forward_transform_seconds
    target.green_seconds += source.green_seconds
    target.inverse_transform_seconds += source.inverse_transform_seconds
    target.output_copy_seconds += source.output_copy_seconds


def _linear_tolerance(
    config: EBISpectralSolverConfig,
    residual: float,
    previous_residual: float | None,
    *,
    force_fixed: bool,
) -> float:
    if config.linear_tolerance_mode == "fixed" or force_fixed:
        return config.gmres_relative_tolerance
    if previous_residual is None:
        eta = config.forcing_initial
    else:
        ratio = residual / max(previous_residual, 1.0e-30)
        eta = config.forcing_gamma * ratio**config.forcing_alpha
        eta = min(config.forcing_maximum, max(config.forcing_minimum, eta))
    if residual <= 10.0 * config.relative_equilibrium_tolerance:
        eta = min(
            eta,
            max(config.forcing_minimum, 0.1 * residual),
        )
    return max(config.forcing_minimum, min(config.forcing_maximum, eta))


@dataclass(frozen=True, slots=True)
class TraditionalTwoStateTrial:
    sample_strain: FloatArray
    sample_stress_mpa: FloatArray
    algorithmic_tangent_in_plane_mpa: FloatArray | None
    mean_stress_mpa: FloatArray
    material_trial: InPlaneConstitutiveTrial


@dataclass(slots=True)
class AcceptedTwoStateTrialCache:
    """Own the accepted line-search trial until the next Newton iteration."""

    trial: TraditionalTwoStateTrial | None = None
    sample_strain: FloatArray | None = None
    residual: FloatArray | None = None
    relative: float | None = None
    absolute: float | None = None

    @property
    def populated(self) -> bool:
        return self.trial is not None

    def store(
        self,
        *,
        trial: TraditionalTwoStateTrial,
        sample_strain: FloatArray,
        residual: FloatArray,
        relative: float,
        absolute: float,
    ) -> None:
        self.trial = trial
        self.sample_strain = np.asarray(sample_strain, dtype=np.float64).copy()
        # The nonlinear divergence uses a persistent mutable buffer. The cache
        # must own its residual until the next Newton iteration consumes it.
        self.residual = np.asarray(residual, dtype=np.float64).copy()
        self.relative = float(relative)
        self.absolute = float(absolute)

    def take(
        self,
    ) -> tuple[TraditionalTwoStateTrial, FloatArray, FloatArray, float, float]:
        if (
            self.trial is None
            or self.sample_strain is None
            or self.residual is None
            or self.relative is None
            or self.absolute is None
        ):
            raise RuntimeError("accepted TRI2 trial cache is empty")
        result = (
            self.trial,
            self.sample_strain,
            self.residual,
            self.relative,
            self.absolute,
        )
        self.clear()
        return result

    def clear(self) -> None:
        self.trial = None
        self.sample_strain = None
        self.residual = None
        self.relative = None
        self.absolute = None


class TraditionalTwoStateTriangleBatch:
    """Direct TRI2 material integration with two independent histories."""

    def __init__(self, material: PlaneStressMaterialBatch, pixel_shape: tuple[int, int]):
        self.material = material
        self.pixel_shape = pixel_shape
        if material.point_count != 2 * pixel_shape[0] * pixel_shape[1]:
            raise ValueError("two-state TRI2 requires two material states per pixel")

    def evaluate_samples(
        self,
        sample_strain: ArrayLike,
        *,
        time_increment: float,
        response_level: ResponseLevel = "tangent",
    ) -> TraditionalTwoStateTrial:
        values = np.asarray(sample_strain, dtype=np.float64)
        expected = (*self.pixel_shape, 2, 3)
        if values.shape != expected:
            raise ValueError(f"expected sample strain shape {expected}, got {values.shape}")
        material_trial = evaluate_in_plane_response(
            self.material,
            values.reshape(-1, 3),
            time_increment=time_increment,
            response_level=response_level,
            consistent_tangent=response_level != "residual",
        )
        stress = np.asarray(material_trial.stress_in_plane_mpa).reshape(*self.pixel_shape, 2, 3)
        tangent = (
            None
            if material_trial.tangent_in_plane_mpa is None
            else np.asarray(material_trial.tangent_in_plane_mpa).reshape(
                *self.pixel_shape, 2, 3, 3
            )
        )
        return TraditionalTwoStateTrial(
            sample_strain=values.copy(),
            sample_stress_mpa=stress,
            algorithmic_tangent_in_plane_mpa=tangent,
            mean_stress_mpa=stress.mean(axis=2),
            material_trial=material_trial,
        )

    def tangent_action(
        self,
        displacement_increment: ArrayLike,
        *,
        kinematics: TwoSubcellDiagnostic2D,
        trial: TraditionalTwoStateTrial,
        action_diagnostics: JacobianActionDiagnostics | None = None,
    ) -> FloatArray:
        gradient_started = time.perf_counter()
        delta_sample = kinematics.strain_samples(displacement_increment)
        gradient_seconds = time.perf_counter() - gradient_started
        tangent_started = time.perf_counter()
        if trial.algorithmic_tangent_in_plane_mpa is None:
            raise ValueError("TRI2 tangent action requires a tangent trial")
        delta_stress = np.einsum(
            "xyqij,xyqj->xyqi", trial.algorithmic_tangent_in_plane_mpa, delta_sample
        )
        tangent_seconds = time.perf_counter() - tangent_started
        divergence_started = time.perf_counter()
        result = kinematics.divergence_from_sample_stress(delta_stress)
        divergence_seconds = time.perf_counter() - divergence_started
        if action_diagnostics is not None:
            action_diagnostics.gradient_seconds += gradient_seconds
            action_diagnostics.tangent_seconds += tangent_seconds
            action_diagnostics.divergence_seconds += divergence_seconds
        return result

    def tangent_action_into(
        self,
        *,
        kinematics: TwoSubcellDiagnostic2D,
        trial: TraditionalTwoStateTrial,
        workspace: TwoStateJacobianWorkspace,
        action_diagnostics: JacobianActionDiagnostics | None = None,
        kernel: TangentKernel = "einsum",
    ) -> FloatArray:
        """Apply the tangent using persistent TRI2 work arrays."""

        gradient_started = time.perf_counter()
        if trial.algorithmic_tangent_in_plane_mpa is None:
            raise ValueError("TRI2 tangent action requires a tangent trial")
        kinematics.strain_samples_into(
            workspace.nodal_increment,
            workspace.sample_strain_increment,
        )
        gradient_seconds = time.perf_counter() - gradient_started
        tangent_started = time.perf_counter()
        apply_tangent_into(
            trial.algorithmic_tangent_in_plane_mpa,
            workspace.sample_strain_increment,
            workspace.sample_stress_increment,
            kernel=kernel,
        )
        tangent_seconds = time.perf_counter() - tangent_started
        divergence_started = time.perf_counter()
        kinematics.divergence_from_sample_stress_into(
            workspace.sample_stress_increment,
            workspace.nodal_force,
        )
        divergence_seconds = time.perf_counter() - divergence_started
        pack_started = time.perf_counter()
        pack_interior_into(workspace.nodal_force, workspace.interior_force)
        pack_seconds = time.perf_counter() - pack_started
        if action_diagnostics is not None:
            action_diagnostics.gradient_seconds += gradient_seconds
            action_diagnostics.tangent_seconds += tangent_seconds
            action_diagnostics.divergence_seconds += divergence_seconds
            action_diagnostics.pack_seconds += pack_seconds
        return workspace.interior_force.copy()

    def complete_trial(self, trial: TraditionalTwoStateTrial):
        return self.material.complete_trial(trial.material_trial)

    def commit(self) -> None:
        self.material.commit()

    def revert(self) -> None:
        self.material.revert()

    def accept_global_trial(self) -> None:
        accept = getattr(self.material, "accept_global_trial", None)
        if callable(accept):
            accept()

    def snapshot_state(self) -> object:
        snapshot = getattr(self.material, "snapshot_state", None)
        if not callable(snapshot):
            raise RuntimeError(
                "step-doubling requires a material backend with snapshot_state()"
            )
        return snapshot()

    def restore_state(self, snapshot: object) -> None:
        restore = getattr(self.material, "restore_state", None)
        if not callable(restore):
            raise RuntimeError(
                "step-doubling requires a material backend with restore_state()"
            )
        restore(snapshot)


def _reshape_two_state(values: ArrayLike, grid: StructuredGrid2D) -> FloatArray:
    array = np.asarray(values)
    return array.reshape(*grid.pixel_shape, 2, *array.shape[1:])


def _step_doubling_observables(result: Spectral2DResult) -> StepObservables:
    """Extract the constitutive fields controlled by SRIX step doubling."""

    required = ("plastic_slip", "equivalent_plastic_slip", "accumulated_slip")
    missing = [name for name in required if name not in result.observables]
    if missing:
        raise RuntimeError(
            "SRIX step-doubling requires per-system observables: "
            + ", ".join(missing)
        )
    return StepObservables(
        displacement=np.asarray(result.displacement, dtype=np.float64),
        stress_in_plane_mpa=np.asarray(result.stress_in_plane_mpa, dtype=np.float64),
        reaction_forces=np.asarray(result.reaction_forces, dtype=np.float64),
        plastic_slip=np.asarray(result.observables["plastic_slip"], dtype=np.float64),
        equivalent_plastic_slip=np.asarray(
            result.observables["equivalent_plastic_slip"], dtype=np.float64
        ),
        accumulated_slip=np.asarray(result.observables["accumulated_slip"], dtype=np.float64),
    )


def _solve_two_state_step_doubling(
    *,
    grid: StructuredGrid2D,
    material: PlaneStressMaterialBatch,
    history: FloatArray,
    config: EBISpectralSolverConfig,
    transform_plan: TransformPlan2D | None,
) -> Spectral2DResult:
    """Run SRIX step doubling with isolated recursive TRI2 attempts.

    The branch solver is deliberately the existing Newton-GMRES implementation
    with step doubling disabled.  Each branch receives a fresh material
    snapshot and returns a committed candidate snapshot; the controller only
    leaves the fine branch committed after an accepted comparison.  This first
    implementation favours transaction correctness over reuse of FFT plans or
    Newton workspaces between branches.
    """

    elements = TraditionalTwoStateTriangleBatch(material, grid.pixel_shape)
    initial_snapshot = elements.snapshot_state()
    branch_config = replace(
        config,
        adaptive_stepping_enabled=False,
        step_doubling=replace(config.step_doubling, enabled=False),
        verify_final_state=False,
    )
    branch_plan = transform_plan or create_full_dirichlet_dsti_plan(grid, config.transform)
    reference_parameters: tuple[float, float] | None = None
    zero_boundary = np.zeros_like(history[0])
    proportional = bool(
        np.allclose(
            history,
            np.linspace(0.0, 1.0, history.shape[0])[:, None, None, None] * history[-1],
            rtol=1.0e-12,
            atol=1.0e-15,
        )
    )

    def boundary_at(fraction: float) -> FloatArray:
        if proportional:
            return (fraction * history[-1]).copy()
        position = fraction * (history.shape[0] - 1)
        lower = min(int(np.floor(position)), history.shape[0] - 2)
        weight = position - lower
        return ((1.0 - weight) * history[lower] + weight * history[lower + 1]).copy()

    def next_mandatory_knot(fraction: float) -> float:
        if proportional:
            return 1.0
        count = history.shape[0] - 1
        next_index = int(np.floor(fraction * count + 1.0e-12)) + 1
        return min(1.0, next_index / count)

    def attempt(start: float, end: float, snapshot: object) -> LoadStepAttempt:
        nonlocal reference_parameters, branch_config
        elements.restore_state(snapshot)
        endpoint_history = np.stack((zero_boundary, boundary_at(end)))
        try:
            attempt_config = branch_config
            if reference_parameters is not None:
                attempt_config = replace(
                    branch_config,
                    reference_parameter_mode="explicit",
                    reference_lambda_0=reference_parameters[0],
                    reference_mu_0=reference_parameters[1],
                )
            result = solve_two_state_dirichlet_plane_stress(
                grid=grid,
                material=material,
                boundary_displacement_history=endpoint_history,
                config=attempt_config,
                transform_plan=branch_plan,
                time_increment_override=end - start,
            )
            if reference_parameters is None:
                reference_parameters = (
                    result.diagnostics.reference_lambda_0,
                    result.diagnostics.reference_mu_0,
                )
            state = elements.snapshot_state()
            return LoadStepAttempt(
                succeeded=True,
                start_fraction=start,
                end_fraction=end,
                state=state,
                observables=_step_doubling_observables(result),
                diagnostics=result,
            )
        except Exception as error:
            elements.restore_state(snapshot)
            return LoadStepAttempt(
                succeeded=False,
                start_fraction=start,
                end_fraction=end,
                state=None,
                observables=None,
                diagnostics=None,
                failure_reason=f"{type(error).__name__}: {error}",
            )

    current = 0.0
    step_size = config.adaptive_step.initial_increment_fraction
    final_result: Spectral2DResult | None = None
    history_entries: list[dict[str, object]] = []
    attempt_index = 0
    while current < 1.0 - 1.0e-14:
        segment_end = next_mandatory_knot(current)
        end = min(segment_end, current + step_size)
        if end <= current:
            raise RuntimeError("step-doubling controller made no progress")
        start_snapshot = initial_snapshot if current == 0.0 else elements.snapshot_state()
        attempt_index += 1
        doubling = estimate_step_error_by_doubling(
            current,
            end,
            start_snapshot,
            attempt_solver=attempt,
            config=config.step_doubling,
        )
        accepted = doubling.accepted
        error = doubling.error
        history_entries.append(
            {
                "attempt_index": attempt_index,
                "start_fraction": current,
                "end_fraction": end,
                "proposed_step_size": step_size,
                "effective_step_size": end - current,
                "mandatory_knot_limited": bool(end == segment_end and not proportional),
                "accepted": accepted,
                "decision_reason": doubling.decision_reason,
                "next_step_size": (end - current) * doubling.next_step_factor,
                "maximum_error_ratio": -1.0 if error is None else error.maximum_ratio,
                "controlling_quantity": "" if error is None else error.controlling_quantity,
                "controlling_system": (
                    -1
                    if error is None or error.controlling_system is None
                    else error.controlling_system
                ),
                "coarse_succeeded": doubling.coarse.succeeded,
                "first_half_succeeded": (
                    False if doubling.first_half is None else doubling.first_half.succeeded
                ),
                "second_half_succeeded": (
                    False if doubling.second_half is None else doubling.second_half.succeeded
                ),
                "error_details": None if error is None else step_error_to_record(error),
            }
        )
        if not accepted:
            elements.restore_state(start_snapshot)
            step_size = max(
                config.adaptive_step.minimum_increment_fraction,
                (end - current) * doubling.next_step_factor,
            )
            if end - current <= config.adaptive_step.minimum_increment_fraction * (1.0 + 1.0e-14):
                raise StepDoublingFailureError(
                    "step_error_tolerance_unreachable_at_minimum_step: "
                    f"{doubling.decision_reason}",
                    history_entries,
                )
            continue
        if doubling.second_half is None or doubling.second_half.observables is None:
            raise RuntimeError("accepted step-doubling result has no fine branch")
        if not isinstance(doubling.second_half.diagnostics, Spectral2DResult):
            raise RuntimeError("step-doubling attempt did not retain its fine result")
        final_result = doubling.second_half.diagnostics
        effective_step = end - current
        current = end
        step_size = min(
            config.adaptive_step.maximum_increment_fraction,
            max(
                config.adaptive_step.minimum_increment_fraction,
                effective_step * doubling.next_step_factor,
            ),
        )
    if final_result is None:
        raise RuntimeError("step-doubling controller accepted no load step")
    return replace(
        final_result,
        diagnostics=replace(
            final_result.diagnostics,
            adaptive_stepping_enabled=True,
            adaptive_step_history=tuple(history_entries),
        ),
    )


@dataclass(frozen=True, slots=True)
class TwoStateIncrementFields:
    """Converged fields of one increment, in the layout `Spectral2DResult` uses.

    The result keeps the last increment only, which is enough to qualify a
    solver and not enough to compare a simulated trajectory against a measured
    one. The three fields here are the ones an identification needs: the
    kinematics to observe, the stress that fixes the J2 flow direction, and the
    plastic strain that carries the history.
    """

    increment: int
    start_fraction: float
    end_fraction: float
    time_increment: float
    boundary: FloatArray
    displacement: FloatArray
    sample_strain: FloatArray
    stress_in_plane_mpa: FloatArray
    algorithmic_tangent_in_plane_mpa: FloatArray
    plastic_strain_tensor: FloatArray | None
    elastic_strain_tensor: FloatArray | None = None
    observables: dict[str, FloatArray] = field(default_factory=dict)


def solve_two_state_dirichlet_plane_stress(
    *,
    grid: StructuredGrid2D,
    material: PlaneStressMaterialBatch,
    boundary_displacement_history: ArrayLike,
    config: EBISpectralSolverConfig,
    transform_plan: TransformPlan2D | None = None,
    time_increment_override: float | None = None,
    load_path_override: Sequence[LoadPathStep] | None = None,
    initial_displacement: ArrayLike | None = None,
    initial_guess_callback: Callable[[LoadPathStep, FloatArray], ArrayLike] | None = None,
    increment_observer: Callable[[TwoStateIncrementFields], None] | None = None,
) -> Spectral2DResult:
    """Solve the direct two-state TRI2 oracle with the EBI Newton machinery.

    `increment_observer` receives the converged fields of every increment, in
    the layout the result uses. Comparing a simulated history against a
    measured one otherwise costs one full resolution per state, and the
    progress callback carries scalars only. The arrays are the solver's live
    buffers -- copy anything kept beyond the call.
    """

    history = np.asarray(boundary_displacement_history, dtype=np.float64)
    expected = (history.shape[0], *grid.node_shape, 2)
    if history.ndim != 4 or history.shape != expected or not np.allclose(history[0], 0.0):
        raise ValueError(f"invalid boundary history shape {history.shape}")
    initial_displacement_array = None
    if initial_displacement is not None:
        initial_displacement_array = np.asarray(initial_displacement, dtype=np.float64)
        if initial_displacement_array.shape != (*grid.node_shape, 2):
            raise ValueError(
                "initial_displacement must have nodal displacement shape "
                f"{(*grid.node_shape, 2)}"
            )
    if config.step_doubling.enabled:
        if increment_observer is not None:
            raise ValueError("increment_observer is not supported with step doubling")
        if not config.adaptive_stepping_enabled:
            raise ValueError("step-doubling requires adaptive stepping")
        return _solve_two_state_step_doubling(
            grid=grid,
            material=material,
            history=history,
            config=config,
            transform_plan=transform_plan,
        )
    kinematics = TwoSubcellDiagnostic2D(grid)
    elements = TraditionalTwoStateTriangleBatch(material, grid.pixel_shape)
    extension = HarmonicDirichletExtension2D()
    material_seconds = 0.0
    gradient_seconds = 0.0
    divergence_seconds = 0.0
    gmres_seconds = 0.0
    material_evaluations = 1
    material_initial_probe_evaluations = 1
    material_newton_tangent_evaluations = 0
    material_line_search_evaluations = 0
    material_rejected_line_search_evaluations = 0
    material_accepted_line_search_evaluations = 0
    material_verification_evaluations = 0
    material_complete_promotion_evaluations = 0
    material_seconds_accepted_attempts = 0.0
    material_seconds_rejected_attempts = 0.0
    line_search_full_step_acceptances = 0
    line_search_reduced_step_acceptances = 0
    line_search_rejections = 0
    line_search_minimum_factor = 1.0

    def evaluate_samples_timed(
        strain: FloatArray,
        dt: float,
        response_level: ResponseLevel = "tangent",
        evaluation_kind: EvaluationKind = "newton_tangent",
    ) -> TraditionalTwoStateTrial:
        nonlocal material_seconds, material_evaluations
        nonlocal material_newton_tangent_evaluations
        nonlocal material_line_search_evaluations, material_verification_evaluations
        started = time.perf_counter()
        result = elements.evaluate_samples(
            strain,
            time_increment=dt,
            response_level=response_level,
        )
        material_seconds += time.perf_counter() - started
        material_evaluations += 1
        if evaluation_kind == "newton_tangent":
            material_newton_tangent_evaluations += 1
        elif evaluation_kind == "line_search":
            material_line_search_evaluations += 1
        else:
            material_verification_evaluations += 1
        return result

    def strain_timed(displacement: FloatArray) -> FloatArray:
        nonlocal gradient_seconds
        started = time.perf_counter()
        result = kinematics.strain_samples(displacement)
        gradient_seconds += time.perf_counter() - started
        return result

    def divergence_timed(stress: FloatArray) -> FloatArray:
        nonlocal divergence_seconds
        started = time.perf_counter()
        kinematics.divergence_from_sample_stress_into(stress, nonlinear_divergence_buffer)
        divergence_seconds += time.perf_counter() - started
        return nonlinear_divergence_buffer

    plan = transform_plan or create_full_dirichlet_dsti_plan(grid, config.transform)
    if config.reference_parameter_mode == "explicit":
        assert config.reference_lambda_0 is not None
        assert config.reference_mu_0 is not None
        lambda_0 = config.reference_lambda_0
        mu_0 = config.reference_mu_0
        projected_lambda = lambda_0
        projected_mu = mu_0
        projection_error = 0.0
    else:
        initial_material_started = time.perf_counter()
        zero_trial = evaluate_in_plane_response(
            material,
            np.zeros((material.point_count, 3)),
            time_increment=1.0,
            response_level="tangent",
            consistent_tangent=True,
        )
        material_seconds += time.perf_counter() - initial_material_started
        material.revert()
        tangent = np.asarray(zero_trial.tangent_in_plane_mpa).reshape(
            *grid.pixel_shape, 2, 3, 3
        )
        projected_lambda, projected_mu, projection_error = project_isotropic_plane_stress_tangent(
            tangent.mean(axis=(0, 1, 2))
        )
        lambda_0 = (
            projected_lambda
            * config.reference_parameter_scale
            * config.reference_lambda_mu_ratio
        )
        mu_0 = projected_mu * config.reference_parameter_scale
    green = B0Green2D(
        kinematics.reference_operator_symbols(plan),
        lambda_0=lambda_0,
        mu_0=mu_0,
        symbol_null_tolerance=config.symbol_null_tolerance,
    )
    interior_shape = (*grid.interior_shape, 2)
    spectral_buffer = np.empty(interior_shape, dtype=np.float64)
    green_buffer = np.empty_like(spectral_buffer)
    physical_buffer = np.empty_like(spectral_buffer)
    nonlinear_divergence_buffer = np.empty((*grid.node_shape, 2), dtype=np.float64)
    jacobian_workspace = TwoStateJacobianWorkspace.create(grid)
    fluctuation = np.zeros((*grid.node_shape, 2))
    initial_guess_applied = False
    residual_history: list[float] = []
    absolute_history: list[float] = []
    verification_history: list[float] = []
    verification_mismatch_history: list[float] = []
    iterations_per_increment: list[int] = []
    reference_updates: list[dict[str, str | int | float | bool]] = []
    gmres_iterations = 0
    linear_solves: list[LinearSolveDiagnostics] = []
    load_step_attempts: list[LoadStepAttemptDiagnostics] = []
    jacobian_totals = JacobianActionDiagnostics()
    preconditioner_totals = PreconditionerActionDiagnostics()
    final_trial = None
    final_sample_strain = None
    verification_residual = 0.0
    final_applied = history[0].copy()
    time_increment = (
        1.0 / (history.shape[0] - 1)
        if time_increment_override is None
        else float(time_increment_override)
    )
    if not np.isfinite(time_increment) or time_increment <= 0.0:
        raise ValueError("time increment must be finite and positive")
    krylov_recycle = KrylovRecycleState()
    adaptive_step_history: list[dict[str, object]] = []
    previous_adaptive_slip: FloatArray | None = None
    previous_adaptive_slip_increment: FloatArray | None = None
    previous_adaptive_step_size: float | None = None

    def _emit_progress(config_for_event: EBISpectralSolverConfig, event: dict[str, object]) -> None:
        native_timing = getattr(material, "timing_statistics", None)
        payload = {
            **event,
            "material_seconds": material_seconds,
            "material_seconds_accepted_attempts": material_seconds_accepted_attempts,
            "material_seconds_rejected_attempts": material_seconds_rejected_attempts,
            "gradient_seconds": gradient_seconds,
            "divergence_seconds": divergence_seconds,
            "gmres_seconds": gmres_seconds,
            "jacobian_seconds": jacobian_totals.total_seconds,
            "preconditioner_seconds": preconditioner_totals.total_seconds,
            "jacobian_calls": jacobian_totals.calls,
            "preconditioner_calls": preconditioner_totals.calls,
            "material_evaluations": material_evaluations,
            "gmres_iterations": gmres_iterations,
            "material_local_condensation_evaluations": int(
                getattr(native_timing, "local_condensation_evaluations", 0)
            ),
            "material_full_batch_integration_calls": int(
                getattr(native_timing, "full_batch_integration_calls", 0)
            ),
            "material_equivalent_active_point_integrations": int(
                getattr(native_timing, "equivalent_active_point_integrations", 0)
            ),
            "material_local_iteration_histogram": list(
                getattr(native_timing, "local_iteration_histogram", ())
            ),
            "krylov_overhead_seconds": max(
                0.0,
                gmres_seconds
                - jacobian_totals.total_seconds
                - preconditioner_totals.total_seconds,
            ),
        }
        for name in (
            "rotation_to_material_seconds",
            "integration_seconds",
            "rotation_to_global_seconds",
            "condensation_seconds",
            "condition_check_seconds",
            "local_solve_seconds",
            "reconstruction_seconds",
            "observable_seconds",
        ):
            payload[f"material_{name}"] = float(
                getattr(native_timing, name, 0.0)
            )
        _emit_progress_raw(config_for_event, payload)

    def fixed_load_path() -> list[LoadPathStep]:
        return [
            LoadPathStep(
                index=index,
                start_fraction=(index - 1) * time_increment,
                end_fraction=index * time_increment,
                boundary=history[index].copy(),
                time_increment=time_increment,
            )
            for index in range(1, history.shape[0])
        ]

    adaptive_path = (
        AdaptiveLoadPath(history, config.adaptive_step)
        if config.adaptive_stepping_enabled
        else None
    )
    if load_path_override is not None:
        load_path: AdaptiveLoadPath | list[LoadPathStep] = list(load_path_override)
        if not load_path:
            raise ValueError("load_path_override must contain at least one step")
    else:
        load_path = adaptive_path if adaptive_path is not None else fixed_load_path()
    for path_item in load_path:
        increment = path_item.index
        boundary_state = path_item.boundary
        time_increment = path_item.time_increment
        material_seconds_at_attempt_start = material_seconds
        material_evaluations_at_attempt_start = material_evaluations
        gmres_seconds_at_attempt_start = gmres_seconds
        gmres_iterations_at_attempt_start = gmres_iterations
        jacobian_calls_at_attempt_start = jacobian_totals.calls
        jacobian_seconds_at_attempt_start = jacobian_totals.total_seconds
        preconditioner_calls_at_attempt_start = preconditioner_totals.calls
        preconditioner_seconds_at_attempt_start = preconditioner_totals.total_seconds
        line_search_rejections_at_attempt_start = line_search_rejections
        linear_solves_at_attempt_start = len(linear_solves)
        native_timing_at_attempt_start = getattr(material, "timing_statistics", None)
        native_integration_seconds_at_attempt_start = float(
            getattr(native_timing_at_attempt_start, "integration_seconds", 0.0)
        )
        native_condensation_seconds_at_attempt_start = float(
            getattr(native_timing_at_attempt_start, "condensation_seconds", 0.0)
        )
        native_mgis_integrations_at_attempt_start = int(
            getattr(native_timing_at_attempt_start, "full_batch_integration_calls", 0)
        )
        _emit_progress(
            config,
            {
                "event": "increment_started",
                "increment": increment,
                "load_fraction_start": path_item.start_fraction,
                "load_fraction_end": path_item.end_fraction,
                "time_increment": time_increment,
            },
        )
        krylov_recycle.reset()
        applied = extension.extend(boundary_state, grid)
        if initial_guess_callback is not None:
            callback_guess = np.asarray(
                initial_guess_callback(path_item, applied + fluctuation), dtype=np.float64
            )
            if callback_guess.shape != (*grid.node_shape, 2):
                raise ValueError(
                    "initial_guess_callback must return nodal displacement shape "
                    f"{(*grid.node_shape, 2)}"
                )
            fluctuation[...] = callback_guess - applied
            initial_guess_applied = True
        elif not initial_guess_applied and initial_displacement_array is not None:
            # ``fluctuation`` is an interior unknown.  Boundary values are
            # supplied by ``applied`` at every Newton iteration and must not
            # be cancelled by copying a full-field initial displacement.  The
            # previous full-array assignment made a non-zero first load step
            # appear one increment late in recorded histories.
            fluctuation[...] = 0.0
            fluctuation[1:-1, 1:-1] = (
                initial_displacement_array[1:-1, 1:-1] - applied[1:-1, 1:-1]
            )
            initial_guess_applied = True
        increment_start_fluctuation = fluctuation.copy()
        converged = False
        increment_failure_reason = ""
        increment_min_line_search_factor = 1.0
        attempt_newton_iterations = 0
        previous_nonlinear_residual: float | None = None
        force_fixed_linear_tolerance = False
        accepted_cache = AcceptedTwoStateTrialCache()
        for iteration in range(config.maximum_newton_iterations):
            attempt_newton_iterations = iteration + 1
            used_cached_trial = accepted_cache.populated
            if not accepted_cache.populated:
                sample_strain = strain_timed(applied + fluctuation)
                trial = evaluate_samples_timed(
                    sample_strain,
                    time_increment,
                    evaluation_kind="newton_tangent",
                )
                residual = divergence_timed(trial.sample_stress_mpa)
                relative, absolute, _ = _equilibrium_metrics(
                    trial.sample_stress_mpa, residual, grid, 2
                )
            else:
                trial, sample_strain, residual, relative, absolute = accepted_cache.take()
            residual_history.append(relative)
            absolute_history.append(absolute)
            _emit_progress(
                config,
                {
                    "event": "newton_residual",
                    "increment": increment,
                    "newton_iteration": iteration + 1,
                    "relative_residual": relative,
                    "absolute_residual": absolute,
                    "used_cached_trial": used_cached_trial,
                },
            )
            if relative <= config.relative_equilibrium_tolerance:
                solver_residual = relative
                if not config.verify_final_state:
                    verification_residual = relative
                    final_trial = elements.complete_trial(trial)
                    elements.commit()
                    final_sample_strain = sample_strain.copy()
                    final_applied = applied.copy()
                    iterations_per_increment.append(iteration + 1)
                    converged = True
                    _emit_progress(
                        config,
                        {
                            "event": "increment_converged",
                            "increment": increment,
                            "newton_iterations": iteration + 1,
                            "relative_residual": relative,
                        },
                    )
                    break
                elements.revert()
                verification_trial = evaluate_samples_timed(
                    sample_strain,
                    time_increment,
                    response_level="complete",
                    evaluation_kind="verification",
                )
                verification_force = divergence_timed(verification_trial.sample_stress_mpa)
                verification_residual = _equilibrium_metrics(
                    verification_trial.sample_stress_mpa,
                    verification_force,
                    grid,
                    2,
                )[0]
                verification_mismatch = abs(verification_residual - solver_residual) / max(
                    solver_residual, 1.0e-30
                )
                verification_history.append(verification_residual)
                verification_mismatch_history.append(verification_mismatch)
                if (
                    verification_residual <= config.relative_equilibrium_tolerance
                    and verification_mismatch <= 1.0e-3
                ):
                    final_trial = elements.complete_trial(verification_trial)
                    elements.commit()
                    final_sample_strain = sample_strain.copy()
                    final_applied = applied.copy()
                    iterations_per_increment.append(iteration + 1)
                    converged = True
                    _emit_progress(
                        config,
                        {
                            "event": "increment_converged",
                            "increment": increment,
                            "newton_iterations": iteration + 1,
                            "relative_residual": verification_residual,
                        },
                    )
                    break
                trial = verification_trial
                residual = verification_force
                relative = verification_residual

            if config.reference_update_mode != "initial":
                if config.reference_parameter_mode == "explicit":
                    reference_updates.append(
                        {
                            "increment": increment,
                            "newton_iteration": iteration + 1,
                            "accepted": False,
                            "reason": "explicit_reference_parameters",
                        }
                    )
                elif config.reference_update_mode == "per_newton" or iteration == 0:
                    if trial.algorithmic_tangent_in_plane_mpa is None:
                        raise RuntimeError("reference update requires a tangent trial")
                    candidate_lambda, candidate_mu, _ = project_isotropic_plane_stress_tangent(
                        trial.algorithmic_tangent_in_plane_mpa.mean(axis=(0, 1, 2))
                    )
                    candidate_lambda *= config.reference_parameter_scale
                    candidate_mu *= config.reference_parameter_scale
                    relaxation = config.reference_update_relaxation
                    updated_lambda = (1.0 - relaxation) * lambda_0 + relaxation * candidate_lambda
                    updated_mu = (1.0 - relaxation) * mu_0 + relaxation * candidate_mu
                    mu_ratio = updated_mu / mu_0
                    bulk_ratio = (updated_lambda + updated_mu) / (lambda_0 + mu_0)
                    relative_change = max(
                        abs(updated_mu - mu_0) / max(abs(mu_0), 1.0e-30),
                        abs(updated_lambda - lambda_0) / max(abs(lambda_0), 1.0e-30),
                    )
                    accepted_update = (
                        np.isfinite(updated_lambda)
                        and np.isfinite(updated_mu)
                        and 0.25 <= mu_ratio <= 4.0
                        and 0.25 <= bulk_ratio <= 4.0
                        and relative_change >= config.reference_minimum_relative_change
                    )
                    if accepted_update:
                        green.update_parameters(
                            lambda_0=updated_lambda,
                            mu_0=updated_mu,
                        )
                        lambda_0 = updated_lambda
                        mu_0 = updated_mu
                        reason = "updated"
                    elif relative_change < config.reference_minimum_relative_change:
                        reason = "below_minimum_change"
                    else:
                        reason = "safeguard_rejected"
                    reference_updates.append(
                        {
                            "increment": increment,
                            "newton_iteration": iteration + 1,
                            "accepted": accepted_update,
                            "reason": reason,
                            "lambda_0": lambda_0,
                            "mu_0": mu_0,
                            "candidate_lambda": candidate_lambda,
                            "candidate_mu": candidate_mu,
                        }
                    )

            size = 2 * (grid.nx - 1) * (grid.ny - 1)
            jacobian_local = JacobianActionDiagnostics()
            preconditioner_local = PreconditionerActionDiagnostics()
            gmres_iterations_before = gmres_iterations
            requested_linear_tolerance = _linear_tolerance(
                config,
                relative,
                previous_nonlinear_residual,
                force_fixed=force_fixed_linear_tolerance,
            )
            force_fixed_linear_tolerance = False

            def jacobian_action(
                vector: FloatArray,
                active_trial=trial,
                action_counters=jacobian_local,
            ) -> FloatArray:
                started = time.perf_counter()
                action_counters.calls += 1
                unpack_started = time.perf_counter()
                unpack_interior_into(vector, grid, jacobian_workspace.nodal_increment)
                action_counters.unpack_seconds += time.perf_counter() - unpack_started
                result = elements.tangent_action_into(
                    kinematics=kinematics,
                    trial=active_trial,
                    workspace=jacobian_workspace,
                    action_diagnostics=action_counters,
                )
                action_counters.total_seconds += time.perf_counter() - started
                return result

            def preconditioner_action(
                vector: FloatArray,
                action_counters=preconditioner_local,
            ) -> FloatArray:
                started = time.perf_counter()
                action_counters.calls += 1
                reshape_started = time.perf_counter()
                interior = np.asarray(vector, dtype=np.float64).reshape(interior_shape)
                action_counters.reshape_seconds += time.perf_counter() - reshape_started
                forward_started = time.perf_counter()
                if hasattr(plan, "forward_into"):
                    buffered_plan = cast(BufferedTransformPlan2D, plan)
                    buffered_plan.forward_into(interior, spectral_buffer)
                else:
                    spectral_buffer[...] = plan.forward_displacement(interior)
                action_counters.forward_transform_seconds += (
                    time.perf_counter() - forward_started
                )
                green_started = time.perf_counter()
                green.apply_into(spectral_buffer, green_buffer)
                action_counters.green_seconds += time.perf_counter() - green_started
                inverse_started = time.perf_counter()
                if hasattr(plan, "forward_into"):
                    buffered_plan = cast(BufferedTransformPlan2D, plan)
                    buffered_plan.inverse_into(green_buffer, physical_buffer)
                else:
                    physical_buffer[...] = plan.inverse_displacement(green_buffer)
                action_counters.inverse_transform_seconds += (
                    time.perf_counter() - inverse_started
                )
                copy_started = time.perf_counter()
                result = physical_buffer.reshape(-1).copy()
                action_counters.output_copy_seconds += time.perf_counter() - copy_started
                action_counters.total_seconds += time.perf_counter() - started
                return result

            gmres_matrix = LinearOperator((size, size), matvec=jacobian_action, dtype=float)
            preconditioner = LinearOperator((size, size), matvec=preconditioner_action, dtype=float)

            def count_gmres(_residual: object) -> None:
                nonlocal gmres_iterations
                gmres_iterations += 1

            gmres_started = time.perf_counter()
            rhs = -pack_interior(residual)
            correction, info, _krylov_calls = solve_nonsymmetric_krylov(
                gmres_matrix,
                rhs,
                preconditioner=preconditioner,
                method=config.krylov_method,
                rtol=requested_linear_tolerance,
                maximum_iterations=config.gmres_maximum_iterations,
                restart=config.gmres_restart,
                recycle=krylov_recycle if config.krylov_recycling else None,
                lgmres_inner_m=config.lgmres_inner_m,
                lgmres_outer_k=config.lgmres_outer_k,
                gcrotmk_m=config.gcrotmk_m,
                gcrotmk_k=config.gcrotmk_k,
                callback=count_gmres,
                blas_threads=config.krylov_blas_threads,
            )
            linear_residual_ratio: float | None = None
            if config.verify_linear_residual and info == 0 and np.isfinite(correction).all():
                linear_residual = jacobian_action(correction) - rhs
                linear_residual_ratio = float(
                    np.linalg.norm(linear_residual)
                    / max(np.linalg.norm(rhs), 1.0e-30)
                )
                force_fixed_linear_tolerance = linear_residual_ratio > (
                    1.5 * requested_linear_tolerance
                )
            gmres_elapsed = time.perf_counter() - gmres_started
            gmres_seconds += gmres_elapsed
            linear_solve = LinearSolveDiagnostics(
                increment=increment,
                newton_iteration=iteration + 1,
                nonlinear_residual_before=relative,
                requested_relative_tolerance=requested_linear_tolerance,
                gmres_info=int(info),
                gmres_iterations=gmres_iterations - gmres_iterations_before,
                jacobian_calls=jacobian_local.calls,
                preconditioner_calls=preconditioner_local.calls,
                gmres_seconds=gmres_elapsed,
                jacobian_seconds=jacobian_local.total_seconds,
                preconditioner_seconds=preconditioner_local.total_seconds,
                krylov_overhead_seconds=max(
                    0.0,
                    gmres_elapsed
                    - jacobian_local.total_seconds
                    - preconditioner_local.total_seconds,
                ),
                restart=config.gmres_restart,
                line_search_factor=None,
                linear_residual_ratio=linear_residual_ratio,
                krylov_method=config.krylov_method,
                krylov_recycling=config.krylov_recycling,
            )

            if info != 0 or not np.isfinite(correction).all():
                _accumulate_jacobian(jacobian_totals, jacobian_local)
                _accumulate_preconditioner(preconditioner_totals, preconditioner_local)
                linear_solves.append(linear_solve)
                elements.revert()
                increment_failure_reason = f"gmres_failure:{info}"
                break
            direction = unpack_interior(correction, grid)
            accepted = False
            factor = 1.0
            for _ in range(config.maximum_line_search_reductions + 1):
                candidate = fluctuation + factor * direction
                candidate_strain = strain_timed(applied + candidate)
                candidate_trial = evaluate_samples_timed(
                    candidate_strain,
                    time_increment,
                    response_level=(
                        "tangent" if config.verify_final_state else "complete"
                    ),
                    evaluation_kind="line_search",
                )
                candidate_residual = divergence_timed(candidate_trial.sample_stress_mpa)
                candidate_relative, candidate_absolute, _ = _equilibrium_metrics(
                    candidate_trial.sample_stress_mpa, candidate_residual, grid, 2
                )
                if candidate_relative < relative:
                    material_accepted_line_search_evaluations += 1
                    if factor == 1.0:
                        line_search_full_step_acceptances += 1
                    else:
                        line_search_reduced_step_acceptances += 1
                    line_search_minimum_factor = min(line_search_minimum_factor, factor)
                    increment_min_line_search_factor = min(
                        increment_min_line_search_factor, factor
                    )
                    fluctuation = candidate
                    accepted_cache.store(
                        trial=candidate_trial,
                        sample_strain=candidate_strain,
                        residual=candidate_residual,
                        relative=candidate_relative,
                        absolute=candidate_absolute,
                    )
                    accept_global_trial = getattr(elements, "accept_global_trial", None)
                    if callable(accept_global_trial):
                        accept_global_trial()
                    accepted = True
                    break
                material_rejected_line_search_evaluations += 1
                line_search_rejections += 1
                factor *= 0.5
            if not accepted:
                _accumulate_jacobian(jacobian_totals, jacobian_local)
                _accumulate_preconditioner(preconditioner_totals, preconditioner_local)
                linear_solves.append(linear_solve)
                elements.revert()
                increment_failure_reason = "line_search_failure"
                _emit_progress(
                    config,
                    {
                        "event": "increment_failed",
                        "increment": increment,
                        "newton_iteration": iteration + 1,
                        "reason": increment_failure_reason,
                    },
                )
                break
            _accumulate_jacobian(jacobian_totals, jacobian_local)
            _accumulate_preconditioner(preconditioner_totals, preconditioner_local)
            linear_solves.append(replace(linear_solve, line_search_factor=factor))
            previous_nonlinear_residual = relative
            _emit_progress(
                config,
                {
                    "event": "newton_step_accepted",
                    "increment": increment,
                    "newton_iteration": iteration + 1,
                    "relative_residual": relative,
                    "line_search_factor": factor,
                    "krylov_outer_callbacks": (
                        gmres_iterations - gmres_iterations_before
                    ),
                    "krylov_seconds": gmres_elapsed,
                    "jacobian_seconds": jacobian_local.total_seconds,
                    "preconditioner_seconds": preconditioner_local.total_seconds,
                    "krylov_overhead_seconds": linear_solve.krylov_overhead_seconds,
                    "jacobian_calls": jacobian_local.calls,
                    "preconditioner_calls": preconditioner_local.calls,
                },
            )
            if factor < 1.0:
                force_fixed_linear_tolerance = True
        if converged and increment_observer is not None:
            assert final_trial is not None
            if final_trial.tangent_in_plane_mpa is None:
                raise RuntimeError("accepted increment has no consistent tangent")
            increment_observer(
                TwoStateIncrementFields(
                    increment=increment,
                    start_fraction=path_item.start_fraction,
                    end_fraction=path_item.end_fraction,
                    time_increment=time_increment,
                    boundary=np.asarray(boundary_state).copy(),
                    displacement=final_applied + fluctuation,
                    sample_strain=np.asarray(final_sample_strain).copy(),
                    stress_in_plane_mpa=_reshape_two_state(
                        final_trial.stress_in_plane_mpa, grid
                    ),
                    algorithmic_tangent_in_plane_mpa=_reshape_two_state(
                        final_trial.tangent_in_plane_mpa, grid
                    ),
                    plastic_strain_tensor=_reshape_two_state(
                        final_trial.plastic_strain_tensor, grid
                    ),
                    elastic_strain_tensor=_reshape_two_state(
                        final_trial.elastic_strain_tensor, grid
                    ),
                    observables={
                        name: _reshape_two_state(values, grid)
                        for name, values in final_trial.observables.items()
                    },
                )
            )
        if not converged:
            elements.revert()
            native_timing = getattr(material, "timing_statistics", None)
            attempt_krylov_seconds = gmres_seconds - gmres_seconds_at_attempt_start
            attempt_jacobian_seconds = (
                jacobian_totals.total_seconds - jacobian_seconds_at_attempt_start
            )
            attempt_preconditioner_seconds = (
                preconditioner_totals.total_seconds
                - preconditioner_seconds_at_attempt_start
            )
            load_step_attempts.append(
                LoadStepAttemptDiagnostics(
                    attempt_index=increment,
                    load_fraction_start=path_item.start_fraction,
                    load_fraction_end=path_item.end_fraction,
                    accepted=False,
                    failure_reason=increment_failure_reason or "newton_iteration_limit",
                    newton_iterations=attempt_newton_iterations,
                    linear_solves=len(linear_solves) - linear_solves_at_attempt_start,
                    krylov_outer_callbacks=(
                        gmres_iterations - gmres_iterations_at_attempt_start
                    ),
                    jacobian_matvec_calls=(
                        jacobian_totals.calls - jacobian_calls_at_attempt_start
                    ),
                    preconditioner_calls=(
                        preconditioner_totals.calls
                        - preconditioner_calls_at_attempt_start
                    ),
                    krylov_seconds=attempt_krylov_seconds,
                    jacobian_seconds=attempt_jacobian_seconds,
                    preconditioner_seconds=attempt_preconditioner_seconds,
                    krylov_overhead_seconds=max(
                        0.0,
                        attempt_krylov_seconds
                        - attempt_jacobian_seconds
                        - attempt_preconditioner_seconds,
                    ),
                    material_seconds=(
                        material_seconds - material_seconds_at_attempt_start
                    ),
                    material_evaluations=(
                        material_evaluations - material_evaluations_at_attempt_start
                    ),
                    material_integration_seconds=float(
                        getattr(native_timing, "integration_seconds", 0.0)
                    )
                    - native_integration_seconds_at_attempt_start,
                    material_condensation_seconds=float(
                        getattr(native_timing, "condensation_seconds", 0.0)
                    )
                    - native_condensation_seconds_at_attempt_start,
                    mgis_integrations=int(
                        getattr(native_timing, "full_batch_integration_calls", 0)
                    )
                    - native_mgis_integrations_at_attempt_start,
                    line_search_rejections=(
                        line_search_rejections
                        - line_search_rejections_at_attempt_start
                    ),
                    minimum_line_search_factor=increment_min_line_search_factor,
                )
            )
            if adaptive_path is not None:
                fluctuation[...] = increment_start_fluctuation
                adaptive_step_history.append(
                    {
                        "attempted_step": increment,
                        "accepted": False,
                        "load_fraction_start": adaptive_path.current_fraction,
                        "load_fraction_end": (
                            adaptive_path.pending.end_fraction
                            if adaptive_path.pending is not None
                            else adaptive_path.current_fraction
                        ),
                        "step_size": time_increment,
                        "newton_iterations": attempt_newton_iterations,
                        "minimum_line_search_factor": increment_min_line_search_factor,
                        "next_step_reason": increment_failure_reason
                        or "newton_iteration_limit",
                    }
                )
                adaptive_path.reject(
                    increment_failure_reason or "newton_iteration_limit"
                )
                _emit_progress(
                    config,
                    {
                        "event": "increment_rejected",
                        "increment": increment,
                        "reason": increment_failure_reason or "newton_iteration_limit",
                        "newton_iterations": attempt_newton_iterations,
                        "attempt_cost": {
                            "krylov_outer_callbacks": (
                                load_step_attempts[-1].krylov_outer_callbacks
                            ),
                            "jacobian_matvec_calls": (
                                load_step_attempts[-1].jacobian_matvec_calls
                            ),
                            "preconditioner_calls": (
                                load_step_attempts[-1].preconditioner_calls
                            ),
                            "krylov_seconds": load_step_attempts[-1].krylov_seconds,
                            "jacobian_seconds": load_step_attempts[-1].jacobian_seconds,
                            "preconditioner_seconds": (
                                load_step_attempts[-1].preconditioner_seconds
                            ),
                            "krylov_overhead_seconds": (
                                load_step_attempts[-1].krylov_overhead_seconds
                            ),
                            "material_seconds": load_step_attempts[-1].material_seconds,
                        },
                    },
                )
                material_seconds_rejected_attempts += (
                    material_seconds - material_seconds_at_attempt_start
                )
                continue
            raise RuntimeError(f"two-state increment {increment} did not converge")
        material_seconds_accepted_attempts += (
            material_seconds - material_seconds_at_attempt_start
        )
        native_timing = getattr(material, "timing_statistics", None)
        attempt_krylov_seconds = gmres_seconds - gmres_seconds_at_attempt_start
        attempt_jacobian_seconds = (
            jacobian_totals.total_seconds - jacobian_seconds_at_attempt_start
        )
        attempt_preconditioner_seconds = (
            preconditioner_totals.total_seconds - preconditioner_seconds_at_attempt_start
        )
        load_step_attempts.append(
            LoadStepAttemptDiagnostics(
                attempt_index=increment,
                load_fraction_start=path_item.start_fraction,
                load_fraction_end=path_item.end_fraction,
                accepted=True,
                failure_reason=None,
                newton_iterations=attempt_newton_iterations,
                linear_solves=len(linear_solves) - linear_solves_at_attempt_start,
                krylov_outer_callbacks=gmres_iterations - gmres_iterations_at_attempt_start,
                jacobian_matvec_calls=(
                    jacobian_totals.calls - jacobian_calls_at_attempt_start
                ),
                preconditioner_calls=(
                    preconditioner_totals.calls - preconditioner_calls_at_attempt_start
                ),
                krylov_seconds=attempt_krylov_seconds,
                jacobian_seconds=attempt_jacobian_seconds,
                preconditioner_seconds=attempt_preconditioner_seconds,
                krylov_overhead_seconds=max(
                    0.0,
                    attempt_krylov_seconds
                    - attempt_jacobian_seconds
                    - attempt_preconditioner_seconds,
                ),
                material_seconds=material_seconds - material_seconds_at_attempt_start,
                material_evaluations=(
                    material_evaluations - material_evaluations_at_attempt_start
                ),
                material_integration_seconds=float(
                    getattr(native_timing, "integration_seconds", 0.0)
                )
                - native_integration_seconds_at_attempt_start,
                material_condensation_seconds=float(
                    getattr(native_timing, "condensation_seconds", 0.0)
                )
                - native_condensation_seconds_at_attempt_start,
                mgis_integrations=int(
                    getattr(native_timing, "full_batch_integration_calls", 0)
                )
                - native_mgis_integrations_at_attempt_start,
                line_search_rejections=(
                    line_search_rejections - line_search_rejections_at_attempt_start
                ),
                minimum_line_search_factor=increment_min_line_search_factor,
            )
        )
        _emit_progress(
            config,
            {
                "event": "attempt_cost_completed",
                "increment": increment,
                "accepted": True,
                "krylov_outer_callbacks": load_step_attempts[-1].krylov_outer_callbacks,
                "jacobian_matvec_calls": load_step_attempts[-1].jacobian_matvec_calls,
                "preconditioner_calls": load_step_attempts[-1].preconditioner_calls,
                "krylov_seconds": load_step_attempts[-1].krylov_seconds,
                "jacobian_seconds": load_step_attempts[-1].jacobian_seconds,
                "preconditioner_seconds": (
                    load_step_attempts[-1].preconditioner_seconds
                ),
                "krylov_overhead_seconds": (
                    load_step_attempts[-1].krylov_overhead_seconds
                ),
                "material_seconds": load_step_attempts[-1].material_seconds,
            },
        )
        if adaptive_path is not None:
            slip_error_ratio: float | None = None
            if (
                config.adaptive_step.slip_error_control == "predictive"
                and final_trial is not None
                and "equivalent_plastic_slip" in final_trial.observables
            ):
                current_slip = np.asarray(
                    final_trial.observables["equivalent_plastic_slip"],
                    dtype=np.float64,
                ).copy()
                if (
                    previous_adaptive_slip is not None
                    and previous_adaptive_slip_increment is not None
                    and previous_adaptive_step_size is not None
                ):
                    slip_error_ratio = predictive_slip_error_ratio(
                        current_slip,
                        previous_adaptive_slip,
                        previous_adaptive_slip_increment,
                        current_step_size=time_increment,
                        previous_step_size=previous_adaptive_step_size,
                        relative_tolerance=(
                            config.adaptive_step.slip_error_relative_tolerance
                        ),
                        absolute_tolerance=(
                            config.adaptive_step.slip_error_absolute_tolerance
                        ),
                    )
                if previous_adaptive_slip is not None:
                    previous_adaptive_slip_increment = (
                        current_slip - previous_adaptive_slip
                    )
                previous_adaptive_slip = current_slip
                previous_adaptive_step_size = time_increment
            decision = adaptive_path.accept(
                LoadStepObservation(
                    converged=True,
                    newton_iterations=iterations_per_increment[-1],
                    minimum_line_search_factor=increment_min_line_search_factor,
                    slip_error_ratio=slip_error_ratio,
                )
            )
            adaptive_step_history.append(
                {
                    "attempted_step": increment,
                    "accepted": True,
                    "load_fraction_start": adaptive_path.current_fraction
                    - time_increment,
                    "load_fraction_end": adaptive_path.current_fraction,
                    "step_size": time_increment,
                    "newton_iterations": iterations_per_increment[-1],
                    "minimum_line_search_factor": increment_min_line_search_factor,
                    "slip_error_ratio": slip_error_ratio,
                    "next_step_size": decision.next_increment_fraction,
                    "next_step_reason": decision.reason,
                }
            )
            _emit_progress(
                config,
                {
                    "event": "increment_accepted",
                    "increment": increment,
                    "load_fraction_start": adaptive_path.current_fraction - time_increment,
                    "load_fraction_end": adaptive_path.current_fraction,
                    "newton_iterations": iterations_per_increment[-1],
                    "next_step_size": decision.next_increment_fraction,
                    "next_step_reason": decision.reason,
                    "attempt_cost": {
                        "krylov_outer_callbacks": (
                            load_step_attempts[-1].krylov_outer_callbacks
                        ),
                        "jacobian_matvec_calls": (
                            load_step_attempts[-1].jacobian_matvec_calls
                        ),
                        "preconditioner_calls": (
                            load_step_attempts[-1].preconditioner_calls
                        ),
                        "krylov_seconds": load_step_attempts[-1].krylov_seconds,
                        "jacobian_seconds": load_step_attempts[-1].jacobian_seconds,
                        "preconditioner_seconds": (
                            load_step_attempts[-1].preconditioner_seconds
                        ),
                        "krylov_overhead_seconds": (
                            load_step_attempts[-1].krylov_overhead_seconds
                        ),
                        "material_seconds": load_step_attempts[-1].material_seconds,
                    },
                },
            )

    if final_trial is None or final_sample_strain is None:
        raise RuntimeError("no two-state increment converged")
    transform_diagnostics = plan.diagnostics
    material_timing = getattr(material, "timing_statistics", None)
    material_timing_values = {
        name: float(getattr(material_timing, name, 0.0))
        for name in (
            "rotation_to_material_seconds",
            "integration_seconds",
            "rotation_to_global_seconds",
            "condensation_seconds",
            "condition_check_seconds",
            "local_solve_seconds",
            "reconstruction_seconds",
            "observable_seconds",
            "local_condensation_evaluations",
            "full_batch_integration_calls",
            "equivalent_active_point_integrations",
            "material_point_integrations",
            "material_point_integrations_with_tangent",
            "material_point_integrations_without_tangent",
            "material_block_integration_calls",
            "material_block_count",
            "native_batch_calls",
            "native_material_points",
            "native_internal_integrations",
            "native_integrate_calls",
            "native_integrate_points",
            "native_total_local_iterations",
            "native_thread_count",
            "native_substep_points",
            "native_substep_cache_hits",
            "native_substep_cache_misses",
            "composite_fd_seconds",
            "composite_fd_points",
            "composite_fd_trajectories",
            "composite_fd_partition_changes",
            "composite_fd_mgis_calls",
            "composite_fd_actual_point_integrations",
            "composite_fd_snapshot_seconds",
            "composite_fd_restore_seconds",
            "composite_fd_integration_seconds",
            "composite_fd_other_seconds",
        )
    }
    provenance = collect_runtime_provenance(
        transform_diagnostics,
        gmres_restart=config.gmres_restart,
        gmres_maximum_iterations=config.gmres_maximum_iterations,
        gmres_relative_tolerance=config.gmres_relative_tolerance,
        linear_tolerance_mode=config.linear_tolerance_mode,
        forcing_initial=config.forcing_initial,
        forcing_minimum=config.forcing_minimum,
        forcing_maximum=config.forcing_maximum,
        forcing_gamma=config.forcing_gamma,
        forcing_alpha=config.forcing_alpha,
        krylov_method=config.krylov_method,
        krylov_recycling=config.krylov_recycling,
        lgmres_inner_m=config.lgmres_inner_m,
        lgmres_outer_k=config.lgmres_outer_k,
        gcrotmk_m=config.gcrotmk_m,
        gcrotmk_k=config.gcrotmk_k,
        reference_update_mode=config.reference_update_mode,
        krylov_blas_threads=config.krylov_blas_threads,
    )
    provenance.update(
        {
            "material_backend": getattr(material, "backend_name", type(material).__name__),
            "material_matrix_type": getattr(
                material, "linear_system_matrix_type", "unspecified"
            ),
            "mfront_threads": getattr(material, "thread_count", None),
            "local_condition_check_mode": getattr(
                material, "local_condition_check_mode", None
            ),
        }
    )
    observables = {
        name: _reshape_two_state(values, grid) for name, values in final_trial.observables.items()
    }
    diagnostics = Spectral2DDiagnostics(
        spatial_scheme="traditional_two_state_triangle",
        green_operator="b0",
        pixels=grid.pixel_shape,
        material_points=material.point_count,
        points_per_pixel=2,
        spacing_x=grid.spacing_x,
        spacing_y=grid.spacing_y,
        relative_residual_history=tuple(residual_history),
        dimensionless_equilibrium_history=tuple(residual_history),
        absolute_residual_history=tuple(absolute_history),
        iterations_per_increment=tuple(iterations_per_increment),
        reference_lambda_0=lambda_0,
        reference_mu_0=mu_0,
        reference_projection_error=projection_error,
        verification_residual=verification_residual,
        verification_residual_history=tuple(verification_history),
        verification_relative_mismatch_history=tuple(verification_mismatch_history),
        transform_backend=transform_diagnostics.backend,
        transform_implementation=transform_diagnostics.implementation,
        transform_interior_shape=transform_diagnostics.interior_shape,
        transform_batch_components=transform_diagnostics.batch_components,
        transform_dtype=transform_diagnostics.dtype,
        transform_workers=transform_diagnostics.workers,
        transform_planner_effort=transform_diagnostics.planner_effort,
        transform_wisdom_loaded=transform_diagnostics.wisdom_loaded,
        transform_planning_seconds=transform_diagnostics.planning_seconds,
        linear_solves=tuple(linear_solves),
        load_step_attempts=tuple(load_step_attempts),
        reference_updates=tuple(reference_updates),
        adaptive_stepping_enabled=adaptive_path is not None,
        adaptive_step_history=tuple(adaptive_step_history),
        provenance=provenance,
        timings={
            "material_evaluations": float(material_evaluations),
            "material_initial_probe_evaluations": float(
                material_initial_probe_evaluations
            ),
            "material_newton_tangent_evaluations": float(
                material_newton_tangent_evaluations
            ),
            "material_line_search_evaluations": float(material_line_search_evaluations),
            "material_rejected_line_search_evaluations": float(
                material_rejected_line_search_evaluations
            ),
            "material_accepted_line_search_evaluations": float(
                material_accepted_line_search_evaluations
            ),
            "material_verification_evaluations": float(
                material_verification_evaluations
            ),
            "material_complete_promotion_evaluations": float(
                material_complete_promotion_evaluations
            ),
            "material_total_evaluations": float(material_evaluations),
            "line_search_full_step_acceptances": float(line_search_full_step_acceptances),
            "line_search_reduced_step_acceptances": float(
                line_search_reduced_step_acceptances
            ),
            "line_search_rejections": float(line_search_rejections),
            "line_search_minimum_factor": line_search_minimum_factor,
            "material_seconds": material_seconds,
            "material_seconds_accepted_attempts": material_seconds_accepted_attempts,
            "material_seconds_rejected_attempts": material_seconds_rejected_attempts,
            "gradient_seconds": gradient_seconds,
            "divergence_seconds": divergence_seconds,
            "jacobian_seconds": jacobian_totals.total_seconds,
            "preconditioner_seconds": preconditioner_totals.total_seconds,
            "gmres_iterations": float(gmres_iterations),
            "krylov_outer_callbacks": float(gmres_iterations),
            "gmres_seconds": gmres_seconds,
            "jacobian_calls": float(jacobian_totals.calls),
            "jacobian_matvec_calls": float(jacobian_totals.calls),
            "jacobian_unpack_seconds": jacobian_totals.unpack_seconds,
            "jacobian_gradient_seconds": jacobian_totals.gradient_seconds,
            "jacobian_tangent_seconds": jacobian_totals.tangent_seconds,
            "jacobian_divergence_seconds": jacobian_totals.divergence_seconds,
            "jacobian_pack_seconds": jacobian_totals.pack_seconds,
            "preconditioner_calls": float(preconditioner_totals.calls),
            "preconditioner_reshape_seconds": preconditioner_totals.reshape_seconds,
            "preconditioner_forward_transform_seconds": (
                preconditioner_totals.forward_transform_seconds
            ),
            "preconditioner_green_seconds": preconditioner_totals.green_seconds,
            "preconditioner_inverse_transform_seconds": (
                preconditioner_totals.inverse_transform_seconds
            ),
            "preconditioner_output_copy_seconds": preconditioner_totals.output_copy_seconds,
            "krylov_overhead_seconds": max(
                0.0,
                gmres_seconds
                - jacobian_totals.total_seconds
                - preconditioner_totals.total_seconds,
            ),
            **{
                f"attempts_{status}_{name}": float(value)
                for status, values in summarize_load_step_attempts(
                    load_step_attempts
                ).items()
                for name, value in values.items()
            },
            **{
                (name if name.startswith("material_") else f"material_{name}"): value
                for name, value in material_timing_values.items()
            },
            "material_condition_checks": float(
                getattr(material_timing, "condition_checks", 0)
            ),
            "material_evaluate_calls": float(
                getattr(material_timing, "evaluate_calls", 0)
            ),
            "mgis_integrations": float(
                getattr(material_timing, "full_batch_integration_calls", 0)
            ),
            "material_warm_start_uses": float(
                getattr(material, "warm_start_uses", 0)
            ),
            "material_warm_start_resets": float(
                getattr(material, "warm_start_resets", 0)
            ),
        },
        material_local_iteration_histogram=tuple(
            getattr(material_timing, "local_iteration_histogram", ())
        ),
    )
    sample_stress = _reshape_two_state(final_trial.stress_in_plane_mpa, grid)
    return Spectral2DResult(
        displacement=final_applied + fluctuation,
        applied_displacement=final_applied,
        fluctuation_displacement=fluctuation.copy(),
        strain_in_plane=final_sample_strain,
        stress_in_plane_mpa=sample_stress,
        full_stress_tensor_mpa=_reshape_two_state(final_trial.full_stress_tensor_mpa, grid),
        full_strain_tensor=_reshape_two_state(final_trial.full_strain_tensor, grid),
        elastic_strain_tensor=_reshape_two_state(final_trial.elastic_strain_tensor, grid),
        plastic_strain_tensor=_reshape_two_state(final_trial.plastic_strain_tensor, grid),
        observables=observables,
        reaction_forces=_boundary_reactions(
            kinematics.divergence_from_sample_stress(sample_stress)
        ),
        diagnostics=diagnostics,
    )
