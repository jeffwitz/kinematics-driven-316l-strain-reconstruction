"""GPS MFront adapter and its qualified integration policy."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.crystal_orientation import mgis_rotation_argument, validate_rotations
from fem_inhouse.core.linear_solver import LinearSystemMatrixType
from fem_inhouse.core.mfront_3d import MFront3DMaterialPointBatch
from fem_inhouse.core.mfront_behaviours import MFrontBehaviourSpec
from fem_inhouse.core.mfront_condensation import condense_kelvin_tangent_to_engineering
from fem_inhouse.core.mfront_runtime import (
    _ENGINEERING_TO_KELVIN_STRAIN_SCALE,
    _KELVIN_TO_ENGINEERING_STRESS_SCALE,
    MFrontIntegrationError,
    _apply_behaviour_parameters,
    _declared_internal_slices,
    _load_mgis,
    _load_mgis_root,
    _variable_offset,
)
from fem_inhouse.core.mfront_state import MFrontTimingStatistics
from fem_inhouse.core.plane_stress_material import (
    ConstitutiveTrial,
    InPlaneConstitutiveTrial,
    PlaneStressBatchStatistics,
)
from fem_inhouse.core.tensor_reconstruction import kelvin_3d_to_tensor

_SQRT_TWO = np.sqrt(2.0)
_PLANE_STRESS_COMPONENTS = np.array([0, 1, 3])
_TRANSVERSE_COMPONENTS_3D = np.array([2, 4, 5])

def _matching_internal_variable_slices(
    mgis: Any, source: Any, target: Any
) -> list[tuple[slice, slice]]:
    """Pairs of slices for the internal variables the two behaviours share.

    Matched on the variable NAME, so a law that carries extra state -- the GPS
    one has the three closure outputs and an iteration counter the raw law does
    not -- transplants cleanly onto the other.
    """

    def layout(behaviour: Any) -> dict[str, tuple[int, int]]:
        offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for variable in behaviour.isvs:
            size = mgis.getVariableSize(variable, mgis.Hypothesis.Tridimensional)
            offsets[variable.name] = (cursor, size)
            cursor += size
        return offsets

    src, dst = layout(source), layout(target)
    pairs: list[tuple[slice, slice]] = []
    for name, (dst_offset, dst_size) in dst.items():
        if name not in src:
            continue
        src_offset, src_size = src[name]
        if src_size != dst_size:
            raise ValueError(
                f"internal variable {name!r} has size {src_size} in the source "
                f"behaviour and {dst_size} in the target"
            )
        pairs.append(
            (slice(src_offset, src_offset + src_size), slice(dst_offset, dst_offset + dst_size))
        )
    return pairs


class MFrontNativeGeneralisedPlaneStressBatch:
    """Passive-bridge generalized-plane-stress crystal adapter (UMAT closure).

    This is deliberately separate from :class:`MFront3DCondensedPlaneStressBatch`.
    The latter remains the production/reference path.  The adapter runs
    :class:`Fcc316LForestRubinSrixGps`, whose local Newton carries the three
    plane-stress closure unknowns in the GLOBAL frame:

    - the nine components of ``Q_global_to_material`` are passed as per-point
      material properties, so the law can rotate the imposed gradient itself
      and enforce ``(Q^T sigma Q)_zz = (Q^T sigma Q)_xz = (Q^T sigma Q)_yz =
      0``;
    - the bridge applies NO gradient rotation (the law owns the rotation) and
      NO Python closure Newton;
    - the bridge rotates the returned stress, elastic strain and tangent back
      to the global frame, post-multiplies the tangent by the in-plane
      rotation operator (the DSL's automatic tangent differentiates the
      system as if every imposed gradient component entered the elastic
      residual, which only holds for the in-plane ones here), and discards
      the out-of-plane block.

    Transaction semantics (commit, revert, snapshot) are those of the other
    MGIS batches. The local transverse predictor options are accepted for
    interface compatibility but inert: the law's own Newton warm-starts from
    its committed closure state.
    """

    def __init__(
        self,
        library_path: str | Path,
        *,
        behaviour_spec: MFrontBehaviourSpec,
        point_count: int,
        rotation_global_to_material: ArrayLike | None = None,
        thread_count: int = 1,
        behaviour_name: str,
        behaviour_parameters: Mapping[str, float] | None = None,
        temperature_k: float = 293.15,
        maximum_local_iterations: int = 25,
        local_relative_tolerance: float = 1.0e-10,
        local_tolerance_mpa: float = 1.0e-8,
        # `tangent` by default here, unlike the reference: the closure lives
        # inside one Newton, so the quality of the starting point decides
        # whether that Newton converges at all rather than merely how many
        # outer iterations it takes.
        local_transverse_predictor: str = "tangent",
        local_condition_check_mode: str = "on_failure",
        shadow_tangent: bool = False,
        shadow_tangent_scope: str = "all",
        composite_fd_tangent: bool = False,
        composite_fd_step: float = 1.0e-6,
        shadow_behaviour_name: str = "Fcc316LForestRubinSrix",
        shadow_behaviour_id: str = "fcc_forest_rubin_srix",
    ) -> None:
        if point_count < 1:
            raise ValueError("point_count must be positive")
        if thread_count < 1:
            raise ValueError("thread_count must be positive")
        self._mgis = _load_mgis()
        self._behaviour = self._mgis.load(
            str(Path(library_path).resolve()),
            behaviour_name,
            self._mgis.Hypothesis.Tridimensional,
        )
        _apply_behaviour_parameters(
            self._mgis, self._behaviour, behaviour_parameters, behaviour_name
        )
        self._specification = behaviour_spec
        self._behaviour_name = behaviour_name
        self._library_path = str(Path(library_path).resolve())
        self._point_count = point_count
        self._thread_count = int(thread_count)
        self._parameters = dict(behaviour_parameters or {})
        self._temperature = float(temperature_k)
        self._maximum_iterations = int(maximum_local_iterations)
        self._relative_tolerance = float(local_relative_tolerance)
        self._absolute_tolerance = float(local_tolerance_mpa)
        # The CondensedTangent flag of the law: when the behaviour was
        # compiled with `@Parameter real condensedTangent = 1`, its
        # @TangentOperator returns the exact plane-stress Schur of the raw
        # law (computed inside the local Newton) instead of the DSL tangent
        # projected by the in-plane operator. The bridge then skips the
        # projection and the one-sided rotation: the returned tangent is
        # already global and already the condensed block.
        condensed_default = float(
            self._mgis.getParameterDefaultValue(self._behaviour, "CondensedTangent")
        )
        self._condensed_tangent = condensed_default > 0.0
        if self._condensed_tangent:
            self._parameters.setdefault("CondensedTangent", 1.0)
        if local_transverse_predictor not in {"committed", "tangent"}:
            raise ValueError("local_transverse_predictor must be 'committed' or 'tangent'")
        self._local_transverse_predictor = local_transverse_predictor
        if shadow_tangent_scope not in {"all", "substepped", "non_substepped"}:
            raise ValueError(
                "shadow_tangent_scope must be 'all', 'substepped' or 'non_substepped'"
            )
        self._shadow_tangent_scope = shadow_tangent_scope
        if composite_fd_step <= 0.0 or not np.isfinite(composite_fd_step):
            raise ValueError("composite_fd_step must be finite and positive")
        self._composite_fd_enabled = bool(composite_fd_tangent)
        self._composite_fd_step = float(composite_fd_step)
        self._composite_fd_materials: dict[int, MFrontNativeGeneralisedPlaneStressBatch] = {}
        self._composite_fd_seconds = 0.0
        self._composite_fd_points = 0
        self._composite_fd_trajectories = 0
        self._composite_fd_partition_changes = 0
        self._composite_fd_mgis_calls = 0
        self._composite_fd_actual_point_integrations = 0
        self._composite_fd_snapshot_seconds = 0.0
        self._composite_fd_restore_seconds = 0.0
        self._composite_fd_integration_seconds = 0.0
        self._last_composite_fd_diagnostics: dict[str, object] | None = None
        self._rotations = (
            None
            if rotation_global_to_material is None
            else validate_rotations(rotation_global_to_material, point_count=point_count)
        )
        self._mgis_rotations = (
            None if self._rotations is None else mgis_rotation_argument(self._rotations)
        )
        self._manager = self._mgis.MaterialDataManager(self._behaviour, point_count)
        self._thread_pool = (
            _load_mgis_root().ThreadPool(thread_count) if thread_count > 1 else None
        )
        # The nine global-to-material rotation components are per-point
        # material properties: the law owns the strain rotation and needs Q to
        # assemble the closure residuals in the global frame.
        rotations = (
            np.broadcast_to(np.eye(3), (point_count, 3, 3)).copy()
            if self._rotations is None
            else self._rotations
        )
        storage_mode = self._mgis.MaterialStateManagerStorageMode.ExternalStorage
        # `ExternalStorage` means MGIS keeps a POINTER into these buffers and
        # reads them as contiguous. Two consequences, and a single material
        # point hides both of them:
        #
        #  - `rotations[:, row, column]` is a strided view, nine doubles apart.
        #    Handed over as a span it makes every point read another point's Q
        #    components. With one point there is nothing to stride over, which
        #    is exactly why every single-orientation qualification passed while
        #    the 400-point EBSD case started its first Newton residual at 0.83
        #    against the reference's 0.18.
        #  - the buffers are temporaries. Once the loop ends they are freed and
        #    MGIS is left pointing at reclaimed memory.
        #
        # Contiguous copies, kept alive on the instance for as long as the
        # manager is.
        self._property_buffers: dict[str, NDArray] = {}
        for row in range(3):
            for column in range(3):
                name = f"Q{row + 1}{column + 1}"
                values = np.ascontiguousarray(rotations[:, row, column], dtype=float)
                self._property_buffers[name] = values
                self._mgis.setMaterialProperty(self._manager.s0, name, values, storage_mode)
                self._mgis.setMaterialProperty(self._manager.s1, name, values, storage_mode)
        temperature_values = np.ascontiguousarray(
            np.full(point_count, self._temperature, dtype=float)
        )
        self._property_buffers["Temperature"] = temperature_values
        self._mgis.setExternalStateVariable(
            self._manager.s0, "Temperature", temperature_values, storage_mode
        )
        self._mgis.setExternalStateVariable(
            self._manager.s1, "Temperature", temperature_values, storage_mode
        )
        self._observable_slices = _declared_internal_slices(
            self._mgis,
            self._behaviour,
            self._mgis.Hypothesis.Tridimensional,
            behaviour_spec,
        )
        elastic_offset = _variable_offset(
            self._mgis,
            self._behaviour.isvs,
            "ElasticStrain",
            self._mgis.Hypothesis.Tridimensional,
            expected_size=6,
        )
        assert elastic_offset is not None
        self._elastic_offset = elastic_offset
        # Closure state variables: the global transverse strains solved by the
        # law's own Newton, read back to assemble the global total strain.
        closure_offsets = {
            name: _variable_offset(
                self._mgis,
                self._behaviour.isvs,
                name,
                self._mgis.Hypothesis.Tridimensional,
                expected_size=1,
            )
            for name in ("ezz", "eyz", "exz")
        }
        assert all(offset is not None for offset in closure_offsets.values())
        self._ezz_offset = closure_offsets["ezz"]
        self._eyz_offset = closure_offsets["eyz"]
        self._exz_offset = closure_offsets["exz"]
        self._committed_strain = np.zeros((point_count, 6), dtype=float)
        self._latest_trial: ConstitutiveTrial | None = None
        self._latest_total_kelvin: NDArray | None = None
        self._latest_in_plane: NDArray | None = None
        self._latest_dt: float | None = None
        self._evaluate_calls = 0
        self._internal_integrations = 0
        self._integration_seconds = 0.0
        self._local_iterations = np.zeros(point_count, dtype=np.int64)
        self._accepted_transverse = np.zeros((point_count, 3), dtype=float)
        self._accepted_in_plane: NDArray | None = None
        self._accepted_cbb: NDArray | None = None
        self._accepted_cba: NDArray | None = None
        self._latest_transverse: NDArray | None = None
        self._latest_cbb: NDArray | None = None
        self._latest_cba: NDArray | None = None
        self._warm_start_uses = 0
        self._warm_start_resets = 0
        self._maximum_residual = 0.0
        self._last_local_failure: dict[str, object] | None = None
        # Sub-stepping bound. Eight halvings take a full increment down to
        # 1/256 of itself; beyond that a failure is not a step-size problem.
        self._maximum_substeps = 256
        self._substep_uses = 0
        self._rerun_uses = 0
        self._rerun_failures = 0
        self._substep_divisions_max = 0
        self._substep_points = 0
        #: Bisection stops at this block size. Measured on P43 20x20: at 32 the
        #: Newton count goes back to 63 from 52, the field agreement loses an
        #: order of magnitude and the material time rises -- sub-stepping
        #: thirty-one healthy points to spare one is worse on every count. One
        #: is the right answer; it is the serial probing that costs, not the
        #: depth.
        self._minimum_substep_span = 1
        #: Indices that refused the full step last time, as single-point spans.
        self._failing_cache: list[tuple[int, int]] = []
        self._last_substep_mask = np.zeros(point_count, dtype=bool)
        self._last_substep_divisions = np.ones(point_count, dtype=np.int64)
        self._last_shadow_diagnostics: dict[str, object] | None = None
        # Above this many the cache would cost more than the bisection it
        # saves, so it scales with the batch: `k` cached spans cost `k`
        # single-point sub-steps plus one pooled proof, against the roughly
        # `n log2(n)` SERIAL point integrations a bisection charges. A fixed 32
        # was fine on four hundred points and silently disabled the cache on
        # ten thousand, where the failing set is proportionally larger -- the
        # 100x100 window then fell back to 0.96x while 20x20 ran at 1.2-1.7x.
        self._maximum_cached_spans = max(32, point_count // 8)
        self._cache_hits = 0
        self._cache_misses = 0
        # Route 2 of the handoff, and the reason it is needed. Sub-stepping is
        # what makes the joint Newton reach the deep plastic states at all, but
        # the matrix it leaves behind is the LAST sub-step's, not the whole
        # increment's, and A6 then fails by a factor of two to five. Route 1
        # (rerun the full increment from the located root, through a predictor
        # external state variable) was implemented and measured to leave the
        # tangent BIT-IDENTICAL -- the mechanism never fires -- so the
        # conclusion drawn from it, that the DSL tangent machinery is at fault,
        # is not supported. Measured instead: the RAW 3D law's consistent
        # tangent is exact to 1e-10 at every depth of the frozen history, and
        # the GPS tangent is exact wherever no sub-stepping happens (7e-08 at
        # increment 1, 1e-07 at 2, 1.8e-06 at 3). The chain is right; only the
        # sub-stepped matrix is wrong.
        #
        # Route 2 was then implemented on that basis and MEASURED TO FAIL, so
        # it is off by default and kept only as the record. The premise was
        # that imposing the located transverse strain turns the problem back
        # into the plain 18-unknown one, whose tangent is the qualified one.
        # It does not: driven in lockstep from the same committed state with
        # exactly the GPS transverse strain imposed, the raw law converges to
        # the OTHER root -- sigma_zz of -152 MPa at increment 2, growing to
        # -4221 by increment 8, against the 0 the closure enforces. That is the
        # multiple-root structure of
        # `validation/srix_plane_stress_branch_diagnostic.md` reappearing
        # inside the 18-unknown problem: imposing the strain does not select
        # the branch. Any future route must carry the branch, not just the
        # strain.
        self._shadow: MFront3DMaterialPointBatch | None = None
        self._shadow_failures = 0
        self._shadow_has_trial = False
        # tr(eel) = tr(eps_total) is the trace of the elastic residual: the
        # Schmid tensors are deviatoric and a rotation preserves the trace, so
        # a converged state that violates it is not a solution of the system
        # the law says it solved. Cheap enough to check on every evaluation,
        # and it is the invariant that exposed the overwrite defect.
        self._maximum_kinematic_defect = 0.0
        self._previous_transverse_delta: NDArray | None = None
        self._previous_in_plane_norm = 0.0
        self._committed_in_plane = np.zeros((point_count, 3), dtype=float)
        if shadow_tangent:
            from fem_inhouse.core.mfront_behaviours import MFRONT_BEHAVIOURS

            shadow_spec = MFRONT_BEHAVIOURS.get(shadow_behaviour_id)
            self._shadow = MFront3DMaterialPointBatch(
                library_path,
                behaviour_spec=shadow_spec,
                point_count=point_count,
                rotation_global_to_material=self._rotations,
                thread_count=thread_count,
                behaviour_name=shadow_behaviour_name,
                behaviour_parameters=behaviour_parameters,
                temperature_k=temperature_k,
            )
            # The two laws do not share an internal-variable layout -- the GPS
            # one carries ezz/eyz/exz and LocalIterations that the raw one does
            # not -- so the transplant is done by NAME, never by offset.
            self._shadow_isv_map = _matching_internal_variable_slices(
                self._mgis, self._behaviour, self._shadow._behaviour
            )

    @property
    def substep_uses(self) -> int:
        """How many increments needed halving. Zero is the healthy case."""

        return self._substep_uses

    @property
    def substep_divisions_max(self) -> int:
        return self._substep_divisions_max

    @property
    def substep_cache_hits(self) -> int:
        return self._cache_hits

    @property
    def substep_cache_misses(self) -> int:
        return self._cache_misses

    @property
    def substep_points(self) -> int:
        """Total points ever sub-stepped, against the whole batch before."""

        return self._substep_points

    @property
    def last_substep_mask(self) -> NDArray:
        """Points sub-stepped during the most recent constitutive call."""

        return self._last_substep_mask.copy()

    @property
    def last_substep_divisions(self) -> NDArray:
        """Per-point division count used by the most recent constitutive call."""

        return self._last_substep_divisions.copy()

    @property
    def last_shadow_diagnostics(self) -> dict[str, object] | None:
        """Pointwise comparison of the runtime shadow and GPS trial."""

        return self._last_shadow_diagnostics

    @property
    def last_composite_fd_diagnostics(self) -> dict[str, object] | None:
        return self._last_composite_fd_diagnostics

    @property
    def point_count(self) -> int:
        return self._point_count

    @property
    def backend_name(self) -> str:
        return "mfront-native-generalised-plane-stress"

    @property
    def completion_strategy(self) -> str:
        return "mfront_native_generalised_plane_stress"

    @property
    def linear_system_matrix_type(self) -> LinearSystemMatrixType:
        return self._specification.linear_system_matrix_type

    @property
    def thread_count(self) -> int:
        return self._thread_count

    @property
    def statistics(self) -> PlaneStressBatchStatistics:
        return PlaneStressBatchStatistics(
            maximum_gauss_point_plane_stress_residual_mpa=self._maximum_residual,
            maximum_local_plane_stress_iterations=int(self._local_iterations.max()),
            mean_local_plane_stress_iterations=float(self._local_iterations.mean()),
        )

    @property
    def timing_statistics(self) -> MFrontTimingStatistics:
        return MFrontTimingStatistics(
            integration_seconds=self._integration_seconds,
            evaluate_calls=self._evaluate_calls,
            material_point_integrations=self._internal_integrations,
            material_point_integrations_with_tangent=self._internal_integrations,
            material_block_integration_calls=self._evaluate_calls,
            material_block_count=1,
            native_batch_calls=self._evaluate_calls,
            native_material_points=self._evaluate_calls * self._point_count,
            native_internal_integrations=self._internal_integrations,
            native_total_local_iterations=int(np.sum(self._local_iterations)),
            native_thread_count=self._thread_count,
            native_substep_points=self._substep_points,
            native_substep_cache_hits=self._cache_hits,
            native_substep_cache_misses=self._cache_misses,
            composite_fd_seconds=self._composite_fd_seconds,
            composite_fd_points=self._composite_fd_points,
            composite_fd_trajectories=self._composite_fd_trajectories,
            composite_fd_partition_changes=self._composite_fd_partition_changes,
            composite_fd_mgis_calls=self._composite_fd_mgis_calls,
            composite_fd_actual_point_integrations=self._composite_fd_actual_point_integrations,
            composite_fd_snapshot_seconds=self._composite_fd_snapshot_seconds,
            composite_fd_restore_seconds=self._composite_fd_restore_seconds,
            composite_fd_integration_seconds=self._composite_fd_integration_seconds,
            composite_fd_other_seconds=max(
                0.0,
                self._composite_fd_seconds
                - self._composite_fd_snapshot_seconds
                - self._composite_fd_restore_seconds
                - self._composite_fd_integration_seconds,
            ),
        )

    @property
    def committed_transverse_strain_kelvin(self) -> NDArray:
        return self._committed_strain[:, _TRANSVERSE_COMPONENTS_3D].copy()

    @property
    def local_transverse_predictor(self) -> str:
        return self._local_transverse_predictor

    @property
    def warm_start_uses(self) -> int:
        return self._warm_start_uses

    @property
    def warm_start_resets(self) -> int:
        return self._warm_start_resets

    @property
    def last_local_failure(self) -> dict[str, object] | None:
        return None if self._last_local_failure is None else dict(self._last_local_failure)

    def _set_parameters(self) -> None:
        _apply_behaviour_parameters(
            self._mgis, self._behaviour, self._parameters, self._behaviour_name
        )

    def _integrate_once(
        self,
        in_plane_kelvin: NDArray,
        time_increment: float,
        transverse_kelvin: NDArray | None = None,
        span: tuple[int, int] | None = None,
    ) -> int:
        """One integration, with the transverse strain kept in the gradient.

        The law adds its closure unknowns to the gradient it is given rather
        than replacing three of its components, so the transverse strain the
        bridge supplies IS part of the imposed gradient. On success the
        increment the closure found is folded back into `s1.gradients`, which
        makes `s1.gradients` the true total strain at every point of the
        transaction: `mgis.update` then carries it into the committed state,
        `mgis.revert` discards it with everything else, and every sub-step
        starts from the transverse strain the previous one reached.

        This is the whole point of the change of 2026-08-07. While the law
        overwrote the transverse components instead, MGIS held a gradient that
        was fictitious on three components; nothing could cross-check the
        state, and the recorded transverse strain drifted from the integrated
        one by a factor equal to the increment number.
        """

        # `in_plane_kelvin` is the TOTAL in-plane strain, not an increment --
        # that is the contract of `evaluate` for every plane-stress batch in
        # this module, and the reference sets the gradient absolutely. This
        # bridge used to write `s0.gradients + gradient`, so it applied the
        # total as if it were an increment and the imposed strain accumulated
        # as 1+2+3+... instead of 1,2,3: at increment 2 the in-plane trace was
        # 3.0e-3 where 2.0e-3 was asked. Increment 1 was unaffected -- total
        # and increment coincide there -- which is exactly why every
        # comparison against the reference agreed at increment 1 and diverged
        # from increment 2.
        gradients = np.asarray(self._manager.s0.gradients).copy()
        gradients[:, _PLANE_STRESS_COMPONENTS] = in_plane_kelvin
        if transverse_kelvin is not None:
            # The transverse predictor of the reference, applied to the
            # GRADIENT rather than to an outer iterate. The law adds its
            # closure unknown on top, so `dezz` no longer has to travel the
            # whole transverse strain -- it only corrects what the prediction
            # missed. That is what makes one joint Newton enough.
            gradients[:, _TRANSVERSE_COMPONENTS_3D] = transverse_kelvin
        self._manager.s1.gradients[:, :] = gradients
        integration_type = self._mgis.IntegrationType.IntegrationWithConsistentTangentOperator
        if span is not None:
            # A sub-range. `integrate` reads `s0` and writes `s1` for the range
            # only, so the points outside it keep whatever a previous call left
            # there -- which is what makes the isolation below safe.
            begin, end = span
            status = int(
                self._mgis.integrate(
                    self._manager,
                    integration_type,
                    float(time_increment),
                    begin,
                    end,
                )
            )
        elif self._thread_pool is None:
            status = int(
                self._mgis.integrate(
                    self._manager,
                    integration_type,
                    float(time_increment),
                    0,
                    self._point_count,
                )
            )
        else:
            # This bridge only ever had the SERIAL overload, so it integrated
            # single-threaded while the reference used the whole pool. On a
            # four-thread run that is most of the "unexplained factor 2.5" the
            # per-integration cost carried: the two backends were not being
            # compared on the same number of cores.
            status = int(
                self._mgis.integrate(
                    self._thread_pool,
                    self._manager,
                    integration_type,
                    float(time_increment),
                )
            )
        if status == 1:
            # The closure variables are OUTPUTS now, and they hold the TOTAL
            # transverse strain rather than an increment, so the gradient takes
            # them directly. `s1.gradients` is then the true total strain and
            # `mgis.update` carries it into the committed state.
            closure = np.array(
                [self._ezz_offset, self._exz_offset, self._eyz_offset], dtype=int
            )
            rows = slice(None) if span is None else slice(span[0], span[1])
            self._manager.s1.gradients[rows, _TRANSVERSE_COMPONENTS_3D] = np.asarray(
                self._manager.s1.internal_state_variables
            )[rows, closure]
        return status

    def _shadow_condensed_tangent(
        self,
        in_plane: NDArray,
        internal_state_variables: NDArray,
        time_increment: float,
    ) -> NDArray | None:
        """Evaluate the runtime raw full-step shadow tangent.

        A DERIVATIVE CALCULATOR, nothing else. The shadow has no life of its
        own: its committed state is transplanted from the GPS's at every call
        -- internal variables by name, committed stress as is, both being
        material-frame in either law, and the committed gradient rotated from
        the GPS's global convention into the crystal one the raw bridge uses --
        it is then integrated once at the imposed in-plane strain WITH the
        transverse strain the GPS closure converged, and reverted. This is a
        separate full-step constitutive trajectory, not a same-state
        derivative oracle when GPS used sub-stepping.

        Runtime diagnostics compare this tangent and the final internal
        variables pointwise with the GPS trial. A mismatch can therefore be a
        full-step versus sub-stepped trajectory difference, not an algebraic
        tangent defect.

        Returns `None` on any failure, leaving the DSL tangent in place.
        """

        if self._shadow is None:
            return None
        source = self._manager.s0
        target = self._shadow._manager.s0
        internal = np.asarray(source.internal_state_variables)
        destination = np.asarray(target.internal_state_variables)
        for source_slice, target_slice in self._shadow_isv_map:
            destination[:, target_slice] = internal[:, source_slice]
        target.internal_state_variables[:, :] = destination
        # Stress is material-frame in both laws; the gradient is not.
        target.thermodynamic_forces[:, :] = np.asarray(source.thermodynamic_forces)
        gradients = np.ascontiguousarray(np.asarray(source.gradients).reshape(-1).copy())
        if self._mgis_rotations is not None:
            self._mgis.rotateGradients(
                gradients, self._shadow._behaviour, self._mgis_rotations
            )
        target.gradients[:, :] = gradients.reshape(self._point_count, 6)
        self._mgis.revert(self._shadow._manager)

        total = np.zeros((self._point_count, 6), dtype=float)
        total[:, _PLANE_STRESS_COMPONENTS] = (
            in_plane * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
        )
        total[:, _TRANSVERSE_COMPONENTS_3D] = np.asarray(self._manager.s1.gradients)[
            :, _TRANSVERSE_COMPONENTS_3D
        ]
        try:
            trial = self._shadow.evaluate(
                total, time_increment=time_increment, collect_observables=False
            )
            engineering, _ = condense_kelvin_tangent_to_engineering(
                trial.consistent_tangent_kelvin_mpa, check_condition=False
            )
            gps_tangent = np.asarray(self._manager.K).copy()
            in_plane_operator = np.zeros((6, 6), dtype=float)
            in_plane_operator[_PLANE_STRESS_COMPONENTS, _PLANE_STRESS_COMPONENTS] = 1.0
            gps_tangent = gps_tangent @ in_plane_operator
            if self._mgis_rotations is not None:
                for column in range(6):
                    flat = np.ascontiguousarray(gps_tangent[:, :, column].reshape(-1))
                    self._mgis.rotateThermodynamicForces(
                        flat, self._behaviour, self._mgis_rotations
                    )
                    gps_tangent[:, :, column] = flat.reshape(self._point_count, 6)
            gps_tangent = gps_tangent[:, _PLANE_STRESS_COMPONENTS][
                :, :, _PLANE_STRESS_COMPONENTS
            ]
            gps_tangent = gps_tangent * _KELVIN_TO_ENGINEERING_STRESS_SCALE[None, :, None]
            gps_tangent = gps_tangent * _ENGINEERING_TO_KELVIN_STRAIN_SCALE[None, None, :]
            self._gps_tangent_engineering = gps_tangent
            tangent_error = np.linalg.norm(engineering - gps_tangent, axis=(1, 2)) / np.maximum(
                np.linalg.norm(gps_tangent, axis=(1, 2)), 1.0e-30
            )
            state_differences: dict[str, NDArray] = {}
            gps_isv = np.asarray(source.internal_state_variables)
            shadow_isv = np.asarray(target.internal_state_variables)
            for source_slice, target_slice in self._shadow_isv_map:
                name = str(source_slice)
                state_differences[name] = np.max(
                    np.abs(gps_isv[:, source_slice] - shadow_isv[:, target_slice]), axis=1
                )
            self._last_shadow_diagnostics = {
                "substep": self._last_substep_mask.copy(),
                "divisions": self._last_substep_divisions.copy(),
                "tangent_relative_error": tangent_error,
                "state_differences": state_differences,
                "gps_tangent": np.asarray(gps_tangent).copy(),
                "shadow_tangent": np.asarray(engineering).copy(),
                "scope": self._shadow_tangent_scope,
            }
        except Exception:
            self._shadow_failures += 1
            self._last_shadow_diagnostics = {"failure": True}
            return None
        finally:
            # No committed evolution of its own, ever.
            self._shadow.revert()
        if not np.isfinite(engineering).all():
            self._shadow_failures += 1
            return None
        selected = np.ones(self._point_count, dtype=bool)
        if self._shadow_tangent_scope == "substepped":
            selected = self._last_substep_mask
        elif self._shadow_tangent_scope == "non_substepped":
            selected = ~self._last_substep_mask
        result = np.asarray(self._gps_tangent_engineering, dtype=float).copy()
        result[selected] = np.asarray(engineering, dtype=float)[selected]
        return result

    def _composite_fd_material(self, point: int) -> MFrontNativeGeneralisedPlaneStressBatch:
        """Return a cached one-point GPS evaluator for composite FD."""

        if point not in self._composite_fd_materials:
            self._composite_fd_materials[point] = type(self)(
                self._library_path,
                behaviour_spec=self._specification,
                point_count=1,
                rotation_global_to_material=(
                    None if self._rotations is None else self._rotations[point : point + 1]
                ),
                thread_count=1,
                behaviour_name=self._behaviour_name,
                behaviour_parameters=self._parameters,
                temperature_k=self._temperature,
                maximum_local_iterations=self._maximum_iterations,
                local_relative_tolerance=self._relative_tolerance,
                local_tolerance_mpa=self._absolute_tolerance,
                local_transverse_predictor=self._local_transverse_predictor,
                local_condition_check_mode="on_failure",
                shadow_tangent=False,
                composite_fd_tangent=False,
            )
        return self._composite_fd_materials[point]

    @staticmethod
    def _point_snapshot(snapshot: tuple[Any, ...], point: int) -> tuple[Any, ...]:
        """Restrict a full GPS snapshot to one material point."""

        values: list[Any] = []
        for index, value in enumerate(snapshot):
            if value is None:
                values.append(None)
            elif index < 5:
                values.append(np.asarray(value)[point : point + 1].copy())
            else:
                values.append(np.asarray(value)[point : point + 1].copy())
        return tuple(values)

    def _composite_fd_tangent(
        self,
        in_plane: NDArray,
        time_increment: float,
        committed_snapshot: tuple[Any, ...],
    ) -> NDArray:
        """Finite-difference the actual sub-stepped application at bad points."""

        started = time.perf_counter()
        result = np.zeros((self._point_count, 3, 3), dtype=float)
        diagnostics: list[dict[str, object]] = []
        active_points = np.flatnonzero(self._last_substep_mask)
        for point in active_points:
            point_material = self._composite_fd_material(int(point))
            snapshot_started = time.perf_counter()
            point_snapshot = self._point_snapshot(committed_snapshot, int(point))
            self._composite_fd_snapshot_seconds += time.perf_counter() - snapshot_started
            base_partition = bool(self._last_substep_mask[point])
            tangent = np.zeros((3, 3), dtype=float)
            partition_changed = False
            for column in range(3):
                plus = np.asarray(in_plane[point : point + 1], dtype=float).copy()
                minus = plus.copy()
                plus[0, column] += self._composite_fd_step
                minus[0, column] -= self._composite_fd_step
                restore_started = time.perf_counter()
                point_material.restore_state(point_snapshot)
                self._composite_fd_restore_seconds += time.perf_counter() - restore_started
                before = point_material.timing_statistics
                trial_plus = point_material.evaluate(
                    plus, time_increment=time_increment, consistent_tangent=True
                )
                after = point_material.timing_statistics
                self._composite_fd_mgis_calls += max(
                    0,
                    after.native_batch_calls - before.native_batch_calls,
                )
                self._composite_fd_actual_point_integrations += max(
                    0,
                    after.native_internal_integrations
                    - before.native_internal_integrations,
                )
                self._composite_fd_integration_seconds += max(
                    0.0,
                    after.integration_seconds - before.integration_seconds,
                )
                plus_partition = bool(point_material.last_substep_mask[0])
                restore_started = time.perf_counter()
                point_material.restore_state(point_snapshot)
                self._composite_fd_restore_seconds += time.perf_counter() - restore_started
                before = point_material.timing_statistics
                trial_minus = point_material.evaluate(
                    minus, time_increment=time_increment, consistent_tangent=True
                )
                after = point_material.timing_statistics
                self._composite_fd_mgis_calls += max(
                    0,
                    after.native_batch_calls - before.native_batch_calls,
                )
                self._composite_fd_actual_point_integrations += max(
                    0,
                    after.native_internal_integrations
                    - before.native_internal_integrations,
                )
                self._composite_fd_integration_seconds += max(
                    0.0,
                    after.integration_seconds - before.integration_seconds,
                )
                minus_partition = bool(point_material.last_substep_mask[0])
                partition_changed |= plus_partition != base_partition
                partition_changed |= minus_partition != base_partition
                tangent[:, column] = (
                    np.asarray(trial_plus.stress_in_plane_mpa)[0]
                    - np.asarray(trial_minus.stress_in_plane_mpa)[0]
                ) / (2.0 * self._composite_fd_step)
            result[point] = tangent
            self._composite_fd_trajectories += 6
            if partition_changed:
                self._composite_fd_partition_changes += 1
            diagnostics.append(
                {
                    "point": int(point),
                    "divisions": int(self._last_substep_divisions[point]),
                    "partition_unchanged": not partition_changed,
                    "tangent": tangent.copy(),
                }
            )
        self._composite_fd_seconds += time.perf_counter() - started
        self._composite_fd_points += len(active_points)
        self._last_composite_fd_diagnostics = {
            "points": diagnostics,
            "step": self._composite_fd_step,
        }
        return result

    @property
    def shadow_failures(self) -> int:
        return self._shadow_failures

    @property
    def maximum_kinematic_defect(self) -> float:
        """Worst relative violation of `tr(eel) = tr(eps_total)` so far."""

        return self._maximum_kinematic_defect

    def _committed_snapshot(self) -> tuple[NDArray, NDArray, NDArray]:
        return (
            np.asarray(self._manager.s0.gradients).copy(),
            np.asarray(self._manager.s0.thermodynamic_forces).copy(),
            np.asarray(self._manager.s0.internal_state_variables).copy(),
        )

    def _restore_committed(self, snapshot: tuple[NDArray, NDArray, NDArray]) -> None:
        gradients, forces, internal = snapshot
        self._manager.s0.gradients[:, :] = gradients
        self._manager.s0.thermodynamic_forces[:, :] = forces
        self._manager.s0.internal_state_variables[:, :] = internal

    def _predicted_transverse(self, in_plane: NDArray) -> NDArray | None:
        """Where the reference would start its closure, section 8.10.

        Two strategies, both taken from `MFront3DCondensedPlaneStressBatch`:
        the accepted transverse strain of the last global iterate, and -- when
        the accepted condensed blocks are available -- a first-order
        extrapolation of the closure along the in-plane increment,

            eps_b <- eps_b_accepted - Cbb^-1 Cba (eps_a - eps_a_accepted)

        which is exact to first order and is what lets the reference converge
        its closure in a handful of iterations from any state. The GPS bridge
        carried all of this machinery, unused, since it was written.
        """

        committed = self._committed_strain[:, _TRANSVERSE_COMPONENTS_3D].copy()
        if self._local_transverse_predictor == "committed":
            return committed
        # The reference extrapolates with `Cbb^-1 Cba`, and that route is NOT
        # available here: the closure enforces sigma_transverse = 0, so the
        # transverse ROWS of the GPS tangent are identically zero and `Cbb` is
        # the zero matrix. The reference's `Cbb` is a block of the
        # UNCONSTRAINED 3D tangent, which this law never exposes.
        #
        # What is available is the previous accepted transverse increment,
        # rescaled by the in-plane increment it came with. On the frozen
        # history that leaves about `3e-05` for the closure to find instead of
        # `1e-03`: a starting point thirty times closer, from information the
        # bridge already holds.
        if self._previous_transverse_delta is None or self._previous_in_plane_norm <= 0.0:
            return committed
        norm = float(
            np.linalg.norm(
                (in_plane - self._committed_in_plane) * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
            )
        )
        scale = norm / self._previous_in_plane_norm
        if not np.isfinite(scale) or scale > 10.0:
            return committed
        return committed + scale * self._previous_transverse_delta

    def _failing_spans(
        self, in_plane_kelvin: NDArray, time_increment: float
    ) -> list[tuple[int, int]]:
        """Which points refuse the full step, found by bisection on the batch.

        `integrate` reports one status for a whole range, so a failing point is
        located by halving. About `k log2(n)` range integrations for `k` bad
        points, against the `n x divisions` the previous whole-batch
        sub-stepping charged for every one of them.
        """

        # The bisection uses the SERIAL range overload -- MGIS only integrates
        # a range without the pool -- so it trades parallelism for isolation.
        # Splitting all the way down to a single point pays that trade far too
        # often; stopping at a block keeps most of the benefit for a fraction
        # of the probes.
        spans: list[tuple[int, int]] = []
        pending = [(0, self._point_count)]
        while pending:
            begin, end = pending.pop()
            if self._integrate_once(
                in_plane_kelvin, time_increment, None, (begin, end)
            ) == 1:
                continue
            if end - begin <= self._minimum_substep_span:
                spans.append((begin, end))
                continue
            middle = (begin + end) // 2
            pending.append((middle, end))
            pending.append((begin, middle))
        return spans

    def _advance_span(self, span: tuple[int, int]) -> None:
        """`mgis.update` restricted to a range, in numpy.

        MGIS updates the whole manager, which would drag every healthy point
        along the sub-steps of a sick one. Only the range being sub-stepped
        may advance.
        """

        begin, end = span
        rows = slice(begin, end)
        self._manager.s0.gradients[rows, :] = np.asarray(
            self._manager.s1.gradients
        )[rows, :]
        self._manager.s0.thermodynamic_forces[rows, :] = np.asarray(
            self._manager.s1.thermodynamic_forces
        )[rows, :]
        self._manager.s0.internal_state_variables[rows, :] = np.asarray(
            self._manager.s1.internal_state_variables
        )[rows, :]

    def _restore_span(
        self, snapshot: tuple[NDArray, NDArray, NDArray], span: tuple[int, int]
    ) -> None:
        gradients, forces, internal = snapshot
        rows = slice(span[0], span[1])
        self._manager.s0.gradients[rows, :] = gradients[rows, :]
        self._manager.s0.thermodynamic_forces[rows, :] = forces[rows, :]
        self._manager.s0.internal_state_variables[rows, :] = internal[rows, :]

    def _substep_span(
        self,
        span: tuple[int, int],
        in_plane_kelvin: NDArray,
        time_increment: float,
        snapshot: tuple[NDArray, NDArray, NDArray],
    ) -> tuple[int, int]:
        """Halve the increment for ONE range until it goes through."""

        committed_in_plane = snapshot[0][:, _PLANE_STRESS_COMPONENTS]
        divisions = 2
        status = -1
        while divisions <= self._maximum_substeps:
            self._restore_span(snapshot, span)
            step = time_increment / divisions
            succeeded = True
            for index in range(divisions):
                # Interpolate the TOTAL in-plane strain between the committed
                # state and the target; a fraction of a total is not a strain.
                weight = (index + 1) / divisions
                stage = committed_in_plane + weight * (
                    in_plane_kelvin - committed_in_plane
                )
                status = self._integrate_once(stage, step, None, span)
                if status != 1:
                    succeeded = False
                    break
                if index < divisions - 1:
                    self._advance_span(span)
            if succeeded:
                self._restore_span(snapshot, span)
                return 1, divisions
            divisions *= 2
        self._restore_span(snapshot, span)
        return status, divisions // 2

    def _try_cached_spans(
        self,
        in_plane_kelvin: NDArray,
        time_increment: float,
        transverse_kelvin: NDArray | None,
        snapshot: tuple[NDArray, NDArray, NDArray],
    ) -> tuple[int, int] | None:
        """Sub-step last call's failures, then PROVE no other point failed.

        Plasticity localises: the points whose joint Newton refuses the full
        step are the same from one call to the next -- two out of four hundred
        on the P43 window, measured. Bisecting for them again costs about nine
        whole-batch passes, and every one of them is serial because MGIS only
        integrates a range without its pool.

        The proof of completeness is the trick, and it is exact. Once a cached
        point has been sub-stepped, its `s0` is ADVANCED onto the state it
        reached. A grouped re-integration then presents that point a strain
        increment of exactly zero -- it converges on the guarded elastic branch
        in one iteration -- while every other point still sees the full
        increment. So a status of 1 on that single POOLED call proves that no
        point outside the cache failed. Nothing has to be read out of the
        state, which the closure-residual detector showed is impossible
        anyway.

        Returns `None` when the cache is empty, incomplete or wrong, leaving
        the caller to bisect.
        """

        if not self._failing_cache:
            return None
        advanced: list[tuple[int, int]] = []
        tangents: list[NDArray] = []
        worst = 1
        complete = True
        for span in self._failing_cache:
            status, divisions = self._substep_span(
                span, in_plane_kelvin, time_increment, snapshot
            )
            if status != 1:
                complete = False
                break
            worst = max(worst, divisions)
            # The verification below hands these points a ZERO increment, and
            # the law then answers on its guarded elastic branch -- with the
            # ELASTIC tangent. Their sub-stepped tangent has to be kept and put
            # back, or the global Newton is given an elastic matrix exactly at
            # the points that are most plastic. Measured the hard way: without
            # this, P43 stops converging at increment 5.
            tangents.append(np.asarray(self._manager.K)[span[0] : span[1]].copy())
            self._advance_span(span)
            advanced.append(span)

        verified = 0
        if complete:
            verified = self._integrate_once(
                in_plane_kelvin, time_increment, transverse_kelvin
            )
            for span, block in zip(advanced, tangents, strict=True):
                self._manager.K[span[0] : span[1]] = block
        for span in advanced:
            self._restore_span(snapshot, span)
        if complete and verified == 1:
            self._cache_hits += 1
            self._substep_uses += 1
            self._substep_points += sum(end - begin for begin, end in advanced)
            self._substep_divisions_max = max(self._substep_divisions_max, worst)
            return 1, worst
        self._cache_misses += 1
        self._failing_cache = []
        return None

    def _integrate_with_substepping(
        self,
        in_plane_kelvin: NDArray,
        time_increment: float,
        transverse_kelvin: NDArray | None = None,
    ) -> tuple[int, int]:
        """Integrate one increment, halving it for the points that need it.

        The joint Newton of the GPS law has no line search and no step control
        -- the Implicit DSL offers neither -- so it fails on large plastic
        increments even though the root is there. Measured by
        `scripts/diagnose_srix_closure_root_sweep.py`: the closure equation has
        exactly one root at every increment of the frozen history. What fails
        is the iteration, not the problem, and halving the increment is the
        remedy.

        Until 2026-08-07 that halving was applied to the WHOLE BATCH. One
        stubborn point out of four hundred therefore charged four hundred
        points for up to thirty-two sub-steps each. The failing points are now
        isolated by bisection and only they are sub-stepped, with `s0` advanced
        and restored on their range alone so no healthy point is dragged along.

        The returned tangent of a sub-stepped point is that of its last
        sub-step. Measured: A6 stays at `1.2e-07` at every increment of the
        frozen history, sub-stepped ones included.
        """

        self._last_substep_mask[:] = False
        self._last_substep_divisions[:] = 1
        self._last_shadow_diagnostics = None
        status = self._integrate_once(
            in_plane_kelvin, time_increment, transverse_kelvin
        )
        if status == 1 or self._maximum_substeps <= 1:
            return status, 1

        snapshot = self._committed_snapshot()
        cached = self._try_cached_spans(
            in_plane_kelvin, time_increment, transverse_kelvin, snapshot
        )
        if cached is not None:
            for begin, end in self._failing_cache:
                self._last_substep_mask[begin:end] = True
                self._last_substep_divisions[begin:end] = cached[1]
            return cached
        spans = self._failing_spans(in_plane_kelvin, time_increment)
        if not spans:
            # The bisection found every range acceptable on its own, so `s1`
            # now holds a complete state assembled range by range.
            self._substep_uses += 1
            return 1, 1
        self._substep_points += sum(end - begin for begin, end in spans)
        worst = 1
        for span in spans:
            status, divisions = self._substep_span(
                span, in_plane_kelvin, time_increment, snapshot
            )
            worst = max(worst, divisions)
            if status != 1:
                self._restore_committed(snapshot)
                return status, worst
            self._last_substep_mask[span[0] : span[1]] = True
            self._last_substep_divisions[span[0] : span[1]] = divisions
        self._substep_uses += 1
        self._substep_divisions_max = max(self._substep_divisions_max, worst)
        self._failing_cache = spans if len(spans) <= self._maximum_cached_spans else []
        return 1, worst

    def evaluate_in_plane(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> InPlaneConstitutiveTrial:
        full = self.evaluate(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
        )
        return InPlaneConstitutiveTrial(
            stress_in_plane_mpa=full.stress_in_plane_mpa,
            tangent_in_plane_mpa=full.tangent_in_plane_mpa,
            observables=full.observables,
            local_plane_stress_iterations=full.local_plane_stress_iterations,
        )

    def evaluate(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> ConstitutiveTrial:
        in_plane = np.asarray(in_plane_strain, dtype=float)
        if in_plane.shape != (self._point_count, 3):
            raise ValueError(f"in_plane_strain must have shape {(self._point_count, 3)}")
        # The full snapshot is only needed when the selective composite-FD
        # tangent is enabled.  Keeping the baseline path free of this copy is
        # important: ``composite_fd=False`` must remain a performance-neutral
        # switch.
        committed_snapshot = self.snapshot_state() if self._composite_fd_enabled else None
        # The closure state variables own the transverse strains: the law
        # overwrites the transverse gradient components with (dezz, deyz,
        # dexz) before rotating. The bridge passes the in-plane gradient in
        # Kelvin storage (the repository convention for MGIS arrays) and zero
        # transverse values, which the law never reads.
        in_plane_kelvin = in_plane * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
        predicted_transverse = self._predicted_transverse(in_plane)
        self._set_parameters()
        started = time.perf_counter()
        # d(deto_m)/ddeto restricted to the in-plane columns, in the MGIS
        # storage: the rotation operator of the stock gradient rotation,
        # with the transverse columns zeroed. The DSL's automatic tangent
        # differentiates the local system as if every imposed gradient
        # component entered the elastic residual; here only the in-plane
        # ones do, so the returned tangent is post-multiplied by this
        # operator.
        in_plane_operator: NDArray
        # d(feel)/d(deto) is now exactly -P, with P the CONSTANT in-plane
        # projector diag(1, 1, 0, 1, 0, 0): the residual is written in the
        # global frame, its in-plane rows are `rot(...) - deto` so they
        # differentiate to -I, and its transverse rows are the plane-stress
        # condition, which does not see `deto` at all. The DSL assumes -I on
        # all six, so the returned tangent is post-multiplied by P -- a
        # projector, not a rotation. The `rotateGradients` round trip that
        # built the old operator is gone with it.
        in_plane_operator = np.zeros((6, 6), dtype=float)
        in_plane_operator[_PLANE_STRESS_COMPONENTS, _PLANE_STRESS_COMPONENTS] = 1.0
        self._last_local_failure = None
        status, divisions = self._integrate_with_substepping(
            in_plane_kelvin, time_increment, predicted_transverse
        )
        if status != 1:
            self._last_local_failure = {
                "status": int(status),
                "substep_divisions": int(divisions),
            }
            self.revert()
            raise MFrontIntegrationError(
                f"GPS UMAT integration failed with status {status} after "
                f"sub-stepping down to 1/{divisions} of the increment"
            )
        # Route 1 -- rerun the full increment from the located root -- is gone.
        # It was measured to leave `manager.K` BIT-IDENTICAL, so it never did
        # what it was written for, and when its own attempt failed it redid the
        # entire sub-stepping: every sub-stepped increment paid for the
        # sub-stepping twice plus one doomed full attempt.
        self._integration_seconds += time.perf_counter() - started
        self._evaluate_calls += 1
        internal_state_variables = np.asarray(
            self._manager.s1.internal_state_variables
        ).copy()
        stress = np.asarray(self._manager.s1.thermodynamic_forces).copy()
        elastic = internal_state_variables[
            :, self._elastic_offset : self._elastic_offset + 6
        ].copy()
        tangent = np.asarray(self._manager.K).copy()
        # Output rotations: stress and elastic strain back to the global frame
        # ALWAYS (the law returns them in the material frame). The tangent is
        # rotated one-sided (its strain indices already are global) and
        # post-multiplied by the in-plane operator ONLY in the default mode:
        # with the law's CondensedTangent flag set, the returned tangent IS
        # the global plane-stress Schur of the raw law, computed inside the
        # local Newton -- no projection, no rotation, the in-plane block is
        # taken as is below.
        if self._mgis_rotations is not None:
            for tensor in (stress, elastic):
                flat = np.ascontiguousarray(tensor.reshape(-1))
                self._mgis.rotateThermodynamicForces(
                    flat, self._behaviour, self._mgis_rotations
                )
                tensor[:, :] = flat.reshape(self._point_count, 6)
        if not self._condensed_tangent:
            tangent = tangent @ in_plane_operator
            if self._mgis_rotations is not None:
                for column in range(6):
                    flat = np.ascontiguousarray(tangent[:, :, column].reshape(-1))
                    self._mgis.rotateThermodynamicForces(
                        flat, self._behaviour, self._mgis_rotations
                    )
                    tangent[:, :, column] = flat.reshape(self._point_count, 6)
        shadow_tangent_engineering = self._shadow_condensed_tangent(
            in_plane, internal_state_variables, time_increment
        )
        if self._condensed_tangent:
            shadow_tangent_engineering = None
        # Global total strain: read straight off the gradient, which now
        # carries the transverse strain too. There is no reconstruction from
        # the closure state variables any more, and with it goes the
        # engineering-versus-Kelvin ambiguity that reconstruction carried --
        # the gradient has one storage convention and MGIS owns it.
        total = np.asarray(self._manager.s1.gradients).copy()
        transverse = total[:, _TRANSVERSE_COMPONENTS_3D].copy()
        kinematic_defect = np.abs(
            elastic[:, :3].sum(axis=1) - total[:, :3].sum(axis=1)
        ) / np.maximum(np.abs(total[:, :3].sum(axis=1)), 1.0e-30)
        self._maximum_kinematic_defect = max(
            self._maximum_kinematic_defect, float(kinematic_defect.max())
        )
        observables = {
            name: internal_state_variables[:, item].copy()
            for name, item in self._observable_slices.items()
        }
        if "equivalent_plastic_slip" in observables:
            observables["accumulated_slip"] = observables["equivalent_plastic_slip"].sum(axis=1)
        total_tensor = kelvin_3d_to_tensor(total, quantity="strain")
        elastic_tensor = kelvin_3d_to_tensor(elastic, quantity="strain")
        stress_tensor = kelvin_3d_to_tensor(stress, quantity="stress")
        plastic_tensor = total_tensor - elastic_tensor
        condensed_tangent = tangent[:, _PLANE_STRESS_COMPONENTS][
            :, :, _PLANE_STRESS_COMPONENTS
        ].copy()
        in_tangent = condensed_tangent.copy()
        in_tangent = in_tangent * _KELVIN_TO_ENGINEERING_STRESS_SCALE[None, :, None]
        in_tangent = in_tangent * _ENGINEERING_TO_KELVIN_STRAIN_SCALE[None, None, :]
        if shadow_tangent_engineering is not None:
            in_tangent = shadow_tangent_engineering
        if self._composite_fd_enabled and np.any(self._last_substep_mask):
            if committed_snapshot is None:
                raise RuntimeError("composite-FD tangent requires a committed snapshot")
            composite_tangent = self._composite_fd_tangent(
                in_plane, time_increment, committed_snapshot
            )
            in_tangent[self._last_substep_mask] = composite_tangent[
                self._last_substep_mask
            ]
        in_stress = stress[:, _PLANE_STRESS_COMPONENTS] * _KELVIN_TO_ENGINEERING_STRESS_SCALE
        residual = np.stack(
            (stress_tensor[:, 2, 2], stress_tensor[:, 0, 2], stress_tensor[:, 1, 2]),
            axis=-1,
        )
        self._maximum_residual = max(
            self._maximum_residual, float(np.max(np.abs(residual)))
        )
        self._latest_in_plane = in_plane.copy()
        self._latest_dt = float(time_increment)
        self._latest_total_kelvin = total.copy()
        self._latest_transverse = transverse.copy()
        self._latest_cbb = tangent[:, _TRANSVERSE_COMPONENTS_3D][
            :, :, _TRANSVERSE_COMPONENTS_3D
        ].copy()
        self._latest_cba = tangent[:, _TRANSVERSE_COMPONENTS_3D][
            :, :, _PLANE_STRESS_COMPONENTS
        ].copy()
        # The law's internal Newton is not observable through stock MGIS; the
        # closure is part of it, so the reported count is the law's own
        # integration (one per point), documented as such.
        self._local_iterations[:] = 1
        self._internal_integrations += self._point_count
        self._latest_trial = ConstitutiveTrial(
            stress_in_plane_mpa=in_stress,
            tangent_in_plane_mpa=in_tangent if consistent_tangent else None,
            full_stress_tensor_mpa=stress_tensor,
            full_strain_tensor=total_tensor,
            elastic_strain_tensor=elastic_tensor,
            plastic_strain_tensor=plastic_tensor,
            plane_stress_residual_mpa=residual,
            observables=observables,
            local_plane_stress_iterations=self._local_iterations.astype(float),
        )
        return self._latest_trial

    def commit(self) -> None:
        self.accept_global_trial()
        self._mgis.update(self._manager)
        if self._latest_total_kelvin is not None:
            # Record what the closure actually moved, and against which
            # in-plane increment, so the next increment can start from it.
            delta = (
                self._latest_total_kelvin[:, _TRANSVERSE_COMPONENTS_3D]
                - self._committed_strain[:, _TRANSVERSE_COMPONENTS_3D]
            )
            in_plane_step = (
                self._latest_total_kelvin[:, _PLANE_STRESS_COMPONENTS]
                - self._committed_strain[:, _PLANE_STRESS_COMPONENTS]
            )
            self._previous_transverse_delta = delta.copy()
            self._previous_in_plane_norm = float(np.linalg.norm(in_plane_step))
            self._committed_strain[:, :] = self._latest_total_kelvin
            self._committed_in_plane = (
                self._latest_total_kelvin[:, _PLANE_STRESS_COMPONENTS]
                / _ENGINEERING_TO_KELVIN_STRAIN_SCALE
            ).copy()
        self._latest_trial = None
        self._latest_total_kelvin = None
        self._latest_transverse = None
        self._latest_cbb = None
        self._latest_cba = None

    def revert(self) -> None:
        self._mgis.revert(self._manager)
        self._latest_trial = None
        self._latest_total_kelvin = None
        self._latest_transverse = None
        self._latest_cbb = None
        self._latest_cba = None
        self._accepted_transverse = self._committed_strain[:, _TRANSVERSE_COMPONENTS_3D].copy()
        self._accepted_in_plane = None
        self._accepted_cbb = None
        self._accepted_cba = None
        self._warm_start_resets += 1

    def accept_global_trial(self) -> None:
        if self._latest_transverse is not None:
            self._accepted_transverse = self._latest_transverse.copy()
            self._accepted_in_plane = (
                None if self._latest_in_plane is None else self._latest_in_plane.copy()
            )
            self._accepted_cbb = None if self._latest_cbb is None else self._latest_cbb.copy()
            self._accepted_cba = None if self._latest_cba is None else self._latest_cba.copy()

    def snapshot_state(self) -> tuple[Any, ...]:
        return (
            np.asarray(self._manager.s0.gradients, dtype=float).copy(),
            np.asarray(self._manager.s0.internal_state_variables, dtype=float).copy(),
            np.asarray(self._manager.s0.thermodynamic_forces, dtype=float).copy(),
            self._committed_strain.copy(),
            self._accepted_transverse.copy(),
            None if self._accepted_in_plane is None else self._accepted_in_plane.copy(),
            None if self._accepted_cbb is None else self._accepted_cbb.copy(),
            None if self._accepted_cba is None else self._accepted_cba.copy(),
        )

    def restore_state(self, snapshot: Any) -> None:
        self.revert()
        for state in (self._manager.s0, self._manager.s1):
            state.gradients[:, :] = snapshot[0]
            state.internal_state_variables[:, :] = snapshot[1]
            state.thermodynamic_forces[:, :] = snapshot[2]
        self._committed_strain[:, :] = snapshot[3]
        self._accepted_transverse[:, :] = snapshot[4]
        self._accepted_in_plane = (
            None if snapshot[5] is None else np.asarray(snapshot[5]).copy()
        )
        self._accepted_cbb = None if snapshot[6] is None else np.asarray(snapshot[6]).copy()
        self._accepted_cba = None if snapshot[7] is None else np.asarray(snapshot[7]).copy()
        self._latest_trial = None
        self._latest_transverse = None
        self._latest_cbb = None
        self._latest_cba = None

    def complete_trial(self, trial: InPlaneConstitutiveTrial) -> ConstitutiveTrial:
        if self._latest_trial is None:
            raise RuntimeError("no native generalized plane-stress trial is available")
        return self._latest_trial
