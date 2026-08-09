"""Sub-stepping policy for the GPS adapter."""

from __future__ import annotations

# mypy: ignore-errors
import numpy as np
from numpy.typing import NDArray

from fem_inhouse.core.mfront_runtime import _ENGINEERING_TO_KELVIN_STRAIN_SCALE

_PLANE_STRESS_COMPONENTS = np.array([0, 1, 3])
_TRANSVERSE_COMPONENTS_3D = np.array([2, 4, 5])


class GPSSubsteppingMixin:
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
            self._substep_counters.cache_hits += 1
            self._substep_counters.uses += 1
            self._substep_counters.points += sum(end - begin for begin, end in advanced)
            self._substep_counters.divisions_max = max(
                self._substep_counters.divisions_max, worst
            )
            return 1, worst
        self._substep_counters.cache_misses += 1
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
        self._gps_diagnostics_counters.last_shadow_diagnostics = None
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
            self._substep_counters.uses += 1
            return 1, 1
        self._substep_counters.points += sum(end - begin for begin, end in spans)
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
        self._substep_counters.uses += 1
        self._substep_counters.divisions_max = max(
            self._substep_counters.divisions_max, worst
        )
        self._failing_cache = spans if len(spans) <= self._maximum_cached_spans else []
        return 1, worst
