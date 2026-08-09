"""Composite tangent construction for the GPS sub-stepping adapter."""

from __future__ import annotations

# mypy: ignore-errors
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

_PLANE_STRESS_COMPONENTS = np.array([0, 1, 3])


class CompositeTangentMixin:
    def _composite_fd_material(self, point: int) -> Any:
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
            self._composite_fd_counters.snapshot_seconds += time.perf_counter() - snapshot_started
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
                self._composite_fd_counters.restore_seconds += time.perf_counter() - restore_started
                before = point_material.timing_statistics
                trial_plus = point_material.evaluate(
                    plus, time_increment=time_increment, consistent_tangent=True
                )
                after = point_material.timing_statistics
                self._composite_fd_counters.mgis_calls += max(
                    0,
                    after.native_batch_calls - before.native_batch_calls,
                )
                self._composite_fd_counters.actual_point_integrations += max(
                    0,
                    after.native_internal_integrations
                    - before.native_internal_integrations,
                )
                self._composite_fd_counters.integration_seconds += max(
                    0.0,
                    after.integration_seconds - before.integration_seconds,
                )
                plus_partition = bool(point_material.last_substep_mask[0])
                restore_started = time.perf_counter()
                point_material.restore_state(point_snapshot)
                self._composite_fd_counters.restore_seconds += time.perf_counter() - restore_started
                before = point_material.timing_statistics
                trial_minus = point_material.evaluate(
                    minus, time_increment=time_increment, consistent_tangent=True
                )
                after = point_material.timing_statistics
                self._composite_fd_counters.mgis_calls += max(
                    0,
                    after.native_batch_calls - before.native_batch_calls,
                )
                self._composite_fd_counters.actual_point_integrations += max(
                    0,
                    after.native_internal_integrations
                    - before.native_internal_integrations,
                )
                self._composite_fd_counters.integration_seconds += max(
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
            self._composite_fd_counters.trajectories += 6
            if partition_changed:
                self._composite_fd_counters.partition_changes += 1
            diagnostics.append(
                {
                    "point": int(point),
                    "divisions": int(self._last_substep_divisions[point]),
                    "partition_unchanged": not partition_changed,
                    "tangent": tangent.copy(),
                }
            )
        self._composite_fd_counters.seconds += time.perf_counter() - started
        self._composite_fd_counters.points += len(active_points)
        self._gps_diagnostics_counters.last_composite_fd_diagnostics = {
            "points": diagnostics,
            "step": self._composite_fd_step,
        }
        return result
