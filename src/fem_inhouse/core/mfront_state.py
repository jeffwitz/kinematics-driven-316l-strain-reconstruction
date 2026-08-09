"""Transactional state and public timing records for MFront bridges."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class MFrontMaterialStateSnapshot:
    """Committed MGIS state captured for an isolated adaptive branch."""

    gradients_s0: NDArray
    internal_state_variables_s0: NDArray
    thermodynamic_forces_s0: NDArray
    committed_global_strain: NDArray
    committed_nonlocal_values: NDArray | None


@dataclass(frozen=True, slots=True)
class MFrontCondensedStateSnapshot:
    """Committed state of the 3-D bridge and plane-stress predictor."""

    bridge: MFrontMaterialStateSnapshot
    accepted_transverse: NDArray
    latest_transverse: NDArray | None
    has_accepted_global_trial: bool
    last_in_plane: NDArray | None
    last_time_increment: float | None
    accepted_in_plane: NDArray | None = None
    accepted_cbb: NDArray | None = None
    accepted_cba: NDArray | None = None


@dataclass(frozen=True, slots=True)
class MFrontCondensedBlocksStateSnapshot:
    """Committed state of a block-partitioned condensed material."""

    blocks: tuple[MFrontCondensedStateSnapshot, ...]


@dataclass(frozen=True, slots=True)
class MFrontTimingStatistics:
    """Accumulated wall times for the native MGIS constitutive bridge."""

    integration_without_tangent_seconds: float = 0.0
    integration_with_tangent_seconds: float = 0.0
    kelvin_conversion_seconds: float = 0.0
    tensor_reconstruction_seconds: float = 0.0
    integration_without_tangent_calls: int = 0
    integration_with_tangent_calls: int = 0
    tensor_reconstruction_calls: int = 0
    rotation_to_material_seconds: float = 0.0
    integration_seconds: float = 0.0
    rotation_to_global_seconds: float = 0.0
    condensation_seconds: float = 0.0
    condition_check_seconds: float = 0.0
    local_solve_seconds: float = 0.0
    reconstruction_seconds: float = 0.0
    observable_seconds: float = 0.0
    evaluate_calls: int = 0
    condition_checks: int = 0
    local_condensation_evaluations: int = 0
    full_batch_integration_calls: int = 0
    equivalent_active_point_integrations: int = 0
    material_point_integrations: int = 0
    material_point_integrations_with_tangent: int = 0
    material_point_integrations_without_tangent: int = 0
    material_block_integration_calls: int = 0
    material_block_count: int = 1
    native_batch_calls: int = 0
    native_material_points: int = 0
    native_internal_integrations: int = 0
    native_integrate_calls: int = 0
    native_integrate_points: int = 0
    native_total_local_iterations: int = 0
    native_thread_count: int = 1
    #: Points that refused the full step and were sub-stepped individually, and
    #: how often the failing-index cache spared the bisection. Zero on every
    #: backend that does not sub-step.
    native_substep_points: int = 0
    native_substep_cache_hits: int = 0
    native_substep_cache_misses: int = 0
    composite_fd_seconds: float = 0.0
    composite_fd_points: int = 0
    composite_fd_trajectories: int = 0
    composite_fd_partition_changes: int = 0
    composite_fd_mgis_calls: int = 0
    composite_fd_actual_point_integrations: int = 0
    composite_fd_snapshot_seconds: float = 0.0
    composite_fd_restore_seconds: float = 0.0
    composite_fd_integration_seconds: float = 0.0
    composite_fd_other_seconds: float = 0.0
    local_iteration_histogram: tuple[int, ...] = ()
