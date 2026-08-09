"""Internal counters for GPS integration layers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GPSSubstepCounters:
    uses: int = 0
    points: int = 0
    divisions_max: int = 1
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass(slots=True)
class CompositeTangentCounters:
    seconds: float = 0.0
    points: int = 0
    trajectories: int = 0
    partition_changes: int = 0
    mgis_calls: int = 0
    actual_point_integrations: int = 0
    snapshot_seconds: float = 0.0
    restore_seconds: float = 0.0
    integration_seconds: float = 0.0


@dataclass(slots=True)
class GPSDiagnosticsCounters:
    shadow_failures: int = 0
    maximum_kinematic_defect: float = 0.0
    last_shadow_diagnostics: dict[str, object] | None = None
    last_composite_fd_diagnostics: dict[str, object] | None = None
