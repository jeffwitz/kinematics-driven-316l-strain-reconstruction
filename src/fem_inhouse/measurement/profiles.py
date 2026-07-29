"""Versioned DISFlow profiles and their provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .disflow import DISFlowConfig, query_disflow_configuration


@dataclass(frozen=True, slots=True)
class DISFlowProfile:
    """Named settings with an explicit historical-certainty boundary."""

    name: str
    config: DISFlowConfig
    source: str

    def manifest(self) -> dict[str, Any]:
        requested = self.config.as_dict()
        return {
            "name": self.name,
            "source": self.source,
            "requested": requested,
            "explicitly_set": sorted(name for name, value in requested.items() if value is not None),
            "left_to_factory": sorted(name for name, value in requested.items() if value is None),
            "queried_opencv": query_disflow_configuration(self.config),
        }


_PROFILES = {
    "declared_medium_v4": DISFlowProfile(
        name="declared_medium_v4",
        source="pre-registered OpenCV 4.14 V4 reproduction profile",
        config=DISFlowConfig(),
    ),
    "legacy_script_2021": DISFlowProfile(
        name="legacy_script_2021",
        source="setters explicitly present in references/legacy_dic/dic_displacement_fields.py",
        config=DISFlowConfig(
            preset=None,
            finest_scale=0,
            gradient_descent_iterations=None,
            patch_size=4,
            patch_stride=1,
            use_mean_normalization=None,
            use_spatial_propagation=None,
            variational_refinement_alpha=100.0,
            variational_refinement_delta=1.0,
            variational_refinement_gamma=0.0,
            variational_refinement_epsilon=0.002,
            variational_refinement_iterations=30,
        ),
    ),
}


def disflow_profile(name: str) -> DISFlowProfile:
    """Return a named immutable profile."""

    try:
        return _PROFILES[name]
    except KeyError as error:
        raise ValueError(
            f"unknown DISFlow profile {name!r}; expected one of {sorted(_PROFILES)}"
        ) from error


def disflow_profile_names() -> tuple[str, ...]:
    """Return stable profile names."""

    return tuple(sorted(_PROFILES))
