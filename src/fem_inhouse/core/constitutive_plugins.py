"""Extension point for constitutive backends used by the global FEM solver.

The global Newton loop must only depend on :class:`PlaneStressMaterialBatch`.
This module lets an application register a material-batch builder without
adding another backend-specific branch to the solver.  In particular, a
crystal-plasticity MFront adapter can be introduced as a plugin while keeping
the current J2 implementations untouched.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from fem_inhouse.core.plane_stress_material import PlaneStressMaterialBatch

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")


@dataclass(frozen=True, slots=True, kw_only=True)
class PlaneStressMaterialRequest:
    """Complete, backend-neutral input for one material-batch construction."""

    initial_yield_stress_mpa: ArrayLike
    hardening_coefficient_mpa: ArrayLike
    hardening_exponent: float
    young_modulus_mpa: float
    poisson_ratio: float
    hardening_mode: str
    plastic_strain_max: float
    plastic_table_points: int
    first_positive_plastic_strain: float
    mfront_library: str
    mfront_threads: int
    local_plane_stress_options: Mapping[str, Any] = field(default_factory=dict)
    nonlocal_coupling_modulus_mpa: float | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "local_plane_stress_options",
            MappingProxyType(dict(self.local_plane_stress_options)),
        )
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


class PlaneStressMaterialPlugin(Protocol):
    """Builder contract for a constitutive implementation."""

    @property
    def identifier(self) -> str: ...

    def create_batch(
        self,
        request: PlaneStressMaterialRequest,
    ) -> PlaneStressMaterialBatch: ...


MaterialBatchBuilder = Callable[[PlaneStressMaterialRequest], "PlaneStressMaterialBatch"]


@dataclass(frozen=True, slots=True)
class CallablePlaneStressMaterialPlugin:
    """Small adapter making an ordinary builder function a material plugin."""

    identifier: str
    builder: MaterialBatchBuilder

    def __post_init__(self) -> None:
        _validate_identifier(self.identifier)

    def create_batch(
        self,
        request: PlaneStressMaterialRequest,
    ) -> PlaneStressMaterialBatch:
        return self.builder(request)


class ConstitutivePluginRegistry:
    """Explicit registry with duplicate protection and deterministic listing."""

    def __init__(self) -> None:
        self._plugins: dict[str, PlaneStressMaterialPlugin] = {}

    def register(
        self,
        plugin: PlaneStressMaterialPlugin,
        *,
        replace: bool = False,
    ) -> None:
        identifier = _validate_identifier(plugin.identifier)
        if identifier in self._plugins and not replace:
            raise ValueError(f"constitutive plugin {identifier!r} is already registered")
        self._plugins[identifier] = plugin

    def get(self, identifier: str) -> PlaneStressMaterialPlugin:
        try:
            return self._plugins[identifier]
        except KeyError as error:
            available = ", ".join(self.identifiers()) or "none"
            raise KeyError(
                f"unknown constitutive plugin {identifier!r}; available: {available}"
            ) from error

    def contains(self, identifier: str) -> bool:
        return identifier in self._plugins

    def identifiers(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))


def _validate_identifier(identifier: str) -> str:
    if not isinstance(identifier, str) or _IDENTIFIER.fullmatch(identifier) is None:
        raise ValueError(
            "plugin identifier must start with a lower-case letter and contain only "
            "lower-case letters, digits, '.', '_' or '-'"
        )
    return identifier


CONSTITUTIVE_PLUGINS = ConstitutivePluginRegistry()
_ENTRY_POINTS_LOADED = False


def register_constitutive_plugin(
    identifier: str,
    builder: MaterialBatchBuilder,
    *,
    replace: bool = False,
) -> None:
    """Register a process-local backend used by ``constitutive_backend``."""

    CONSTITUTIVE_PLUGINS.register(
        CallablePlaneStressMaterialPlugin(identifier, builder),
        replace=replace,
    )


def load_constitutive_plugins() -> None:
    """Load installed plugins from ``fem_inhouse.constitutive_plugins`` once."""

    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    for entry_point in entry_points(group="fem_inhouse.constitutive_plugins"):
        loaded = entry_point.load()
        if hasattr(loaded, "identifier") and hasattr(loaded, "create_batch"):
            CONSTITUTIVE_PLUGINS.register(loaded)
        elif callable(loaded):
            register_constitutive_plugin(entry_point.name, loaded)
        else:
            raise TypeError(
                f"constitutive entry point {entry_point.name!r} must expose a "
                "plugin or builder"
            )
    _ENTRY_POINTS_LOADED = True
