"""Low-level MGIS runtime helpers and Kelvin conversions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.mfront_behaviours import MFrontBehaviourSpec
from fem_inhouse.core.plane_stress_material import ConstitutiveIntegrationError
from fem_inhouse.core.tensor_reconstruction import (
    kelvin_plane_stress_to_tensor,
    tensor_to_engineering_strain_2d,
    tensor_to_engineering_stress_2d,
)

_SQRT_TWO = np.sqrt(2.0)
_PLANE_STRESS_COMPONENTS = np.array([0, 1, 3])
_KELVIN_TO_ENGINEERING_STRESS_SCALE = np.array([1.0, 1.0, 1.0 / _SQRT_TWO])
_ENGINEERING_TO_KELVIN_STRAIN_SCALE = np.array([1.0, 1.0, 1.0 / _SQRT_TWO])


class MFrontUnavailableError(RuntimeError):
    """Raised when the optional TFEL/MGIS runtime cannot be loaded."""


class MFrontIntegrationError(ConstitutiveIntegrationError):
    """Raised when MFront fails to integrate a material-point batch."""


@dataclass(frozen=True, slots=True)
class MFrontIntegrationResult:
    """Engineering-component result of one MFront material-point evaluation."""

    stress_mpa: NDArray
    plastic_strain: NDArray
    equivalent_plastic_strain: NDArray
    yield_surface_radius_mpa: NDArray
    consistent_tangent_mpa: NDArray | None



def engineering_strain_to_kelvin(
    strain: ArrayLike,
    *,
    out: NDArray | None = None,
) -> NDArray:
    """Convert ``[e11, e22, gamma12]`` to MFront's 2D Kelvin stensor."""

    values = np.asarray(strain, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("engineering strain must have trailing dimension 3")
    expected_shape = (*values.shape[:-1], 4)
    result = np.empty(expected_shape, dtype=float) if out is None else out
    if result.shape != expected_shape:
        raise ValueError(f"out must have shape {expected_shape}")
    result[..., 0] = values[..., 0]
    result[..., 1] = values[..., 1]
    result[..., 2] = 0.0
    result[..., 3] = values[..., 2] / _SQRT_TWO
    return result


def kelvin_strain_to_engineering(strain: ArrayLike) -> NDArray:
    """Convert a MFront 2D Kelvin strain to ``[e11, e22, gamma12]``."""

    return tensor_to_engineering_strain_2d(kelvin_plane_stress_to_tensor(strain, quantity="strain"))


def kelvin_stress_to_engineering(stress: ArrayLike) -> NDArray:
    """Convert a MFront 2D Kelvin stress to ``[s11, s22, s12]``."""

    return tensor_to_engineering_stress_2d(kelvin_plane_stress_to_tensor(stress, quantity="stress"))


def kelvin_tangent_to_engineering(tangent: ArrayLike) -> NDArray:
    """Convert a 4x4 Kelvin tangent to engineering plane-stress components."""

    values = np.asarray(tangent, dtype=float)
    if values.ndim < 2 or values.shape[-2:] != (4, 4):
        raise ValueError("Kelvin tangent must have trailing dimensions (4, 4)")
    selected = np.take(values, _PLANE_STRESS_COMPONENTS, axis=-2)
    selected = np.take(selected, _PLANE_STRESS_COMPONENTS, axis=-1)
    return (
        selected
        * _KELVIN_TO_ENGINEERING_STRESS_SCALE[:, None]
        * _ENGINEERING_TO_KELVIN_STRAIN_SCALE[None, :]
    )


def _load_mgis() -> Any:
    try:
        return import_module("mgis.behaviour")
    except (ImportError, OSError) as error:
        # The experimental generalized-plane-stress binding is intentionally
        # shipped as a standalone extension while it is being qualified.  It
        # exposes the same behaviour API and is accepted only as an explicit
        # opt-in backend below.
        try:
            return import_module("behaviour")
        except (ImportError, OSError):
            raise MFrontUnavailableError(
                "MGIS Python bindings are unavailable; source the TFEL environment "
                "before starting Python"
            ) from error


def _load_mgis_root() -> Any:
    try:
        return import_module("mgis")
    except (ImportError, OSError) as error:
        try:
            return import_module("behaviour")
        except (ImportError, OSError):
            raise MFrontUnavailableError(
                "MGIS Python bindings are unavailable; source the TFEL environment "
                "before starting Python"
            ) from error


def _broadcast_material_properties(
    initial_yield_stress_mpa: ArrayLike,
    hardening_coefficient_mpa: ArrayLike,
    hardening_exponent: ArrayLike,
) -> tuple[NDArray, NDArray, NDArray]:
    values = np.broadcast_arrays(
        np.asarray(initial_yield_stress_mpa, dtype=float),
        np.asarray(hardening_coefficient_mpa, dtype=float),
        np.asarray(hardening_exponent, dtype=float),
    )
    flattened = tuple(np.ravel(value).copy() for value in values)
    yield_stress, coefficient, exponent = flattened
    if yield_stress.size == 0:
        raise ValueError("at least one material point is required")
    if not all(np.isfinite(value).all() for value in flattened):
        raise ValueError("MFront material properties must be finite")
    if np.any(yield_stress <= 0):
        raise ValueError("initial yield stress must be positive")
    if np.any(coefficient < 0):
        raise ValueError("hardening coefficient must be non-negative")
    if np.any(exponent <= 0):
        raise ValueError("hardening exponent must be positive")
    return yield_stress, coefficient, exponent


def _broadcast_point_property(
    values: ArrayLike,
    point_count: int,
    *,
    name: str,
    nonnegative: bool = False,
) -> NDArray:
    """Broadcast one scalar material-point property to external storage."""

    array = np.asarray(values, dtype=float)
    try:
        broadcast = np.broadcast_to(array, (point_count,)).copy()
    except ValueError as error:
        raise ValueError(f"{name} must be scalar or have shape {(point_count,)}") from error
    if not np.isfinite(broadcast).all():
        raise ValueError(f"{name} must be finite")
    if nonnegative and np.any(broadcast < 0):
        raise ValueError(f"{name} must be nonnegative")
    return broadcast


def _declared_internal_slices(
    mgis: Any,
    behaviour: Any,
    hypothesis: Any,
    specification: MFrontBehaviourSpec | None,
) -> dict[str, slice]:
    """Locate each declared internal-state family in the flat MGIS array.

    The offset is resolved from the first member of the family and the extent
    from the declared `component_count`, checked against the sizes MGIS
    reports. Names such as `PlasticSlip[7]` are never parsed: MFront indexes
    array variables with a suffix, but relying on that spelling would break on
    any behaviour that names its members differently.
    """

    if specification is None:
        return {}
    sizes = {
        variable.name: int(mgis.getVariableSize(variable, hypothesis))
        for variable in behaviour.isvs
    }
    offsets: dict[str, int] = {}
    running = 0
    for variable in behaviour.isvs:
        offsets[variable.name] = running
        running += sizes[variable.name]

    slices: dict[str, slice] = {}
    for declared in specification.internal_state_variables:
        head = declared.entry_name
        if head not in offsets:
            # MFront suffixes the members of an array variable.
            head = f"{declared.entry_name}[0]"
        if head not in offsets:
            if declared.required:
                raise MFrontUnavailableError(
                    f"MFront behaviour does not expose internal state "
                    f"{declared.entry_name!r} declared by the catalogue"
                )
            continue
        start = offsets[head]
        stop = start + declared.component_count
        if stop > running:
            raise MFrontUnavailableError(
                f"internal state {declared.entry_name!r} declares "
                f"{declared.component_count} components but only {running - start} "
                "remain in the behaviour"
            )
        slices[declared.canonical_name] = slice(start, stop)
    return slices


def _apply_behaviour_parameters(
    mgis: Any,
    behaviour: Any,
    values: Mapping[str, float] | None,
    behaviour_name: str,
) -> None:
    """Override MFront `@Parameter` values on a loaded behaviour.

    This is how a registered parameter set reaches the law without editing and
    recompiling the `.mfront` source.

    **`mgis.load` does not give you a private behaviour.** Two `load` calls for
    the same library, name and hypothesis return handles onto the *same*
    underlying object, so a `setParameter` through one is visible through the
    other, process-wide. Measured, not assumed: setting `tau0 = 999` on one
    handle moved the plateau of a run driven through a second, untouched handle.

    The consequence is that applying a set once at construction is not enough.
    Two batches with different sets alive in the same process would silently
    share whichever set was applied last -- and the wrong one would be the
    plausible-looking one. `_reassert_behaviour_parameters` therefore re-applies
    the batch's own values immediately before every integration, which is exact
    under any interleaving and costs a handful of `setParameter` calls against
    the integration of every point in the batch.

    An unknown name is refused rather than ignored. MGIS `setParameter` on a
    name the behaviour does not export raises, but only when it is called, and
    a silently mistyped parameter would otherwise look like a run that simply
    used the defaults -- which is exactly the failure that must not be possible
    to publish. The check happens here, before the first integration.
    """

    if not values:
        return
    declared = set(behaviour.parameters)
    unknown = sorted(name for name in values if name not in declared)
    if unknown:
        raise ValueError(
            f"MFront behaviour {behaviour_name!r} does not expose the parameter(s) "
            f"{', '.join(unknown)}; it exposes: {', '.join(sorted(declared))}"
        )
    for name, value in values.items():
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"parameter {name!r} must be finite, got {value!r}")
        mgis.setParameter(behaviour, name, numeric)


def _variable_offset(
    mgis: Any,
    variables: Any,
    name: str,
    hypothesis: Any,
    *,
    expected_size: int,
    required: bool = True,
) -> int | None:
    matches = [variable for variable in variables if variable.name == name]
    if not matches:
        if required:
            raise MFrontUnavailableError(
                f"MFront behaviour does not expose required variable {name!r}"
            )
        return None
    if len(matches) != 1:
        raise MFrontUnavailableError(f"MFront behaviour exposes {name!r} more than once")
    actual_size = int(mgis.getVariableSize(matches[0], hypothesis))
    if actual_size != expected_size:
        raise MFrontUnavailableError(
            f"MFront variable {name!r} has size {actual_size}, expected {expected_size}"
        )
    return int(mgis.getVariableOffset(variables, name, hypothesis))
