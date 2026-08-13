"""Pluggable scalar nonlocal criteria for the staggered constitutive solve."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.plane_stress_material import InPlaneConstitutiveTrial
from fem_inhouse.postprocessing.helmholtz import helmholtz_filter_element_field

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class NonlocalRegularisationContext:
    """Geometry and numerical controls supplied to a spatial operator."""

    length_scale_mm: float
    spacing_x_mm: float
    spacing_y_mm: float


@dataclass(frozen=True, slots=True)
class NonlocalRegularisationResult:
    """Operator-neutral result required by the fixed-point driver."""

    filtered_element_field: FloatArray
    residual_relative: float
    mean_drift: float


class ScalarNonlocalCriterion(Protocol):
    """One scalar source, constitutive feedback field and spatial operator."""

    @property
    def identifier(self) -> str: ...

    @property
    def source_name(self) -> str: ...

    @property
    def requires_nonnegative_field(self) -> bool: ...

    def supports_material(self, material_batch: object) -> bool: ...

    def set_external_field(self, material_batch: object, values: ArrayLike) -> None: ...

    def evaluate_source_and_safety(
        self,
        material_batch: object,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
    ) -> tuple[FloatArray, FloatArray]: ...

    def source_from_trial(self, trial: InPlaneConstitutiveTrial) -> FloatArray: ...

    def safety_from_trial(self, trial: InPlaneConstitutiveTrial) -> FloatArray: ...

    def regularise(
        self,
        source_element_field: ArrayLike,
        context: NonlocalRegularisationContext,
    ) -> NonlocalRegularisationResult: ...


@dataclass(frozen=True, slots=True)
class EquivalentPlasticStrainHelmholtzCriterion:
    """Current scalar PEEQ micromorphic model behind the generic contract."""

    identifier: str = "peeq_helmholtz"
    source_name: str = "equivalent_plastic_strain"
    requires_nonnegative_field: bool = True

    def supports_material(self, material_batch: object) -> bool:
        return all(
            hasattr(material_batch, name)
            for name in (
                "set_nonlocal_equivalent_plastic_strain",
                "evaluate_nonlocal_state",
                "evaluate_in_plane",
            )
        )

    def set_external_field(self, material_batch: object, values: ArrayLike) -> None:
        cast(Any, material_batch).set_nonlocal_equivalent_plastic_strain(values)

    def evaluate_source_and_safety(
        self,
        material_batch: object,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
    ) -> tuple[FloatArray, FloatArray]:
        source, safety = cast(Any, material_batch).evaluate_nonlocal_state(
            in_plane_strain,
            time_increment=time_increment,
        )
        return np.asarray(source, dtype=np.float64), np.asarray(safety, dtype=np.float64)

    def source_from_trial(self, trial: InPlaneConstitutiveTrial) -> FloatArray:
        return np.asarray(trial.observables[self.source_name], dtype=np.float64)

    def safety_from_trial(self, trial: InPlaneConstitutiveTrial) -> FloatArray:
        return np.asarray(
            trial.observables["yield_surface_radius_mpa"],
            dtype=np.float64,
        )

    def regularise(
        self,
        source_element_field: ArrayLike,
        context: NonlocalRegularisationContext,
    ) -> NonlocalRegularisationResult:
        result = helmholtz_filter_element_field(
            source_element_field,
            length_scale_mm=context.length_scale_mm,
            spacing_x_mm=context.spacing_x_mm,
            spacing_y_mm=context.spacing_y_mm,
        )
        return NonlocalRegularisationResult(
            filtered_element_field=result.filtered_element_field,
            residual_relative=result.residual_relative,
            mean_drift=result.mean_drift,
        )


@dataclass(frozen=True, slots=True)
class AccumulatedSlipHelmholtzCriterion:
    """Scalar SRIX source ``Gamma=sum_s EquivalentPlasticSlip_s``.

    This is a numerical transposition of the J2 scalar architecture.  It does
    not assign the name PEEQ to a crystal quantity and it deliberately keeps
    the local SRIX parameters separate from the micromorphic coupling.
    """

    identifier: str = "accumulated_slip_helmholtz"
    source_name: str = "accumulated_slip"
    requires_nonnegative_field: bool = True

    def supports_material(self, material_batch: object) -> bool:
        return all(
            hasattr(material_batch, name)
            for name in (
                "set_nonlocal_equivalent_plastic_strain",
                "evaluate_nonlocal_state",
                "evaluate_in_plane",
            )
        )

    def set_external_field(self, material_batch: object, values: ArrayLike) -> None:
        cast(Any, material_batch).set_nonlocal_equivalent_plastic_strain(values)

    def evaluate_source_and_safety(
        self,
        material_batch: object,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
    ) -> tuple[FloatArray, FloatArray]:
        source, safety = cast(Any, material_batch).evaluate_nonlocal_state(
            in_plane_strain,
            time_increment=time_increment,
        )
        return np.asarray(source, dtype=np.float64), np.asarray(safety, dtype=np.float64)

    def source_from_trial(self, trial: InPlaneConstitutiveTrial) -> FloatArray:
        return np.asarray(trial.observables[self.source_name], dtype=np.float64)

    def safety_from_trial(self, trial: InPlaneConstitutiveTrial) -> FloatArray:
        return np.ones_like(self.source_from_trial(trial))

    def regularise(
        self,
        source_element_field: ArrayLike,
        context: NonlocalRegularisationContext,
    ) -> NonlocalRegularisationResult:
        result = helmholtz_filter_element_field(
            source_element_field,
            length_scale_mm=context.length_scale_mm,
            spacing_x_mm=context.spacing_x_mm,
            spacing_y_mm=context.spacing_y_mm,
        )
        return NonlocalRegularisationResult(
            filtered_element_field=result.filtered_element_field,
            residual_relative=result.residual_relative,
            mean_drift=result.mean_drift,
        )


NonlocalCriterionFactory = Callable[[Mapping[str, Any]], ScalarNonlocalCriterion]


class NonlocalCriterionRegistry:
    """Registry of criterion factories, including criterion-specific options."""

    def __init__(self) -> None:
        self._factories: dict[str, NonlocalCriterionFactory] = {}

    def register(
        self,
        identifier: str,
        factory: NonlocalCriterionFactory,
        *,
        replace: bool = False,
    ) -> None:
        if not identifier:
            raise ValueError("nonlocal criterion identifier must not be empty")
        if identifier in self._factories and not replace:
            raise ValueError(f"nonlocal criterion {identifier!r} is already registered")
        self._factories[identifier] = factory

    def create(
        self,
        identifier: str,
        options: Mapping[str, Any] | None = None,
    ) -> ScalarNonlocalCriterion:
        try:
            factory = self._factories[identifier]
        except KeyError as error:
            available = ", ".join(self.identifiers()) or "none"
            raise KeyError(
                f"unknown nonlocal criterion {identifier!r}; available: {available}"
            ) from error
        criterion = factory({} if options is None else dict(options))
        if criterion.identifier != identifier:
            raise ValueError(
                f"criterion factory {identifier!r} returned {criterion.identifier!r}"
            )
        return criterion

    def identifiers(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


NONLOCAL_CRITERIA = NonlocalCriterionRegistry()
_ENTRY_POINTS_LOADED = False


def _create_peeq_helmholtz(options: Mapping[str, Any]) -> ScalarNonlocalCriterion:
    if options:
        unexpected = ", ".join(sorted(options))
        raise ValueError(f"peeq_helmholtz does not accept criterion options: {unexpected}")
    return EquivalentPlasticStrainHelmholtzCriterion()


NONLOCAL_CRITERIA.register("peeq_helmholtz", _create_peeq_helmholtz)


def _create_accumulated_slip_helmholtz(
    options: Mapping[str, Any],
) -> ScalarNonlocalCriterion:
    if options:
        unexpected = ", ".join(sorted(options))
        raise ValueError(
            "accumulated_slip_helmholtz does not accept criterion options: "
            f"{unexpected}"
        )
    return AccumulatedSlipHelmholtzCriterion()


NONLOCAL_CRITERIA.register(
    "accumulated_slip_helmholtz",
    _create_accumulated_slip_helmholtz,
)


def load_nonlocal_criteria() -> None:
    """Load installed factories from ``fem_inhouse.nonlocal_criteria`` once."""

    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    for entry_point in entry_points(group="fem_inhouse.nonlocal_criteria"):
        loaded = entry_point.load()
        if not callable(loaded):
            raise TypeError(
                f"nonlocal entry point {entry_point.name!r} must be a factory"
            )
        NONLOCAL_CRITERIA.register(entry_point.name, loaded)
    _ENTRY_POINTS_LOADED = True
