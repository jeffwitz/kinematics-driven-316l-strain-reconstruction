"""Common contract for two-dimensional plane-stress mechanics solvers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.constitutive import (
    PLANE_STRESS_VON_MISES_METRIC,
    make_hardening,
    return_mapping,
)
from fem_inhouse.core.constitutive import (
    consistent_tangent as python_consistent_tangent,
)
from fem_inhouse.core.element import plane_stress_elasticity
from fem_inhouse.core.linear_solver import LinearSystemMatrixType
from fem_inhouse.core.tensor_reconstruction import reconstruct_python_plane_stress_state

FloatArray = NDArray[np.float64]
ResponseLevel = Literal["residual", "tangent", "complete"]
SYMMETRIC_TANGENT_RELATIVE_TOLERANCE = 1e-12


def relative_tangent_asymmetry(tangent: ArrayLike) -> float:
    """Return the maximum skew part relative to the tangent amplitude."""

    values = np.asarray(tangent, dtype=np.float64)
    if values.ndim < 2 or values.shape[-2:] != (3, 3):
        raise ValueError("tangent must have trailing dimensions (3, 3)")
    if not np.isfinite(values).all():
        raise ValueError("tangent must be finite")
    scale = max(float(np.max(np.abs(values))), 1.0)
    maximum_skew = max(
        float(np.max(np.abs(values[..., 0, 1] - values[..., 1, 0]))),
        float(np.max(np.abs(values[..., 0, 2] - values[..., 2, 0]))),
        float(np.max(np.abs(values[..., 1, 2] - values[..., 2, 1]))),
    )
    return maximum_skew / scale


class ConstitutiveIntegrationError(RuntimeError):
    """A constitutive trial failed and must not be committed.

    Raisers attach a `diagnostics` mapping describing what the point looked
    like when it failed; the failure-diagnostic archives read it back. It is
    declared here rather than set ad hoc on the instance so that a consumer can
    rely on it existing.
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.diagnostics: dict[str, Any] = {}


class LocalPlaneStressConvergenceError(ConstitutiveIntegrationError):
    """The local elimination of transverse strains did not converge."""


@dataclass(frozen=True, slots=True, kw_only=True)
class InPlaneConstitutiveTrial:
    """Light non-committed response required by the global Newton loop."""

    stress_in_plane_mpa: FloatArray
    tangent_in_plane_mpa: FloatArray | None
    observables: dict[str, FloatArray] = field(default_factory=dict)
    local_plane_stress_iterations: FloatArray | None = None
    cbb_condition_number: FloatArray | None = None


def evaluate_in_plane_response(
    material: PlaneStressMaterialBatch,
    in_plane_strain: ArrayLike,
    *,
    time_increment: float,
    response_level: ResponseLevel,
    consistent_tangent: bool = True,
) -> InPlaneConstitutiveTrial:
    """Evaluate the lightest response supported by a material backend."""

    evaluator = getattr(material, "evaluate_in_plane_response", None)
    if callable(evaluator):
        return evaluator(
            in_plane_strain,
            time_increment=time_increment,
            response_level=response_level,
            consistent_tangent=consistent_tangent,
        )
    return material.evaluate_in_plane(
        in_plane_strain,
        time_increment=time_increment,
        consistent_tangent=consistent_tangent,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ConstitutiveTrial(InPlaneConstitutiveTrial):
    """In-plane response enriched with the complete three-dimensional state."""

    full_stress_tensor_mpa: FloatArray
    full_strain_tensor: FloatArray
    elastic_strain_tensor: FloatArray
    plastic_strain_tensor: FloatArray
    plane_stress_residual_mpa: FloatArray


@dataclass(frozen=True, slots=True)
class PlaneStressBatchStatistics:
    """Accumulated diagnostics for local plane-stress enforcement."""

    maximum_gauss_point_plane_stress_residual_mpa: float = 0.0
    maximum_local_plane_stress_iterations: int = 0
    mean_local_plane_stress_iterations: float = 0.0
    local_plane_stress_failures: int = 0
    maximum_cbb_condition_number: float = 0.0


@runtime_checkable
class PlaneStressMaterialBatch(Protocol):
    """Transaction-safe material integration contract seen by global Newton."""

    @property
    def point_count(self) -> int: ...

    @property
    def backend_name(self) -> str: ...

    @property
    def completion_strategy(self) -> str: ...

    @property
    def linear_system_matrix_type(self) -> LinearSystemMatrixType: ...

    @property
    def statistics(self) -> PlaneStressBatchStatistics: ...

    def evaluate(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> ConstitutiveTrial: ...

    def evaluate_in_plane(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> InPlaneConstitutiveTrial: ...

    def complete_trial(self, trial: InPlaneConstitutiveTrial) -> ConstitutiveTrial: ...

    def commit(self) -> None: ...

    def revert(self) -> None: ...


@runtime_checkable
class HookeanPlaneStressMaterialBatch(PlaneStressMaterialBatch, Protocol):
    """Plane-stress material with an immutable elastic condensed tangent."""

    @property
    def elastic_tangent_in_plane_mpa(self) -> FloatArray:
        """Return fixed per-state tangents with shape ``(point_count, 3, 3)``."""


class CachedHookeanPlaneStressMaterialBatch:
    """Cache the unloaded elastic tangent of a transactional material batch."""

    def __init__(self, material: PlaneStressMaterialBatch) -> None:
        self._material = material
        trial = material.evaluate_in_plane(
            np.zeros((material.point_count, 3), dtype=np.float64),
            time_increment=1.0,
            consistent_tangent=True,
        )
        material.revert()
        if trial.tangent_in_plane_mpa is None:
            raise ValueError("Hookean EBI requires an unloaded elastic tangent")
        if np.max(np.abs(trial.stress_in_plane_mpa)) > 1.0e-10:
            raise ValueError("Hookean EBI tangent cache requires an unloaded state")
        tangent = np.asarray(trial.tangent_in_plane_mpa, dtype=np.float64)
        self._elastic_tangent = tangent.reshape(material.point_count, 3, 3).copy()
        self._elastic_tangent.setflags(write=False)
        full_tangent_probe = getattr(material, "reference_full_tangent_kelvin_mpa", None)
        self._elastic_tangent_3d = (
            np.asarray(full_tangent_probe(), dtype=np.float64).copy()
            if callable(full_tangent_probe)
            else None
        )
        if self._elastic_tangent_3d is not None:
            self._elastic_tangent_3d.setflags(write=False)

    @property
    def point_count(self) -> int:
        return self._material.point_count

    @property
    def backend_name(self) -> str:
        return self._material.backend_name

    @property
    def completion_strategy(self) -> str:
        return self._material.completion_strategy

    @property
    def linear_system_matrix_type(self) -> LinearSystemMatrixType:
        return self._material.linear_system_matrix_type

    @property
    def statistics(self) -> PlaneStressBatchStatistics:
        return self._material.statistics

    @property
    def elastic_tangent_in_plane_mpa(self) -> FloatArray:
        return self._elastic_tangent

    @property
    def elastic_tangent_3d_kelvin_mpa(self) -> FloatArray | None:
        """Return the measured full tangent when the backend exposes it."""

        return self._elastic_tangent_3d

    def evaluate(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> ConstitutiveTrial:
        return self._material.evaluate(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
        )

    def evaluate_in_plane(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> InPlaneConstitutiveTrial:
        return self._material.evaluate_in_plane(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
        )

    def complete_trial(self, trial: InPlaneConstitutiveTrial) -> ConstitutiveTrial:
        return self._material.complete_trial(trial)

    def commit(self) -> None:
        self._material.commit()

    def revert(self) -> None:
        self._material.revert()


class PythonJ2PlaneStressBatch:
    """Transaction-safe adapter for the historical in-house J2 return mapping."""

    def __init__(
        self,
        initial_yield_stress_mpa: ArrayLike,
        hardening_coefficient_mpa: ArrayLike,
        hardening_exponent: float,
        *,
        young_modulus_mpa: float,
        poisson_ratio: float,
        hardening_mode: str = "ludwik",
        plastic_strain_max: float = 0.2,
        plastic_table_points: int = 1_000,
        first_positive_plastic_strain: float = 1e-6,
    ) -> None:
        yield_stress, coefficient = np.broadcast_arrays(
            np.asarray(initial_yield_stress_mpa, dtype=float),
            np.asarray(hardening_coefficient_mpa, dtype=float),
        )
        self._yield_stress = np.ravel(yield_stress).copy()
        self._coefficient = np.ravel(coefficient).copy()
        if self._yield_stress.size == 0:
            raise ValueError("at least one material point is required")
        if not np.isfinite(self._yield_stress).all() or np.any(self._yield_stress <= 0):
            raise ValueError("initial_yield_stress_mpa must be finite and positive")
        if not np.isfinite(self._coefficient).all() or np.any(self._coefficient < 0):
            raise ValueError("hardening_coefficient_mpa must be finite and non-negative")
        self._young = float(young_modulus_mpa)
        self._poisson = float(poisson_ratio)
        self._elasticity = plane_stress_elasticity(self._young, self._poisson)
        metric_product = self._elasticity @ PLANE_STRESS_VON_MISES_METRIC
        self._cm11 = float(metric_product[0, 0])
        self._cm12 = float(metric_product[0, 1])
        self._cm33 = float(metric_product[2, 2])
        self._hardening, self._hardening_derivative = make_hardening(
            hardening_exponent,
            hardening_mode,  # type: ignore[arg-type]
            plastic_strain_max,
            plastic_table_points,
            first_positive_plastic_strain,
        )
        self._plastic_strain = np.zeros((self.point_count, 3), dtype=float)
        self._peeq = np.zeros(self.point_count, dtype=float)
        self._trial_plastic: FloatArray | None = None
        self._trial_peeq: FloatArray | None = None

    @property
    def point_count(self) -> int:
        return self._yield_stress.size

    @property
    def backend_name(self) -> str:
        return "python-j2-plane-stress"

    @property
    def completion_strategy(self) -> str:
        return "j2_isotropic_analytical"

    @property
    def linear_system_matrix_type(self) -> LinearSystemMatrixType:
        """J2 with non-negative isotropic hardening gives an SPD tangent."""

        return "symmetric_positive_definite"

    @property
    def statistics(self) -> PlaneStressBatchStatistics:
        return PlaneStressBatchStatistics()

    def evaluate(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> ConstitutiveTrial:
        if not np.isfinite(time_increment) or time_increment <= 0:
            raise ValueError("time_increment must be finite and positive")
        total = np.asarray(in_plane_strain, dtype=float)
        if total.shape != (self.point_count, 3):
            raise ValueError(f"in_plane_strain must have shape {(self.point_count, 3)}")
        if not np.isfinite(total).all():
            raise ValueError("in_plane_strain must be finite")
        trial_stress = np.einsum(
            "ij,pj->pi",
            self._elasticity,
            total - self._plastic_strain,
        )
        stress, increment, peeq_increment = return_mapping(
            trial_stress,
            self._peeq,
            self._yield_stress,
            self._coefficient,
            self._hardening,
            self._cm11,
            self._cm12,
            self._cm33,
        )
        trial_plastic = self._plastic_strain + increment
        trial_peeq = self._peeq + peeq_increment
        tangent: FloatArray | None = None
        if consistent_tangent:
            tangent = np.broadcast_to(
                self._elasticity,
                (self.point_count, 3, 3),
            ).copy()
            plastic = peeq_increment > 0
            if np.any(plastic):
                tangent[plastic] = python_consistent_tangent(
                    stress[plastic],
                    peeq_increment[plastic],
                    self._peeq[plastic],
                    self._yield_stress[plastic],
                    self._coefficient[plastic],
                    self._hardening,
                    self._hardening_derivative,
                    self._elasticity,
                    self._cm11,
                    self._cm12,
                    self._cm33,
                )
        full = reconstruct_python_plane_stress_state(
            total,
            trial_plastic,
            stress,
            self._poisson,
        )
        self._trial_plastic = trial_plastic
        self._trial_peeq = trial_peeq
        return ConstitutiveTrial(
            stress_in_plane_mpa=stress,
            tangent_in_plane_mpa=tangent,
            full_stress_tensor_mpa=full.stress_tensor_mpa,
            full_strain_tensor=full.total_strain_tensor,
            elastic_strain_tensor=full.elastic_strain_tensor,
            plastic_strain_tensor=full.plastic_strain_tensor,
            plane_stress_residual_mpa=full.plane_stress_residual_vector_mpa,
            observables={
                "plastic_strain_2d": trial_plastic,
                "equivalent_plastic_strain": trial_peeq,
                "yield_surface_radius_mpa": (
                    self._yield_stress + self._coefficient * self._hardening(trial_peeq)
                ),
            },
        )

    def evaluate_in_plane(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> InPlaneConstitutiveTrial:
        return self.evaluate(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
        )

    def evaluate_equivalent_plastic_strain(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
    ) -> FloatArray:
        return self.evaluate(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=False,
        ).observables["equivalent_plastic_strain"]

    def complete_trial(self, trial: InPlaneConstitutiveTrial) -> ConstitutiveTrial:
        if not isinstance(trial, ConstitutiveTrial):
            raise TypeError("Python J2 in-plane trial is missing its reconstructed state")
        return trial

    def commit(self) -> None:
        if self._trial_plastic is None or self._trial_peeq is None:
            raise RuntimeError("no successful Python J2 trial state to commit")
        self._plastic_strain = self._trial_plastic
        self._peeq = self._trial_peeq
        self._trial_plastic = None
        self._trial_peeq = None

    def revert(self) -> None:
        self._trial_plastic = None
        self._trial_peeq = None


def _create_fcc_single_crystal_batch(
    backend: str,
    behaviour: Any,
    *,
    point_count: int,
    mfront_library: str,
    mfront_threads: int,
    nonlocal_coupling_modulus_mpa: float | None,
    local_plane_stress_options: dict[str, Any] | None,
    constitutive_options: Mapping[str, Any] | None,
) -> PlaneStressMaterialBatch:
    """Build a crystal-plasticity batch through the condensed 3D bridge.

    A crystal has no native plane-stress hypothesis, so the 3D law is condensed
    and the plane-stress condition is imposed in the GLOBAL frame, never in the
    crystal frame.
    """

    from fem_inhouse.core.crystal_orientation import (
        HomogeneousOrientationProvider,
        orientation_provider_from_mapping,
    )
    from fem_inhouse.core.mfront import (
        MFront3DCondensedPlaneStressBatch,
        MFront3DCondensedPlaneStressBlockBatch,
        MFrontNativeGeneralisedPlaneStressBatch,
    )
    from fem_inhouse.core.srix_parameters import resolve_srix_parameters

    if backend == "mfront-native-plane-stress":
        raise ValueError(
            f"MFront behaviour {behaviour.identifier!r} is a tridimensional single "
            "crystal and has no native plane-stress hypothesis; use "
            "'mfront-3d-condensed-plane-stress'"
        )
    # Crystal laws may expose a scalar non-local field even though their
    # constitutive source is not J2 PEEQ.  The selected criterion decides which
    # observable drives Helmholtz; the bridge only carries the external scalar.
    if nonlocal_coupling_modulus_mpa is not None and backend not in {
        "mfront-3d-condensed-plane-stress",
        "mfront-srix-generic-plane-stress",
    }:
        raise ValueError(
            "SRIX scalar non-local coupling is currently available only through "
            "the qualified 3D condensed plane-stress bridge"
        )

    options = dict(constitutive_options or {})
    orientation_configuration = options.pop("crystal_orientation", None)
    paired_parameter_set = options.pop("paired_parameter_set", None)
    parameter_set = options.pop("parameter_set", None)
    explicit_parameters = options.pop("parameters", None)
    # Diagnostic bench, GPS backend only: replace selected tangent points by a
    # raw full-step shadow trajectory. This is intentionally experimental: it
    # is not a same-state Schur oracle when the GPS path uses sub-stepping, and
    # it is off by default because it costs a full extra 3D integration.
    shadow_tangent = bool(options.pop("gps_shadow_tangent", False))
    shadow_tangent_scope = str(options.pop("gps_shadow_tangent_scope", "all"))
    failure_diagnostics = bool(options.pop("gps_failure_diagnostics", False))
    composite_fd_tangent = bool(options.pop("gps_composite_fd_tangent", False))
    composite_fd_step = float(options.pop("gps_composite_fd_step", 1.0e-6))
    if paired_parameter_set is not None and (
        parameter_set is not None or explicit_parameters is not None
    ):
        raise ValueError(
            "paired_parameter_set cannot be combined with legacy parameter_set or parameters"
        )
    provider = (
        HomogeneousOrientationProvider.identity()
        if orientation_configuration is None
        else orientation_provider_from_mapping(dict(orientation_configuration))
    )
    if options:
        raise ValueError(
            f"unsupported constitutive_options for {behaviour.identifier!r}: "
            f"{', '.join(sorted(options))}"
        )
    # Both crystal laws share this bridge but not their flow parameters: SRIX
    # has R, Meric-Cailletaud has (K, n). Applying one law's names to the other
    # is what the parameter guard in the batch caught the first time this was
    # wired, so the selection is gated on the registry the behaviour declares
    # rather than assumed.
    if paired_parameter_set is not None:
        if behaviour.crystal_flow_rule is None:
            raise ValueError(
                f"behaviour {behaviour.identifier!r} is not a registered crystal flow law"
            )
        from fem_inhouse.core.crystal_parameter_pairs import (
            resolve_paired_crystal_parameters,
        )

        overrides, _ = resolve_paired_crystal_parameters(
            paired_parameter_set=paired_parameter_set,
            law=behaviour.crystal_flow_rule,
        )
    elif behaviour.parameter_registry == "srix":
        overrides, _ = resolve_srix_parameters(
            parameter_set=parameter_set,
            explicit=explicit_parameters,
        )
    elif parameter_set is not None or explicit_parameters is not None:
        if behaviour.crystal_flow_rule == "meric_cailletaud" and parameter_set is not None:
            raise ValueError(
                "parameter_set is not supported for Méric-Cailletaud; use paired_parameter_set"
            )
        raise ValueError(
            f"MFront behaviour {behaviour.identifier!r} exposes no selectable "
            "parameter set; 'parameter_set' and 'parameters' are only accepted by "
            "behaviours that declare one"
        )
    else:
        overrides = None

    local_options = dict(local_plane_stress_options or {})
    crystal_material_properties: dict[str, Any] = {}
    if any(
        variable.canonical_name == "coupling_modulus_mpa"
        for variable in behaviour.material_properties
    ):
        # The SRIX micromorphic extension keeps Hchi as a required MFront
        # property, so the caller's request has to be carried here explicitly:
        # this factory does not forward `micromorphic_coupling_modulus_mpa` to
        # the condensed batch, and that batch defaults an unsupplied property
        # to zero. Populating the dictionary only in the local case therefore
        # left every non-local crystal run with Hchi = 0 -- a silently inert
        # coupling that made the Generic/legacy equivalence test pass because
        # BOTH sides were uncoupled.
        #
        # Zero remains the local value: it is what reduces the law exactly to
        # the historical response.
        crystal_material_properties["MicromorphicCouplingModulus"] = np.full(
            point_count,
            0.0 if nonlocal_coupling_modulus_mpa is None else float(nonlocal_coupling_modulus_mpa),
        )
    if backend in {
        "mfront-native-generalised-plane-stress",
        "mfront-structural-plane-stress",
    }:
        # The structural backend selects the generated closure variant from
        # the catalogue.  The host adapter is identical for SRIX and Méric;
        # only the registered MFront behaviour changes.
        if backend == "mfront-structural-plane-stress":
            from fem_inhouse.core.mfront_behaviours import MFRONT_BEHAVIOURS

            structural_id = f"{behaviour.identifier}_structural_plane_stress"
            behaviour = MFRONT_BEHAVIOURS.get(structural_id)
        elif behaviour.identifier == "fcc_forest_rubin_srix":
            from fem_inhouse.core.mfront_behaviours import MFRONT_BEHAVIOURS

            behaviour = MFRONT_BEHAVIOURS.get("fcc_forest_rubin_srix_gps")
        return MFrontNativeGeneralisedPlaneStressBatch(
            mfront_library,
            behaviour_spec=behaviour,
            point_count=point_count,
            rotation_global_to_material=provider.rotations_global_to_material(point_count),
            thread_count=mfront_threads,
            behaviour_name=behaviour.behaviour_name(
                "structural_plane_stress"
                if backend == "mfront-structural-plane-stress"
                else "condensed_3d"
            ),
            behaviour_parameters=overrides,
            backend_label=backend,
            shadow_tangent=shadow_tangent,
            shadow_tangent_scope=shadow_tangent_scope,
            composite_fd_tangent=composite_fd_tangent,
            composite_fd_step=composite_fd_step,
            failure_diagnostics=failure_diagnostics,
            **local_options,
        )
    if shadow_tangent:
        raise ValueError(
            "gps_shadow_tangent only applies to the "
            "'mfront-native-generalised-plane-stress' backend"
        )
    block_size = local_options.pop("condensation_block_size", None)
    condensed_factory = (
        MFront3DCondensedPlaneStressBlockBatch
        if block_size is not None
        else MFront3DCondensedPlaneStressBatch
    )
    if block_size is not None:
        local_options["condensation_block_size"] = int(block_size)
    return condensed_factory(
        mfront_library,
        behaviour_spec=behaviour,
        point_count=point_count,
        rotation_global_to_material=provider.rotations_global_to_material(point_count),
        thread_count=mfront_threads,
        behaviour_name=behaviour.behaviour_name("condensed_3d"),
        behaviour_parameters=overrides,
        material_property_values=crystal_material_properties,
        **local_options,
    )


def create_plane_stress_material_batch(
    backend: str,
    initial_yield_stress_mpa: ArrayLike,
    hardening_coefficient_mpa: ArrayLike,
    hardening_exponent: float,
    *,
    young_modulus_mpa: float,
    poisson_ratio: float,
    hardening_mode: str,
    plastic_strain_max: float,
    plastic_table_points: int,
    first_positive_plastic_strain: float,
    mfront_library: str,
    mfront_threads: int,
    local_plane_stress_options: dict[str, Any] | None = None,
    nonlocal_coupling_modulus_mpa: float | None = None,
    mfront_behaviour_id: str | None = None,
    constitutive_options: Mapping[str, Any] | None = None,
) -> PlaneStressMaterialBatch:
    """Construct a backend without exposing its implementation to global Newton."""

    if backend == "python":
        return PythonJ2PlaneStressBatch(
            initial_yield_stress_mpa,
            hardening_coefficient_mpa,
            hardening_exponent,
            young_modulus_mpa=young_modulus_mpa,
            poisson_ratio=poisson_ratio,
            hardening_mode=hardening_mode,
            plastic_strain_max=plastic_strain_max,
            plastic_table_points=plastic_table_points,
            first_positive_plastic_strain=first_positive_plastic_strain,
        )
    if backend in {
        "mfront",
        "mfront-native-plane-stress",
        "mfront-3d-condensed-plane-stress",
        "mfront-srix-generic-plane-stress",
        "mfront-native-generalised-plane-stress",
        "mfront-structural-plane-stress",
    }:
        from fem_inhouse.core.mfront import (
            MFront3DCondensedPlaneStressBatch,
            MFrontNativePlaneStressBatch,
        )
        from fem_inhouse.core.mfront_behaviours import MFRONT_BEHAVIOURS

        # The behaviour names come from the catalogue rather than from inline
        # conditionals, so a new law is a catalogue entry rather than an edit
        # here. The two registered entries reproduce the previous names exactly.
        selected_behaviour_id = mfront_behaviour_id or (
            "micromorphic_ludwik_j2" if nonlocal_coupling_modulus_mpa is not None else "ludwik_j2"
        )
        behaviour = MFRONT_BEHAVIOURS.get(selected_behaviour_id)
        if backend == "mfront-srix-generic-plane-stress":
            if selected_behaviour_id != "fcc_forest_rubin_srix_generic_validation":
                raise ValueError(
                    "mfront-srix-generic-plane-stress requires the validation "
                    "behaviour id 'fcc_forest_rubin_srix_generic_validation'"
                )
            if nonlocal_coupling_modulus_mpa is None:
                coupling = 0.0
            else:
                coupling = nonlocal_coupling_modulus_mpa
            from fem_inhouse.core.crystal_orientation import (
                HomogeneousOrientationProvider,
                orientation_provider_from_mapping,
            )
            from fem_inhouse.core.mfront import (
                SrixGeneric3DCondensedPlaneStressBatch,
                SrixGeneric3DMaterialPointBatch,
            )
            from fem_inhouse.core.srix_parameters import resolve_srix_parameters

            options = dict(constitutive_options or {})
            orientation_configuration = options.pop("crystal_orientation", None)
            parameter_set = options.pop("parameter_set", None)
            explicit_parameters = options.pop("parameters", None)
            if options:
                raise ValueError(
                    "unsupported constitutive_options for the validation Generic SRIX "
                    f"backend: {', '.join(sorted(options))}"
                )
            overrides, _ = resolve_srix_parameters(
                parameter_set=parameter_set,
                explicit=explicit_parameters,
            )
            provider = (
                HomogeneousOrientationProvider.identity()
                if orientation_configuration is None
                else orientation_provider_from_mapping(dict(orientation_configuration))
            )
            local_options = dict(local_plane_stress_options or {})
            allowed = {
                "local_tolerance_mpa",
                "local_relative_tolerance",
                "maximum_local_iterations",
                "maximum_cbb_condition_number",
                "local_condition_check_mode",
            }
            unsupported = set(local_options) - allowed
            if unsupported:
                raise ValueError(
                    "unsupported local_plane_stress_options for Generic SRIX: "
                    f"{', '.join(sorted(unsupported))}"
                )
            bridge = SrixGeneric3DMaterialPointBatch(
                mfront_library,
                point_count=int(np.asarray(initial_yield_stress_mpa).size),
                micromorphic_coupling_modulus_mpa=coupling,
                behaviour_name=behaviour.behaviour_name("condensed_3d"),
                rotation_global_to_material=provider.rotations_global_to_material(
                    int(np.asarray(initial_yield_stress_mpa).size)
                ),
                behaviour_parameters=overrides,
                thread_count=mfront_threads,
            )
            return cast(
                PlaneStressMaterialBatch,
                SrixGeneric3DCondensedPlaneStressBatch(
                    bridge,
                    local_tolerance_mpa=local_options.get("local_tolerance_mpa", 1.0e-8),
                    maximum_local_iterations=local_options.get("maximum_local_iterations", 15),
                ),
            )
        if behaviour.bridge_profile == "fcc_single_crystal_v1":
            return _create_fcc_single_crystal_batch(
                backend,
                behaviour,
                point_count=int(np.asarray(initial_yield_stress_mpa).size),
                mfront_library=mfront_library,
                mfront_threads=mfront_threads,
                nonlocal_coupling_modulus_mpa=nonlocal_coupling_modulus_mpa,
                local_plane_stress_options=local_plane_stress_options,
                constitutive_options=constitutive_options,
            )
        if behaviour.bridge_profile != "ludwik_j2_v1":
            raise ValueError(
                f"MFront behaviour {selected_behaviour_id!r} requires bridge profile "
                f"{behaviour.bridge_profile!r}; register a constitutive plugin for that "
                "profile"
            )
        # The elastic constants and the hardening table are conventions of the
        # J2 behaviours, which hardcode them. A crystal law carries its own
        # cubic elasticity inside MFront, so this check is reached only after
        # the crystal profile has returned above.
        if not np.isclose(young_modulus_mpa, 205_000.0) or not np.isclose(poisson_ratio, 0.3):
            raise ValueError("the compiled MFront behaviours require E=205000 MPa and nu=0.3")
        if not np.isclose(first_positive_plastic_strain, 1e-6):
            raise ValueError(
                "the compiled MFront behaviours require first_positive_plastic_strain=1e-6"
            )
        exposes_nonlocal_peeq = any(
            variable.canonical_name == "nonlocal_equivalent_plastic_strain"
            for variable in behaviour.external_state_variables
        )
        if exposes_nonlocal_peeq != (nonlocal_coupling_modulus_mpa is not None):
            raise ValueError(
                f"MFront behaviour {selected_behaviour_id!r} and "
                "nonlocal_coupling_modulus_mpa select incompatible local/nonlocal modes"
            )
        if exposes_nonlocal_peeq:
            behaviour.external_entry_name("nonlocal_equivalent_plastic_strain")

        common = (
            mfront_library,
            initial_yield_stress_mpa,
            hardening_coefficient_mpa,
            np.full(np.asarray(initial_yield_stress_mpa).size, hardening_exponent),
        )
        micromorphic_options: dict[str, Any] = {}
        if nonlocal_coupling_modulus_mpa is not None:
            micromorphic_options = {
                "micromorphic_coupling_modulus_mpa": nonlocal_coupling_modulus_mpa,
            }
        if backend in {"mfront", "mfront-native-plane-stress"}:
            return MFrontNativePlaneStressBatch(
                *common,
                thread_count=mfront_threads,
                behaviour_name=behaviour.behaviour_name("native"),
                **micromorphic_options,
            )
        return MFront3DCondensedPlaneStressBatch(
            *common,
            thread_count=mfront_threads,
            behaviour_name=behaviour.behaviour_name("condensed_3d"),
            **micromorphic_options,
            **(local_plane_stress_options or {}),
        )

    from fem_inhouse.core.constitutive_plugins import (
        CONSTITUTIVE_PLUGINS,
        PlaneStressMaterialRequest,
        load_constitutive_plugins,
    )

    load_constitutive_plugins()
    if CONSTITUTIVE_PLUGINS.contains(backend):
        request = PlaneStressMaterialRequest(
            initial_yield_stress_mpa=initial_yield_stress_mpa,
            hardening_coefficient_mpa=hardening_coefficient_mpa,
            hardening_exponent=hardening_exponent,
            young_modulus_mpa=young_modulus_mpa,
            poisson_ratio=poisson_ratio,
            hardening_mode=hardening_mode,
            plastic_strain_max=plastic_strain_max,
            plastic_table_points=plastic_table_points,
            first_positive_plastic_strain=first_positive_plastic_strain,
            mfront_library=mfront_library,
            mfront_threads=mfront_threads,
            local_plane_stress_options=local_plane_stress_options or {},
            nonlocal_coupling_modulus_mpa=nonlocal_coupling_modulus_mpa,
            options=constitutive_options or {},
        )
        batch = CONSTITUTIVE_PLUGINS.get(backend).create_batch(request)
        if not isinstance(batch, PlaneStressMaterialBatch):
            raise TypeError(
                f"constitutive plugin {backend!r} did not return a PlaneStressMaterialBatch"
            )
        return batch
    plugins = ", ".join(CONSTITUTIVE_PLUGINS.identifiers())
    plugin_suffix = f"; registered plugins: {plugins}" if plugins else ""
    raise ValueError(
        "constitutive_backend must be 'python', 'mfront-native-plane-stress', "
        f"or 'mfront-3d-condensed-plane-stress'{plugin_suffix}"
    )
