from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from fem_inhouse.core.constitutive_plugins import (
    CallablePlaneStressMaterialPlugin,
    ConstitutivePluginRegistry,
    PlaneStressMaterialRequest,
    register_constitutive_plugin,
)
from fem_inhouse.core.mfront_behaviours import (
    MFRONT_BEHAVIOURS,
    MFrontBehaviourRegistry,
    MFrontBehaviourSpec,
    MFrontVariableSpec,
)
from fem_inhouse.core.nonlocal_criteria import (
    NonlocalCriterionRegistry,
    NonlocalRegularisationContext,
    NonlocalRegularisationResult,
)
from fem_inhouse.core.plane_stress_material import (
    ConstitutiveTrial,
    PlaneStressBatchStatistics,
    create_plane_stress_material_batch,
)


class _DummyBatch:
    point_count = 1
    backend_name = "dummy-crystal-plasticity"
    completion_strategy = "mfront-elastic-strain"
    linear_system_matrix_type = "nonsymmetric"
    statistics = PlaneStressBatchStatistics()

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate_in_plane(self, *args, **kwargs):
        raise NotImplementedError

    def complete_trial(self, trial):
        return trial

    def commit(self) -> None:
        pass

    def revert(self) -> None:
        pass


def _request() -> PlaneStressMaterialRequest:
    return PlaneStressMaterialRequest(
        initial_yield_stress_mpa=np.array([250.0]),
        hardening_coefficient_mpa=np.array([380.0]),
        hardening_exponent=0.245,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.3,
        hardening_mode="ludwik",
        plastic_strain_max=0.2,
        plastic_table_points=1000,
        first_positive_plastic_strain=1e-6,
        mfront_library="behaviour.so",
        mfront_threads=1,
        options={"orientation_source": "ebsd"},
    )


def test_constitutive_registry_protects_duplicates_and_forwards_options() -> None:
    registry = ConstitutivePluginRegistry()
    captured: list[PlaneStressMaterialRequest] = []
    plugin = CallablePlaneStressMaterialPlugin(
        "crystal_plasticity",
        lambda request: captured.append(request) or _DummyBatch(),
    )

    registry.register(plugin)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(plugin)

    batch = registry.get("crystal_plasticity").create_batch(_request())

    assert batch.backend_name == "dummy-crystal-plasticity"
    assert captured[0].options["orientation_source"] == "ebsd"
    with pytest.raises(TypeError):
        captured[0].options["new"] = 1  # type: ignore[index]


def test_public_factory_dispatches_to_registered_plugin() -> None:
    identifier = "test_crystal_plasticity"
    captured: list[PlaneStressMaterialRequest] = []
    register_constitutive_plugin(
        identifier,
        lambda request: captured.append(request) or _DummyBatch(),
        replace=True,
    )

    batch = create_plane_stress_material_batch(
        identifier,
        np.array([250.0]),
        np.array([380.0]),
        0.245,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.3,
        hardening_mode="ludwik",
        plastic_strain_max=0.2,
        plastic_table_points=1000,
        first_positive_plastic_strain=1e-6,
        mfront_library="crystal.so",
        mfront_threads=4,
        constitutive_options={"slip_systems": "fcc"},
    )

    assert batch.backend_name == "dummy-crystal-plasticity"
    assert captured[0].mfront_library == "crystal.so"
    assert captured[0].mfront_threads == 4
    assert captured[0].options == {"slip_systems": "fcc"}


def test_builtin_mfront_catalogue_preserves_current_behaviour_names() -> None:
    local = MFRONT_BEHAVIOURS.get("ludwik_j2")
    coupled = MFRONT_BEHAVIOURS.get("micromorphic_ludwik_j2")

    assert local.behaviour_name("native") == "PixelLudwikJ2Plasticity"
    assert local.behaviour_name("condensed_3d") == "PixelLudwikJ2Plasticity3D"
    assert coupled.external_entry_name("nonlocal_equivalent_plastic_strain") == (
        "NonlocalEquivalentPlasticStrain"
    )


def test_mfront_catalogue_rejects_incompatible_local_nonlocal_selection() -> None:
    with pytest.raises(ValueError, match="incompatible local/nonlocal modes"):
        create_plane_stress_material_batch(
            "mfront-native-plane-stress",
            np.array([250.0]),
            np.array([380.0]),
            0.245,
            young_modulus_mpa=205_000.0,
            poisson_ratio=0.3,
            hardening_mode="ludwik",
            plastic_strain_max=0.2,
            plastic_table_points=1000,
            first_positive_plastic_strain=1e-6,
            mfront_library="missing.so",
            mfront_threads=1,
            mfront_behaviour_id="micromorphic_ludwik_j2",
        )


def test_mfront_catalogue_describes_future_crystal_plasticity_requirements() -> None:
    registry = MFrontBehaviourRegistry()
    specification = MFrontBehaviourSpec(
        identifier="fcc_crystal_plasticity",
        native_plane_stress_behaviour=None,
        tridimensional_behaviour="FCCCrystalPlasticity",
        material_properties=(MFrontVariableSpec("critical_resolved_shear", "tau0"),),
        internal_state_variables=(
            MFrontVariableSpec("elastic_strain", "ElasticStrain", "symmetric_tensor"),
            MFrontVariableSpec("slip", "Slip", "vector"),
        ),
        requires_rotation_matrix=True,
        bridge_profile="crystal_plasticity_v1",
    )

    registry.register(specification)

    assert registry.get("fcc_crystal_plasticity").requires_rotation_matrix
    assert registry.get("fcc_crystal_plasticity").behaviour_name("condensed_3d") == (
        "FCCCrystalPlasticity"
    )
    with pytest.raises(ValueError, match="does not support"):
        registry.get("fcc_crystal_plasticity").behaviour_name("native")


@dataclass(frozen=True, slots=True)
class _IdentityCriterion:
    identifier: str = "signed_identity"
    source_name: str = "signed_source"
    requires_nonnegative_field: bool = False

    def supports_material(self, material_batch: object) -> bool:
        return True

    def set_external_field(self, material_batch: object, values) -> None:
        del material_batch, values

    def evaluate_source_and_safety(
        self,
        material_batch: object,
        in_plane_strain,
        *,
        time_increment: float,
    ):
        del material_batch, time_increment
        points = np.asarray(in_plane_strain).shape[0]
        return -np.ones(points), np.full(points, 300.0)

    def source_from_trial(self, trial: ConstitutiveTrial):
        return trial.observables[self.source_name]

    def safety_from_trial(self, trial: ConstitutiveTrial):
        return trial.observables["yield_surface_radius_mpa"]

    def regularise(
        self,
        source_element_field,
        context: NonlocalRegularisationContext,
    ) -> NonlocalRegularisationResult:
        del context
        return NonlocalRegularisationResult(
            filtered_element_field=np.asarray(source_element_field).copy(),
            residual_relative=0.0,
            mean_drift=0.0,
        )


def test_nonlocal_registry_builds_custom_signed_criterion() -> None:
    registry = NonlocalCriterionRegistry()
    registry.register("signed_identity", lambda options: _IdentityCriterion())

    criterion = registry.create("signed_identity")

    assert criterion.identifier == "signed_identity"
    assert not criterion.requires_nonnegative_field
