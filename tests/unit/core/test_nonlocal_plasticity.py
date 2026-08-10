from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.core.nonlocal_criteria import (
    NonlocalRegularisationContext,
    NonlocalRegularisationResult,
)
from fem_inhouse.core.nonlocal_plasticity import (
    NonlocalCouplingConvergenceError,
    _element_average,
    _gauss_values,
    _mixed_relative_maximum_norm,
    classify_fixed_point_history,
    evaluate_nonlocal_fixed_point,
)
from fem_inhouse.core.plane_stress_material import ConstitutiveTrial


@pytest.mark.parametrize("order", ["C", "F"])
def test_element_layout_round_trip_preserves_non_symmetric_cells(order: str) -> None:
    field = np.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
    point_values = _gauss_values(field, 2, element_order=order)
    recovered = _element_average(
        point_values,
        element_shape=field.shape,
        gauss_points_per_element=2,
        name="test_field",
        element_order=order,
    )
    np.testing.assert_array_equal(recovered, field)


class _FakeNonlocalBatch:
    def __init__(
        self,
        point_count: int,
        *,
        local_peeq: float,
        follows_chi: bool = False,
        feedback: float | None = None,
        yield_radius_mpa: float | None = None,
    ):
        self.point_count = point_count
        self.local_peeq = local_peeq
        self.follows_chi = follows_chi
        self.feedback = feedback
        self.yield_radius_mpa = yield_radius_mpa
        self.external_chi = np.zeros(point_count)
        self.evaluate_calls = 0

    def set_nonlocal_equivalent_plastic_strain(self, values) -> None:
        self.external_chi = np.asarray(values, dtype=np.float64).copy()

    def evaluate(
        self,
        in_plane_strain,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> ConstitutiveTrial:
        del time_increment, consistent_tangent
        self.evaluate_calls += 1
        assert np.asarray(in_plane_strain).shape == (self.point_count, 3)
        if self.feedback is not None:
            peeq = self.local_peeq + self.feedback * self.external_chi
        elif self.follows_chi:
            peeq = self.external_chi + self.local_peeq
        else:
            peeq = np.full(self.point_count, self.local_peeq)
        tensor = np.zeros((self.point_count, 3, 3))
        return ConstitutiveTrial(
            stress_in_plane_mpa=np.zeros((self.point_count, 3)),
            tangent_in_plane_mpa=np.zeros((self.point_count, 3, 3)),
            full_stress_tensor_mpa=tensor.copy(),
            full_strain_tensor=tensor.copy(),
            elastic_strain_tensor=tensor.copy(),
            plastic_strain_tensor=tensor.copy(),
            plane_stress_residual_mpa=np.zeros((self.point_count, 3)),
            observables={
                "equivalent_plastic_strain": peeq,
                "yield_surface_radius_mpa": (
                    np.full(self.point_count, self.yield_radius_mpa)
                    if self.yield_radius_mpa is not None
                    else 300.0 + 1_000.0 * (peeq - self.external_chi)
                ),
            },
        )

    def evaluate_equivalent_plastic_strain(
        self,
        in_plane_strain,
        *,
        time_increment: float,
    ):
        return self.evaluate(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=False,
        ).observables["equivalent_plastic_strain"]

    def evaluate_nonlocal_state(
        self,
        in_plane_strain,
        *,
        time_increment: float,
    ):
        trial = self.evaluate(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=False,
        )
        return (
            trial.observables["equivalent_plastic_strain"],
            trial.observables["yield_surface_radius_mpa"],
        )

    def evaluate_in_plane(
        self,
        in_plane_strain,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ):
        return self.evaluate(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
        )

    def commit(self) -> None:
        raise AssertionError("the fixed-point solver must never commit material state")


def _evaluate(
    batch: _FakeNonlocalBatch,
    *,
    coupling_modulus_mpa: float = 1_000.0,
    relaxation: float = 1.0,
    maximum_iterations: int = 5,
    initial_nonlocal_peeq: np.ndarray | None = None,
    relaxation_strategy: str = "fixed",
    criterion=None,
):
    return evaluate_nonlocal_fixed_point(
        batch,
        np.zeros((batch.point_count, 3)),
        time_increment=0.1,
        element_shape=(3, 2),
        gauss_points_per_element=4,
        initial_nonlocal_peeq=(
            np.zeros((3, 2))
            if initial_nonlocal_peeq is None
            else initial_nonlocal_peeq
        ),
        length_scale_mm=0.05,
        spacing_x_mm=0.01,
        spacing_y_mm=0.01,
        coupling_modulus_mpa=coupling_modulus_mpa,
        relaxation=relaxation,
        relaxation_strategy=relaxation_strategy,
        criterion=criterion,
        minimum_relaxation=0.05,
        maximum_relaxation=0.8,
        relative_tolerance=1e-10,
        maximum_iterations=maximum_iterations,
        maximum_helmholtz_residual=1e-10,
    )


def test_uniform_local_plasticity_gives_identical_nonlocal_field_without_commit() -> None:
    batch = _FakeNonlocalBatch(24, local_peeq=0.02)

    result = _evaluate(batch)

    np.testing.assert_allclose(result.nonlocal_peeq, 0.02, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(result.mismatch, 0.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(result.residual_field, 0.0, rtol=0.0, atol=1e-14)
    assert result.iterations == 2
    assert batch.evaluate_calls == 3


def test_zero_coupling_uses_one_source_evaluation_and_produces_outputs() -> None:
    batch = _FakeNonlocalBatch(24, local_peeq=0.03)

    result = _evaluate(batch, coupling_modulus_mpa=0.0, relaxation=0.5)

    assert result.iterations == 1
    assert batch.evaluate_calls == 2
    np.testing.assert_array_equal(result.nonlocal_hardening_mpa, 0.0)
    np.testing.assert_allclose(result.nonlocal_peeq, 0.03, rtol=0.0, atol=1e-14)


def test_mixed_maximum_norm_is_independent_of_field_size() -> None:
    small_delta = np.full((3, 2), 4.0e-7)
    large_delta = np.full((300, 200), 4.0e-7)

    assert _mixed_relative_maximum_norm(
        small_delta,
        np.zeros_like(small_delta),
    ) == pytest.approx(4.0e-7)
    assert _mixed_relative_maximum_norm(
        large_delta,
        np.zeros_like(large_delta),
    ) == pytest.approx(4.0e-7)


def test_mixed_maximum_norm_uses_relative_branch_above_unit_state() -> None:
    state = np.full((2, 2), 4.0)
    difference = np.full((2, 2), 2.0e-5)

    assert _mixed_relative_maximum_norm(difference, state) == pytest.approx(5.0e-6)


def test_nonconvergent_constitutive_feedback_is_reported() -> None:
    batch = _FakeNonlocalBatch(24, local_peeq=0.01, follows_chi=True)

    with pytest.raises(
        NonlocalCouplingConvergenceError,
        match="did not converge",
    ):
        _evaluate(batch, maximum_iterations=2, relaxation=0.5)


def test_aitken_accelerates_oscillatory_contracting_feedback() -> None:
    batch = _FakeNonlocalBatch(24, local_peeq=0.02, feedback=-0.9)

    result = _evaluate(
        batch,
        relaxation=0.2,
        relaxation_strategy="aitken",
        maximum_iterations=10,
    )

    np.testing.assert_allclose(
        result.nonlocal_peeq,
        0.02 / 1.9,
        rtol=0.0,
        atol=1e-10,
    )
    assert result.iterations <= 4
    assert any(item.acceleration_accepted for item in result.iteration_history)
    assert classify_fixed_point_history(result.iteration_history) == "oscillating"


def test_nonpositive_yield_radius_stops_before_final_tangent() -> None:
    batch = _FakeNonlocalBatch(
        24,
        local_peeq=0.02,
        yield_radius_mpa=-1.0,
    )

    with pytest.raises(NonlocalCouplingConvergenceError) as caught:
        _evaluate(batch, relaxation=0.5)

    assert caught.value.reason == "nonpositive_yield_surface"
    assert len(caught.value.iteration_history) == 1
    assert batch.evaluate_calls == 1


@pytest.mark.parametrize("value", [np.full((3, 2), np.nan), np.full((3, 2), -1.0)])
def test_invalid_initial_nonlocal_field_is_rejected(value: np.ndarray) -> None:
    batch = _FakeNonlocalBatch(24, local_peeq=0.01)

    with pytest.raises(ValueError, match="finite and nonnegative"):
        _evaluate(batch, initial_nonlocal_peeq=value)


class _SignedIdentityCriterion:
    """A criterion whose field may be negative, and whose operator is identity.

    Exists to pin that the fixed point no longer clips at zero on its own: the
    nonnegativity of PEEQ is the criterion's property, not the solver's.
    """

    identifier = "signed_identity"
    source_name = "signed_plastic_activity"
    requires_nonnegative_field = False

    def supports_material(self, material_batch: object) -> bool:
        return isinstance(material_batch, _FakeNonlocalBatch)

    def set_external_field(self, material_batch: object, values) -> None:
        assert isinstance(material_batch, _FakeNonlocalBatch)
        material_batch.external_chi = np.asarray(values).copy()

    def evaluate_source_and_safety(
        self,
        material_batch: object,
        in_plane_strain,
        *,
        time_increment: float,
    ):
        assert isinstance(material_batch, _FakeNonlocalBatch)
        del time_increment
        points = np.asarray(in_plane_strain).shape[0]
        material_batch.evaluate_calls += 1
        return np.full(points, -0.02), np.full(points, 300.0)

    def source_from_trial(self, trial: ConstitutiveTrial):
        return np.full(trial.stress_in_plane_mpa.shape[0], -0.02)

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


def test_custom_signed_criterion_passes_through_fixed_point_without_clipping() -> None:
    batch = _FakeNonlocalBatch(24, local_peeq=0.03)

    result = _evaluate(
        batch,
        coupling_modulus_mpa=0.0,
        initial_nonlocal_peeq=np.full((3, 2), -0.02),
        criterion=_SignedIdentityCriterion(),
    )

    np.testing.assert_allclose(result.nonlocal_peeq, -0.02, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.local_element_peeq, -0.02, rtol=0.0, atol=0.0)
