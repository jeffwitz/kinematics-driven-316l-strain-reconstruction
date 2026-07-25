from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.core.nonlocal_plasticity import (
    NonlocalCouplingConvergenceError,
    evaluate_nonlocal_fixed_point,
)
from fem_inhouse.core.plane_stress_material import ConstitutiveTrial


class _FakeNonlocalBatch:
    def __init__(self, point_count: int, *, local_peeq: float, follows_chi: bool = False):
        self.point_count = point_count
        self.local_peeq = local_peeq
        self.follows_chi = follows_chi
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
        peeq = (
            self.external_chi + self.local_peeq
            if self.follows_chi
            else np.full(self.point_count, self.local_peeq)
        )
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
                "yield_surface_radius_mpa": 300.0
                + 1_000.0 * (peeq - self.external_chi),
            },
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


def test_nonconvergent_constitutive_feedback_is_reported() -> None:
    batch = _FakeNonlocalBatch(24, local_peeq=0.01, follows_chi=True)

    with pytest.raises(
        NonlocalCouplingConvergenceError,
        match="did not converge",
    ):
        _evaluate(batch, maximum_iterations=2, relaxation=0.5)


@pytest.mark.parametrize("value", [np.full((3, 2), np.nan), np.full((3, 2), -1.0)])
def test_invalid_initial_nonlocal_field_is_rejected(value: np.ndarray) -> None:
    batch = _FakeNonlocalBatch(24, local_peeq=0.01)

    with pytest.raises(ValueError, match="finite and nonnegative"):
        _evaluate(batch, initial_nonlocal_peeq=value)
