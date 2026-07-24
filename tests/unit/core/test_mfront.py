from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from fem_inhouse.core.mfront import (
    MFrontMaterialPointBatch,
    engineering_strain_to_kelvin,
    kelvin_strain_to_engineering,
    kelvin_stress_to_engineering,
    kelvin_tangent_to_engineering,
)


def test_kelvin_strain_round_trip_preserves_engineering_shear() -> None:
    engineering = np.array([[1e-3, -2e-3, 4e-3], [0.0, 3e-4, -8e-4]])
    kelvin = engineering_strain_to_kelvin(engineering)

    assert kelvin.shape == (2, 4)
    np.testing.assert_allclose(kelvin_strain_to_engineering(kelvin), engineering)


def test_kelvin_stress_and_tangent_recover_plane_stress_elasticity() -> None:
    young = 205_000.0
    poisson = 0.3
    factor = young / (1 - poisson**2)
    kelvin_stress = np.array([12.0, -3.0, 0.0, np.sqrt(2.0) * 7.0])
    kelvin_tangent = np.array(
        [
            [factor, poisson * factor, 0.0, 0.0],
            [poisson * factor, factor, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, young / (1 + poisson)],
        ]
    )

    np.testing.assert_allclose(
        kelvin_stress_to_engineering(kelvin_stress),
        [12.0, -3.0, 7.0],
    )
    np.testing.assert_allclose(
        kelvin_tangent_to_engineering(kelvin_tangent),
        factor
        * np.array(
            [
                [1.0, poisson, 0.0],
                [poisson, 1.0, 0.0],
                [0.0, 0.0, (1 - poisson) / 2],
            ]
        ),
    )


@pytest.mark.parametrize(
    ("function", "values"),
    [
        (engineering_strain_to_kelvin, np.zeros((2, 4))),
        (kelvin_strain_to_engineering, np.zeros((2, 3))),
        (kelvin_stress_to_engineering, np.zeros((2, 3))),
        (kelvin_tangent_to_engineering, np.zeros((3, 3))),
    ],
)
def test_kelvin_conversions_reject_wrong_shapes(function: object, values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        function(values)  # type: ignore[operator]


def test_mfront_batch_rejects_missing_library(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MFrontMaterialPointBatch(
            str(tmp_path) + "/missing.so",
            250.0,
            380.0,
            0.245,
        )


@pytest.mark.mfront
def test_compiled_mfront_behaviour_matches_elastic_plane_stress() -> None:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    batch = MFrontMaterialPointBatch(
        library,
        1e9,
        380.0,
        0.245,
    )
    result = batch.evaluate(
        np.array([[1e-3, 0.0, 2e-3]]),
        consistent_tangent=True,
    )
    young = 205_000.0
    poisson = 0.3
    factor = young / (1 - poisson**2)
    elasticity = factor * np.array(
        [
            [1.0, poisson, 0.0],
            [poisson, 1.0, 0.0],
            [0.0, 0.0, (1 - poisson) / 2],
        ]
    )

    np.testing.assert_allclose(result.stress_mpa[0], elasticity @ [1e-3, 0.0, 2e-3])
    assert result.consistent_tangent_mpa is not None
    np.testing.assert_allclose(result.consistent_tangent_mpa[0], elasticity)
    np.testing.assert_array_equal(result.plastic_strain, np.zeros((1, 3)))
    np.testing.assert_array_equal(result.equivalent_plastic_strain, np.zeros(1))


@pytest.mark.mfront
def test_mfront_trial_can_be_reverted_before_commit() -> None:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    batch = MFrontMaterialPointBatch(
        library,
        250.0,
        380.0,
        0.245,
    )
    plastic_trial = batch.evaluate(np.array([[0.005, 0.0, 0.0]]))
    assert plastic_trial.equivalent_plastic_strain[0] > 0
    batch.revert()

    zero_trial = batch.evaluate(np.zeros((1, 3)))
    np.testing.assert_allclose(zero_trial.stress_mpa, 0.0, atol=1e-12)
    np.testing.assert_allclose(zero_trial.equivalent_plastic_strain, 0.0, atol=1e-15)
