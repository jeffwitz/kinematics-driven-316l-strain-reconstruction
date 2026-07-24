from __future__ import annotations

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


def test_mfront_batch_rejects_missing_library(tmp_path: object) -> None:
    with pytest.raises(FileNotFoundError):
        MFrontMaterialPointBatch(
            str(tmp_path) + "/missing.so",
            250.0,
            380.0,
            0.245,
        )
