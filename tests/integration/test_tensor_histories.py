from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np
import pytest

from fem_inhouse.core.constitutive import (
    PLANE_STRESS_VON_MISES_METRIC,
    make_hardening,
    return_mapping,
)
from fem_inhouse.core.element import plane_stress_elasticity
from fem_inhouse.core.mfront import MFrontMaterialPointBatch
from fem_inhouse.core.tensor_reconstruction import (
    FullTensorState,
    reconstruct_python_plane_stress_state,
)
from fem_inhouse.postprocessing import instantaneous_equivalent_plastic_strain

YOUNG = 205_000.0
POISSON = 0.3
YIELD = 250.0
HARDENING = 380.0
EXPONENT = 0.245


def _plastic_strain_from_stress(equivalent_stress_mpa: np.ndarray) -> np.ndarray:
    overstress = np.maximum(equivalent_stress_mpa - YIELD, 0.0)
    return (overstress / HARDENING) ** (1.0 / EXPONENT)


def _uniaxial_path() -> np.ndarray:
    stress = np.linspace(0.0, 410.0, 31)
    plastic = _plastic_strain_from_stress(stress)
    return np.stack(
        (
            stress / YOUNG + plastic,
            -POISSON * stress / YOUNG - 0.5 * plastic,
            np.zeros_like(stress),
        ),
        axis=1,
    )


def _equibiaxial_path() -> np.ndarray:
    stress = np.linspace(0.0, 410.0, 31)
    plastic = _plastic_strain_from_stress(stress)
    in_plane = (1.0 - POISSON) * stress / YOUNG + 0.5 * plastic
    return np.stack((in_plane, in_plane, np.zeros_like(stress)), axis=1)


def _pure_shear_path() -> np.ndarray:
    shear_stress = np.linspace(0.0, 260.0, 31)
    equivalent_stress = np.sqrt(3.0) * shear_stress
    plastic = _plastic_strain_from_stress(equivalent_stress)
    shear_modulus = YOUNG / (2.0 * (1.0 + POISSON))
    engineering_shear = shear_stress / shear_modulus + np.sqrt(3.0) * plastic
    return np.stack(
        (
            np.zeros_like(shear_stress),
            np.zeros_like(shear_stress),
            engineering_shear,
        ),
        axis=1,
    )


def _segment(start: np.ndarray, stop: np.ndarray, count: int = 12) -> np.ndarray:
    return np.linspace(start, stop, count + 1)[1:]


def _nonproportional_path() -> np.ndarray:
    tension = _uniaxial_path()
    first = tension[-1]
    combined = first + np.array([0.0, 0.0, 0.012])
    partially_unloaded = combined + np.array([-0.004, 0.001, -0.004])
    return np.concatenate(
        (
            tension,
            _segment(first, combined),
            _segment(combined, partially_unloaded),
        )
    )


def _python_history(path: Iterable[np.ndarray]) -> tuple[FullTensorState, float]:
    elasticity = plane_stress_elasticity(YOUNG, POISSON)
    metric_product = elasticity @ PLANE_STRESS_VON_MISES_METRIC
    hardening, _ = make_hardening(
        EXPONENT,
        "ludwik",
        0.2,
        1000,
        1e-6,
    )
    plastic = np.zeros((1, 3))
    accumulated = np.zeros(1)
    stress = np.zeros((1, 3))
    total = np.zeros((1, 3))
    for target in path:
        total[0] = target
        trial = (elasticity @ (total - plastic).T).T
        stress, increment, equivalent_increment = return_mapping(
            trial,
            accumulated,
            np.array([YIELD]),
            np.array([HARDENING]),
            hardening,
            metric_product[0, 0],
            metric_product[0, 1],
            metric_product[2, 2],
        )
        plastic += increment
        accumulated += equivalent_increment
    state = reconstruct_python_plane_stress_state(total, plastic, stress, POISSON)
    return state, float(accumulated[0])


def _mfront_history(path: np.ndarray) -> tuple[FullTensorState, float]:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    batch = MFrontMaterialPointBatch(library, YIELD, HARDENING, EXPONENT)
    state: FullTensorState | None = None
    accumulated = 0.0
    for target in path:
        result = batch.evaluate(
            target[None, :],
            time_increment=1.0 / len(path),
        )
        state = batch.current_full_tensor_state()
        accumulated = float(result.equivalent_plastic_strain[0])
        batch.commit()
    assert state is not None
    return state, accumulated


@pytest.mark.parametrize(
    ("path_factory", "expected_axial_relation"),
    [
        (_uniaxial_path, "uniaxial"),
        (_equibiaxial_path, "equibiaxial"),
        (_pure_shear_path, "pure_shear"),
    ],
)
def test_proportional_python_paths_reconstruct_expected_plastic_state(
    path_factory,
    expected_axial_relation: str,
) -> None:
    state, accumulated = _python_history(path_factory())
    plastic = state.plastic_strain_tensor[0]

    assert np.trace(plastic) == pytest.approx(0.0, abs=1e-14)
    if expected_axial_relation == "uniaxial":
        assert plastic[2, 2] == pytest.approx(-0.5 * plastic[0, 0], abs=1e-12)
    elif expected_axial_relation == "equibiaxial":
        assert plastic[0, 0] == pytest.approx(plastic[1, 1], abs=1e-12)
        assert plastic[2, 2] == pytest.approx(-2.0 * plastic[0, 0], abs=1e-12)
    else:
        assert plastic[2, 2] == pytest.approx(0.0, abs=1e-14)
    assert instantaneous_equivalent_plastic_strain(plastic) == pytest.approx(
        accumulated,
        rel=1e-8,
        abs=1e-12,
    )


@pytest.mark.mfront
@pytest.mark.parametrize("path_factory", [_uniaxial_path, _equibiaxial_path, _pure_shear_path])
def test_proportional_mfront_and_python_tensor_states_agree(path_factory) -> None:
    path = path_factory()
    python_state, python_accumulated = _python_history(path)
    mfront_state, mfront_accumulated = _mfront_history(path)

    np.testing.assert_allclose(
        mfront_state.stress_tensor_mpa,
        python_state.stress_tensor_mpa,
        rtol=1e-6,
        atol=1e-6,
    )
    for field_name in (
        "total_strain_tensor",
        "elastic_strain_tensor",
        "plastic_strain_tensor",
    ):
        np.testing.assert_allclose(
            getattr(mfront_state, field_name),
            getattr(python_state, field_name),
            rtol=1e-6,
            atol=1e-10,
        )
    assert mfront_accumulated == pytest.approx(python_accumulated, rel=1e-6, abs=1e-10)
    assert abs(mfront_state.plane_stress_residual_mpa[0]) <= 1e-9


def test_unloading_preserves_converged_plastic_axial_strain() -> None:
    loading = _uniaxial_path()
    elastic_unload = np.array([-100.0 / YOUNG, POISSON * 100.0 / YOUNG, 0.0])
    unloading = _segment(loading[-1], loading[-1] + elastic_unload, count=20)
    loaded_state, _ = _python_history(loading)
    unloaded_state, _ = _python_history(np.concatenate((loading, unloading)))

    np.testing.assert_allclose(
        unloaded_state.plastic_strain_tensor,
        loaded_state.plastic_strain_tensor,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.trace(unloaded_state.plastic_strain_tensor[0]) == pytest.approx(
        0.0,
        abs=1e-14,
    )


@pytest.mark.mfront
def test_nonproportional_path_distinguishes_tensor_norm_from_accumulated_peeq() -> None:
    path = _nonproportional_path()
    python_state, python_accumulated = _python_history(path)
    mfront_state, mfront_accumulated = _mfront_history(path)
    python_norm = float(
        instantaneous_equivalent_plastic_strain(python_state.plastic_strain_tensor)[0]
    )

    assert abs(python_norm - python_accumulated) > 1e-5
    assert mfront_accumulated == pytest.approx(python_accumulated, rel=1e-6, abs=1e-10)
    np.testing.assert_allclose(
        mfront_state.plastic_strain_tensor,
        python_state.plastic_strain_tensor,
        rtol=1e-6,
        atol=1e-10,
    )
