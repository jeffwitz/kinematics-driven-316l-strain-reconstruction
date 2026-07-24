import numpy as np
import pytest

from fem_inhouse.material import LudwikLaw, abaqus_plastic_table


def test_ludwik_law_is_monotonic_and_starts_at_yield() -> None:
    law = LudwikLaw(124.0, 380.0, 0.245)
    strain = np.array([0.0, 1e-6, 0.01, 0.2])
    stress = law.stress(strain)

    assert stress[0] == pytest.approx(124.0)
    assert np.all(np.diff(stress) > 0)


def test_ludwik_tangent_is_zero_at_origin_and_positive_afterwards() -> None:
    law = LudwikLaw(124.0, 380.0, 0.245)
    tangent = law.tangent(np.array([0.0, 1e-6, 0.1]))
    assert tangent[0] == 0.0
    assert np.all(tangent[1:] > 0)


def test_article_abaqus_table_records_grid_choices() -> None:
    law = LudwikLaw(124.0, 380.0, 0.245)
    table = abaqus_plastic_table(law)

    assert table.shape == (1_000, 2)
    assert table[0, 1] == 0.0
    assert table[1, 1] == pytest.approx(1e-6)
    assert table[-1, 1] == pytest.approx(0.2)
    assert table[0, 0] == pytest.approx(124.0)
    assert np.all(np.diff(table[:, 0]) > 0)
    assert np.all(np.diff(table[:, 1]) > 0)


def test_uniform_grid_is_available_for_legacy_comparison() -> None:
    law = LudwikLaw(124.0, 380.0, 0.245)
    table = abaqus_plastic_table(law, n_points=11, first_positive_strain=None)
    assert np.diff(table[:, 1]) == pytest.approx(np.full(10, 0.02))


@pytest.mark.parametrize(
    "law",
    [
        lambda: LudwikLaw(0.0, 380.0, 0.245),
        lambda: LudwikLaw(124.0, -1.0, 0.245),
        lambda: LudwikLaw(124.0, 380.0, 0.0),
    ],
)
def test_invalid_ludwik_parameters_are_rejected(law) -> None:
    with pytest.raises(ValueError):
        law()


def test_negative_plastic_strain_is_rejected() -> None:
    law = LudwikLaw(124.0, 380.0, 0.245)
    with pytest.raises(ValueError, match="cannot be negative"):
        law.stress([-1e-6])
    with pytest.raises(ValueError, match="cannot be negative"):
        law.tangent([-1e-6])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"plastic_strain_max": 0},
        {"n_points": 2},
        {"first_positive_strain": 0.2},
    ],
)
def test_invalid_table_grid_is_rejected(kwargs) -> None:
    law = LudwikLaw(124.0, 380.0, 0.245)
    with pytest.raises(ValueError):
        abaqus_plastic_table(law, **kwargs)
