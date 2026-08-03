"""CPS4R through the whole solver, against CPS4.

Sections 15.8 and 16. Small and analytical: the loading is affine, so the two
formulations must agree exactly and the hourglass energy must vanish. That is
the strongest comparison available, and any disagreement is a defect rather
than the expected difference between a full and a reduced element.
"""

from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.config import SolverConfig
from fem_inhouse.core.nonlinear import run_fem

MESH = 6
SIZE = 0.006
ELEMENT = 0.001


def _affine_boundary(x_gradient: float, y_gradient: float) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.linspace(0.0, SIZE, MESH + 1)
    grid_x, grid_y = np.meshgrid(nodes, nodes, indexing="ij")
    return grid_x * x_gradient, grid_y * y_gradient


def _solve(formulation: str, **overrides) -> dict:
    displacement_x, displacement_y = _affine_boundary(0.33, -0.10)
    return run_fem(
        displacement_x,
        displacement_y,
        np.full((MESH, MESH), 124.0),
        np.full((MESH, MESH), 380.0),
        0.245,
        SIZE,
        SIZE,
        ELEMENT,
        1.0,
        N_inc=4,
        constitutive_backend="python",
        element_formulation=formulation,
        verbose=False,
        **overrides,
    )


@pytest.fixture(scope="module")
def solutions() -> dict[str, dict]:
    return {name: _solve(name) for name in ("cps4", "cps4r")}


def test_the_reduced_element_uses_one_material_point_per_element(
    solutions: dict[str, dict],
) -> None:
    """The whole point: four times fewer constitutive integrations."""

    full = solutions["cps4"]["CONSTITUTIVE_MATERIAL_POINT_COUNT"]
    reduced = solutions["cps4r"]["CONSTITUTIVE_MATERIAL_POINT_COUNT"]

    assert solutions["cps4"]["GAUSS_POINTS_PER_ELEMENT"] == 4
    assert solutions["cps4r"]["GAUSS_POINTS_PER_ELEMENT"] == 1
    assert full == 4 * reduced
    assert reduced == MESH * MESH


def test_an_affine_load_gives_the_same_answer_either_way(
    solutions: dict[str, dict],
) -> None:
    """Every Gauss point sees the same strain, so nothing can differ.

    This is the sharp test. Under a general load the two formulations are
    genuinely different elements and are not expected to agree; here they must.
    """

    full, reduced = solutions["cps4"], solutions["cps4r"]

    np.testing.assert_allclose(reduced["U"], full["U"], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(reduced["S"], full["S"], rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(reduced["PEEQ"], full["PEEQ"], rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(reduced["PE"], full["PE"], rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(reduced["RF"], full["RF"], rtol=1e-8, atol=1e-9)


def test_an_affine_load_excites_no_hourglass_energy(solutions: dict[str, dict]) -> None:
    reduced = solutions["cps4r"]

    assert reduced["HOURGLASS_ENERGY_RATIO"] < 1e-10
    assert np.abs(reduced["HOURGLASS_ENERGY_BY_ELEMENT"]).max() < 1e-12


def test_the_hourglass_energy_field_matches_the_mesh(solutions: dict[str, dict]) -> None:
    """Section 13 needs it spatially, not only as a global ratio."""

    field = solutions["cps4r"]["HOURGLASS_ENERGY_BY_ELEMENT"]

    assert field.shape == (MESH, MESH)
    assert field.sum() == pytest.approx(solutions["cps4r"]["HOURGLASS_ENERGY"], abs=1e-18)


def test_the_full_formulation_reports_no_hourglass_energy(
    solutions: dict[str, dict],
) -> None:
    assert solutions["cps4"]["HOURGLASS_ENERGY"] == 0.0
    assert solutions["cps4"]["ELEMENT_FORMULATION"] == "cps4"


@pytest.mark.parametrize("scale", [0.1, 0.25, 0.5, 1.0])
def test_the_answer_survives_the_whole_range_of_beta(scale: float) -> None:
    """Section 16. No value is selected here; the numbers are produced.

    On an affine load every beta gives the same answer, because the modes the
    stabilisation acts on are not excited. That is the useful baseline: it says
    the scale cannot corrupt a well-posed problem, and leaves the choice of beta
    to a case that actually excites the modes.
    """

    reference = _solve("cps4")
    reduced = _solve("cps4r", hourglass_scale=scale)

    np.testing.assert_allclose(reduced["U"], reference["U"], rtol=1e-10, atol=1e-14)
    assert reduced["HOURGLASS_ENERGY_RATIO"] < 1e-10


def test_the_reduced_element_refuses_the_micromorphic_coupling() -> None:
    """Section 4. Not validated together, so not silently attempted.

    A hourglass mode inside a localisation band would be indistinguishable from
    the physics the coupling exists to capture.
    """

    with pytest.raises(ValueError, match="not validated with the micromorphic"):
        _solve("cps4r", nonlocal_plasticity_enabled=True)


def test_the_configuration_refuses_a_scale_on_the_full_element() -> None:
    """Silently ignoring it would suggest CPS4 was being stabilised."""

    with pytest.raises(ValueError, match="no meaning for the fully integrated"):
        SolverConfig(element_formulation="cps4", hourglass_scale=0.5)


@pytest.mark.parametrize("scale", [0.0, -1.0, 1.5])
def test_the_configuration_bounds_the_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="0 < beta <= 1"):
        SolverConfig(element_formulation="cps4r", hourglass_scale=scale)


def test_an_unknown_formulation_is_refused() -> None:
    with pytest.raises(ValueError, match="cps4, cps4r"):
        SolverConfig(element_formulation="cps8")  # type: ignore[arg-type]


def test_a_failure_threshold_that_is_met_stops_the_solve() -> None:
    """An impossible threshold, to prove the guard is wired at all."""

    with pytest.raises(Exception, match="hourglass energy ratio"):
        _solve("cps4r", hourglass_energy_failure_ratio=1e-30)

def _nonaffine_elastic_solve(formulation: str) -> dict:
    """Small Dirichlet case that genuinely excites the hourglass modes."""

    nodes = np.linspace(0.0, SIZE, MESH + 1)
    grid_x, grid_y = np.meshgrid(nodes, nodes, indexing="ij")
    displacement_x = (
        0.01 * grid_x
        + 2.0e-5 * (grid_x / SIZE) * np.sin(np.pi * grid_y / SIZE)
    )
    displacement_y = (
        -0.003 * grid_y
        + 1.0e-5 * (grid_y / SIZE) * np.sin(2.0 * np.pi * grid_x / SIZE)
    )
    return run_fem(
        displacement_x,
        displacement_y,
        np.full((MESH, MESH), 1.0e9),
        np.zeros((MESH, MESH)),
        0.245,
        SIZE,
        SIZE,
        ELEMENT,
        1.0,
        N_inc=2,
        constitutive_backend="python",
        element_formulation=formulation,
        verbose=False,
    )


def test_a_nonaffine_elastic_case_excites_the_stabilisation() -> None:
    """Unlike the affine baseline, this test makes the diagnostic observable."""

    reduced = _nonaffine_elastic_solve("cps4r")

    assert reduced["HOURGLASS_ENERGY"] > 0.0
    assert reduced["HOURGLASS_ENERGY_BY_ELEMENT"].max() > 0.0
    assert reduced["INTERNAL_WORK"] > 0.0
    assert reduced["HOURGLASS_ENERGY_RATIO"] == pytest.approx(
        reduced["HOURGLASS_ENERGY"] / reduced["INTERNAL_WORK"]
    )


def test_beta_one_recovers_cps4_on_a_nonaffine_elastic_case() -> None:
    """The full-minus-reduced control must work when its modes are active."""

    full = _nonaffine_elastic_solve("cps4")
    reduced = _nonaffine_elastic_solve("cps4r")

    np.testing.assert_allclose(reduced["U"], full["U"], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(reduced["S"], full["S"], rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(reduced["RF"], full["RF"], rtol=1e-9, atol=1e-9)


def test_internal_work_is_integrated_over_the_accepted_path() -> None:
    """For a linear path, the trapezoidal integral has an analytical value."""

    reduced = _nonaffine_elastic_solve("cps4r")
    analytical_work = 0.5 * abs(
        float(np.sum(reduced["RF"] * reduced["U"]))
    )

    assert reduced["INTERNAL_WORK"] == pytest.approx(
        analytical_work,
        rel=1e-10,
        abs=1e-18,
    )

