"""The Broyden correction as the Newton loop sees it.

Sections 15 to 19 and 22.1 to 22.3 of the 2026-08-04 specification. The algebra
is tested in `test_limited_memory_broyden.py`; what is tested here is the
contract with the solver -- that the correction reaches the matrix and nothing
else, that its memory is a device of the increment, and that a correction which
turns out to be useless is still harmless.

The solver-level cases run the in-house Python J2 backend, so they need no
MFront library and can run in CI. The verdict on whether the correction is worth
having is not theirs to give: it is measured on the registered SRIX case by
`scripts/qualify_broyden_correction.py` and recorded in
`validation/cps4r_as_broyden_results.md`.
"""

from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.config import CaseStudyConfig, MaterialConfig, MeshConfig, SolverConfig
from fem_inhouse.core.assumed_strain import batched_stabilisation, central_operators
from fem_inhouse.core.element import plane_stress_elasticity
from fem_inhouse.core.jacobian_correction import (
    BroydenHourglassCorrection,
    NoJacobianCorrection,
    make_jacobian_correction,
)
from fem_inhouse.core.limited_memory_broyden import ElementBroydenMemories
from fem_inhouse.solver import run_case_study

ELASTICITY = plane_stress_elasticity(205_000.0, 0.3)
NODES = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
OPERATORS = central_operators(NODES)


def _rigid_modes() -> list[np.ndarray]:
    return [
        np.tile([1.0, 0.0], 4),
        np.tile([0.0, 1.0], 4),
        np.array([value for x, y in NODES for value in (-y, x)], dtype=float),
    ]


def _states(count: int, seed: int) -> np.ndarray:
    """Displacements with a genuine hourglass content, in the plastic range."""

    generator = np.random.default_rng(seed)
    return 1.0e-3 * generator.standard_normal((count, 8))


def _forces(displacements: np.ndarray) -> np.ndarray:
    tangents = np.repeat(ELASTICITY[None, :, :], displacements.shape[0], axis=0)
    force, _, _ = batched_stabilisation(OPERATORS, displacements, tangents)
    return force


def _tangents(count: int) -> np.ndarray:
    _, stiffness, _ = batched_stabilisation(
        OPERATORS,
        np.zeros((count, 8)),
        np.repeat(ELASTICITY[None, :, :], count, axis=0),
    )
    return stiffness


class TestRegistry:
    def test_none_is_an_object_and_does_nothing(self) -> None:
        correction = make_jacobian_correction("none")
        assert isinstance(correction, NoJacobianCorrection)
        assert correction.name == "none"
        correction.begin_increment()
        correction.observe(np.zeros((2, 8)), np.zeros((2, 8)))
        assert correction.matrix(np.zeros((2, 8, 8))) is None
        correction.discard()
        assert correction.diagnostics == {}

    def test_broyden_without_a_geometry_says_which_formulation_it_needs(self) -> None:
        with pytest.raises(ValueError, match="cps4r_as"):
            make_jacobian_correction("broyden", operators=None, element_count=4)

    def test_an_unknown_name_lists_what_exists(self) -> None:
        with pytest.raises(ValueError, match="none, broyden"):
            make_jacobian_correction("bfgs")

    @pytest.mark.parametrize("memory", [0, 6, -1])
    def test_the_memory_is_bounded_by_the_reduced_dimension(self, memory: int) -> None:
        with pytest.raises(ValueError, match=r"1\.\.5"):
            make_jacobian_correction(
                "broyden", operators=OPERATORS, element_count=2, memory=memory
            )


class TestTransactions:
    """Section 17. The memory belongs to the increment, not to the material."""

    def test_one_observation_produces_no_correction(self) -> None:
        correction = BroydenHourglassCorrection(OPERATORS, 3)
        correction.begin_increment()
        states = _states(3, 11)
        correction.observe(states, _forces(states))
        # A pair needs two states; `None` rather than a zero array so the solver
        # skips the addition entirely.
        assert correction.matrix(_tangents(3)) is None

    def test_two_observations_produce_a_correction(self) -> None:
        correction = BroydenHourglassCorrection(OPERATORS, 3)
        correction.begin_increment()
        for seed in (11, 12):
            states = _states(3, seed)
            correction.observe(states, _forces(states))
        matrix = correction.matrix(_tangents(3))
        assert matrix is not None
        assert matrix.shape == (3, 8, 8)
        assert correction.diagnostics["broyden_pairs_accepted"] == 3.0

    def test_beginning_an_increment_forgets_the_previous_one(self) -> None:
        correction = BroydenHourglassCorrection(OPERATORS, 2)
        correction.begin_increment()
        for seed in (21, 22):
            states = _states(2, seed)
            correction.observe(states, _forces(states))
        assert correction.matrix(_tangents(2)) is not None
        correction.begin_increment()
        assert correction.matrix(_tangents(2)) is None
        # And the state is forgotten too, so the first observation of the new
        # increment cannot be paired with the last of the old one.
        states = _states(2, 23)
        correction.observe(states, _forces(states))
        assert correction.matrix(_tangents(2)) is None

    def test_a_cutback_purges_the_memory_and_says_so(self) -> None:
        correction = BroydenHourglassCorrection(OPERATORS, 2)
        correction.begin_increment()
        for seed in (31, 32):
            states = _states(2, seed)
            correction.observe(states, _forces(states))
        assert correction.matrix(_tangents(2)) is not None
        correction.discard()
        assert correction.matrix(_tangents(2)) is None
        assert correction.diagnostics["broyden_memory_purges"] == 1.0

    def test_a_force_outside_the_hourglass_span_excludes_its_element(self) -> None:
        """Section 7. A projection that loses part of the force teaches a lie."""

        correction = BroydenHourglassCorrection(OPERATORS, 2)
        correction.begin_increment()
        for seed in (41, 42):
            states = _states(2, seed)
            forces = _forces(states)
            # A pure translation force: orthogonal to both hourglass modes, so
            # the modal projection cannot represent it at all.
            forces[1] = 1.0e-3 * np.tile([1.0, 0.0], 4)
            correction.observe(states, forces)
        assert correction.diagnostics["broyden_elements_excluded"] >= 1.0
        assert correction.diagnostics["broyden_maximum_projection_defect"] > 1.0e-10
        matrix = correction.matrix(_tangents(2))
        assert matrix is not None
        # The healthy element is corrected; the excluded one is not touched.
        assert np.linalg.norm(matrix[0]) > 0.0
        np.testing.assert_allclose(matrix[1], 0.0, atol=0.0)

    def test_a_masked_pair_is_not_counted_as_refused(self) -> None:
        memories = ElementBroydenMemories(2, memory=3)
        accepted = memories.add_batch(
            np.ones((2, 5)),
            np.ones((2, 2)),
            np.ones(2),
            mask=np.array([True, False]),
        )
        assert accepted == 1
        assert memories.total_rejected == 0


class TestStructuralProperties:
    def test_the_correction_puts_no_force_on_a_rigid_mode(self) -> None:
        """Section 22.5, at the level the solver actually assembles."""

        correction = BroydenHourglassCorrection(OPERATORS, 2)
        correction.begin_increment()
        for seed in (51, 52, 53):
            states = _states(2, seed)
            correction.observe(states, _forces(states))
        matrix = correction.matrix(_tangents(2))
        assert matrix is not None
        scale = float(np.abs(matrix).max())
        assert scale > 0.0
        for mode in _rigid_modes():
            np.testing.assert_allclose(
                matrix @ mode, 0.0, atol=1.0e-12 * scale * np.linalg.norm(mode)
            )

    def test_the_correction_produces_only_hourglass_force(self) -> None:
        correction = BroydenHourglassCorrection(OPERATORS, 2)
        correction.begin_increment()
        for seed in (61, 62, 63):
            states = _states(2, seed)
            correction.observe(states, _forces(states))
        matrix = correction.matrix(_tangents(2))
        assert matrix is not None
        gamma = OPERATORS.gamma
        modes = np.zeros((2, 8))
        modes[0, 0::2] = gamma
        modes[1, 1::2] = gamma
        projector = modes.T @ np.linalg.solve(modes @ modes.T, modes)
        for element in matrix:
            np.testing.assert_allclose(
                projector @ element, element, atol=1.0e-10 * np.abs(element).max()
            )


def _case(mesh_size: int = 6) -> dict[str, object]:
    mesh = MeshConfig(nx=mesh_size, ny=mesh_size, base_pixel_size_mm=0.00184)
    span = mesh.physical_size_mm[0]
    nodes = np.linspace(0.0, span, mesh_size + 1)
    grid_x, grid_y = np.meshgrid(nodes, nodes, indexing="ij")
    perturbation = 0.05 * 0.010 * span
    return {
        "mesh": mesh,
        "displacement_x_mm": -0.004 * grid_x
        + perturbation * np.sin(2.0 * np.pi * grid_y / span) * (grid_x / span),
        "displacement_y_mm": 0.010 * grid_y
        + perturbation * np.sin(2.0 * np.pi * grid_x / span) * (grid_y / span),
        "yield_stress_mpa": np.full((mesh_size, mesh_size), 250.0),
        "hardening_coefficient_mpa": np.full((mesh_size, mesh_size), 500.0),
    }


def _solve(case: dict[str, object], correction: str, memory: int = 5):
    solver = SolverConfig(
        increments=3,
        constitutive_backend="python",
        element_formulation="cps4r_as",
        jacobian_correction=correction,
        jacobian_correction_memory=memory,
    )
    configuration = CaseStudyConfig(
        mesh=case["mesh"], material=MaterialConfig(), solver=solver
    )
    return run_case_study(
        configuration,
        displacement_x_mm=case["displacement_x_mm"],
        displacement_y_mm=case["displacement_y_mm"],
        yield_stress_mpa=case["yield_stress_mpa"],
        hardening_coefficient_mpa=case["hardening_coefficient_mpa"],
    )


class TestSolverContract:
    """Sections 22.1 and 22.2, on a case small enough to run in CI."""

    def test_the_converged_solution_does_not_move(self) -> None:
        case = _case()
        baseline = _solve(case, "none")
        corrected = _solve(case, "broyden")
        for field in ("displacement_mm", "stress_mpa", "reaction_force"):
            reference = np.asarray(getattr(baseline, field))
            candidate = np.asarray(getattr(corrected, field))
            error = np.linalg.norm(candidate - reference) / np.linalg.norm(reference)
            # Both runs stop on the SAME residual tolerance, so they land at two
            # points of one convergence ball rather than at the same point. The
            # bound is the tolerance, not machine precision.
            assert error < 1.0e-6, f"{field} moved by {error:.3e}"

    def test_no_constitutive_call_is_added(self) -> None:
        case = _case()
        baseline = _solve(case, "none")
        corrected = _solve(case, "broyden")
        assert baseline.diagnostics is not None
        assert corrected.diagnostics is not None
        assert corrected.diagnostics.gauss_points_per_element == 1
        assert (
            corrected.diagnostics.constitutive_material_point_count
            == baseline.diagnostics.constitutive_material_point_count
        )

    def test_the_run_records_which_correction_it_used(self) -> None:
        result = _solve(_case(), "broyden", memory=3)
        assert result.diagnostics is not None
        assert result.diagnostics.jacobian_correction == "broyden"
        measured = result.diagnostics.jacobian_correction_diagnostics
        assert measured["broyden_memory"] == 3.0
        assert measured["broyden_pairs_accepted"] > 0.0
        # Section 7 again, this time over a whole solve: the stabilising force
        # never left the hourglass span, so no element was ever excluded.
        assert measured["broyden_elements_excluded"] == 0.0

    def test_the_default_run_says_it_used_no_correction(self) -> None:
        result = _solve(_case(), "none")
        assert result.diagnostics is not None
        assert result.diagnostics.jacobian_correction == "none"
        assert result.diagnostics.jacobian_correction_diagnostics == {}


class TestConfiguration:
    def test_the_correction_is_refused_on_a_formulation_without_stabilisation(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="cps4r_as"):
            SolverConfig(element_formulation="cps4", jacobian_correction="broyden")

    @pytest.mark.parametrize("memory", [0, 6])
    def test_the_memory_is_validated_when_the_configuration_is_read(
        self, memory: int
    ) -> None:
        with pytest.raises(ValueError, match="jacobian_correction_memory"):
            SolverConfig(
                element_formulation="cps4r_as",
                jacobian_correction="broyden",
                jacobian_correction_memory=memory,
            )

    def test_an_unknown_correction_fails_before_any_solve(self) -> None:
        with pytest.raises(ValueError, match="unknown jacobian_correction"):
            SolverConfig(
                element_formulation="cps4r_as",
                jacobian_correction="bfgs",  # type: ignore[arg-type]
            )
