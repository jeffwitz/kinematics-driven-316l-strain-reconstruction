"""Reduced coordinates and the multisecant Broyden correction.

Sections 6 to 14 and 22.4 to 22.8 of the 2026-08-04 specification. Pure linear
algebra: no solver, no material model, no MFront, and by construction no
constitutive call -- the correction is learned from secant pairs the iteration
has already produced.
"""

from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.core.assumed_strain import AssumedStrainStabilisation, central_operators
from fem_inhouse.core.element import plane_stress_elasticity
from fem_inhouse.core.hourglass_modal_coordinates import (
    MODAL_PROJECTION_TOLERANCE,
    modal_coordinates,
)
from fem_inhouse.core.limited_memory_broyden import (
    MAXIMUM_MEMORY,
    BroydenMemory,
    ElementBroydenMemories,
    build_correction,
)

ELASTICITY = plane_stress_elasticity(205_000.0, 0.3)
GEOMETRIES: dict[str, np.ndarray] = {
    "unit_square": np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    "parallelogram": np.array([[0.0, 0.0], [2.0, 0.0], [2.7, 1.3], [0.7, 1.3]]),
    "distorted": np.array([[0.0, 0.0], [2.0, 0.3], [1.4, 1.6], [0.15, 0.9]]),
}


def _rigid_modes(nodes: np.ndarray) -> list[np.ndarray]:
    translation_x = np.tile([1.0, 0.0], 4)
    translation_y = np.tile([0.0, 1.0], 4)
    rotation = np.array(
        [value for x, y in nodes for value in (-y, x)], dtype=float
    )
    return [translation_x, translation_y, rotation]


def _affine(nodes: np.ndarray) -> np.ndarray:
    return np.array(
        [
            value
            for x, y in nodes
            for value in (1.3e-3 * x - 4.0e-4 * y, 7.0e-4 * x + 2.2e-3 * y)
        ]
    )


class TestModalCoordinates:
    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    def test_the_stabilising_force_lives_in_the_two_hourglass_modes(
        self, name: str
    ) -> None:
        """Section 7. Verified, not assumed: an element failing it is excluded."""

        nodes = GEOMETRIES[name]
        operators = central_operators(nodes)
        coordinates = modal_coordinates(operators)
        force = AssumedStrainStabilisation().evaluate(
            operators, np.arange(8, dtype=float) * 1e-4, ELASTICITY
        ).internal_force

        defect = float(coordinates.modal_projection_defect(force)[0])

        assert defect < MODAL_PROJECTION_TOLERANCE, (name, defect)

    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    def test_the_reduced_state_is_the_central_strain_and_the_amplitudes(
        self, name: str
    ) -> None:
        nodes = GEOMETRIES[name]
        operators = central_operators(nodes)
        coordinates = modal_coordinates(operators)
        displacement = np.arange(8, dtype=float) * 1e-4

        reduced = coordinates.reduced_state(displacement)

        assert reduced.shape == (5,)
        np.testing.assert_allclose(
            reduced[:3], operators.strain_displacement_centre @ displacement, atol=1e-18
        )
        np.testing.assert_allclose(
            reduced[3], operators.gamma @ displacement[0::2], atol=1e-18
        )

    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    def test_a_rigid_motion_has_a_zero_reduced_state(self, name: str) -> None:
        coordinates = modal_coordinates(central_operators(GEOMETRIES[name]))

        for mode in _rigid_modes(GEOMETRIES[name]):
            assert np.abs(coordinates.reduced_state(mode)).max() < 1e-14

    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    def test_an_affine_field_has_zero_hourglass_amplitudes(self, name: str) -> None:
        coordinates = modal_coordinates(central_operators(GEOMETRIES[name]))

        reduced = coordinates.reduced_state(_affine(GEOMETRIES[name]))

        assert np.abs(reduced[3:]).max() < 1e-16
        assert np.abs(reduced[:3]).max() > 1e-6

    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    def test_the_expanded_correction_kills_every_rigid_mode(self, name: str) -> None:
        """Section 22.7, and it is structural: `H` annihilates a rigid motion."""

        nodes = GEOMETRIES[name]
        coordinates = modal_coordinates(central_operators(nodes))
        correction = coordinates.expand_correction(
            np.random.default_rng(0).normal(size=(2, 5))
        )

        for mode in _rigid_modes(nodes):
            assert np.abs(correction @ mode).max() < 1e-12

    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    def test_the_correction_produces_only_hourglass_force(self, name: str) -> None:
        """Whatever it learns, `H^T` puts the output in the two modes."""

        nodes = GEOMETRIES[name]
        coordinates = modal_coordinates(central_operators(nodes))
        correction = coordinates.expand_correction(
            np.random.default_rng(1).normal(size=(2, 5))
        )
        produced = correction @ np.arange(8, dtype=float) * 1e-4

        assert float(coordinates.modal_projection_defect(produced)[0]) < 1e-12

    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    def test_an_affine_direction_is_not_in_the_kernel_of_the_correction(
        self, name: str
    ) -> None:
        """A tension in the specification, recorded rather than papered over.

        Section 8 states the construction guarantees that "le champ affine n'est
        pas modifié", and section 22.8 asks for no extra stabilising force on an
        affine field. Measured here: `K_B u_affine` is **not** zero, at `9e-4`
        for the unit square with a unit-scale reduced correction.

        That is not a defect of the construction; it is the term being learned.
        `T = [B_c; H]`, and an affine field has zero hourglass amplitudes but a
        NON-ZERO central strain. The missing Jacobian term is exactly
        `(df_stab/dC)(dC/du)`, and `C` moves with the central strain -- so a
        correction that annihilated affine directions would be unable to learn
        the very thing it exists for.

        What remains true, and is what the patch test actually needs: the
        residual is untouched, and at an affine state the hourglass amplitudes
        are zero so the stabilising FORCE is zero. The correction changes the
        matrix, never the force. Whether section 22.8 should be read as a
        statement about the force or about the matrix is a question for the
        specification, not something to settle by zeroing three columns.
        """

        nodes = GEOMETRIES[name]
        coordinates = modal_coordinates(central_operators(nodes))
        correction = coordinates.expand_correction(
            np.random.default_rng(2).normal(size=(2, 5))
        )

        assert np.abs(correction @ _affine(nodes)).max() > 1e-6

    def test_the_reduced_jacobian_has_the_declared_shape(self) -> None:
        operators = central_operators(GEOMETRIES["distorted"])
        coordinates = modal_coordinates(operators)
        tangent = AssumedStrainStabilisation().evaluate(
            operators, np.zeros(8), ELASTICITY
        ).tangent

        assert coordinates.reduced_jacobian(tangent).shape == (2, 5)
        assert coordinates.reduced_jacobian(np.stack([tangent] * 3)).shape == (3, 2, 5)

    def test_a_misshapen_tangent_is_refused(self) -> None:
        coordinates = modal_coordinates(central_operators(GEOMETRIES["unit_square"]))

        with pytest.raises(ValueError, match=r"\(8, 8\)"):
            coordinates.reduced_jacobian(np.zeros((3, 3)))


class TestMemory:
    def test_a_memory_beyond_the_reduced_dimension_is_refused(self) -> None:
        """More pairs than the input dimension add no independent information."""

        with pytest.raises(ValueError, match="carry no"):
            BroydenMemory(memory=MAXIMUM_MEMORY + 1)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_non_positive_memory_is_refused(self, bad: int) -> None:
        with pytest.raises(ValueError, match="memory must lie"):
            BroydenMemory(memory=bad)

    def test_pairs_are_normalised_by_the_step_length(self) -> None:
        """Section 11: preserves the secant condition, improves conditioning."""

        memory = BroydenMemory(memory=3)
        step = np.array([3.0, 4.0, 0.0, 0.0, 0.0])

        assert memory.add(step, np.array([10.0, 5.0]), scale=1.0)
        np.testing.assert_allclose(np.linalg.norm(memory.steps[0]), 1.0)
        np.testing.assert_allclose(memory.increments[0], np.array([2.0, 1.0]))

    def test_a_vanishing_step_is_rejected_and_counted(self) -> None:
        memory = BroydenMemory(memory=3)

        assert not memory.add(np.zeros(5), np.ones(2), scale=1.0)
        assert memory.pair_count == 0
        assert memory.rejected == 1

    def test_a_non_finite_pair_is_rejected(self) -> None:
        memory = BroydenMemory(memory=3)

        assert not memory.add(np.ones(5), np.array([np.nan, 1.0]), scale=1.0)
        assert memory.rejected == 1

    def test_the_memory_is_circular(self) -> None:
        memory = BroydenMemory(memory=2)
        generator = np.random.default_rng(3)
        for _ in range(5):
            memory.add(generator.normal(size=5), generator.normal(size=2), scale=1.0)

        assert memory.pair_count == 2

    def test_clearing_empties_it(self) -> None:
        memory = BroydenMemory(memory=3)
        memory.add(np.ones(5), np.ones(2), scale=1.0)
        memory.clear()

        assert memory.pair_count == 0


class TestCorrection:
    @staticmethod
    def _exact(pairs: int, seed: int = 4):
        generator = np.random.default_rng(seed)
        base = generator.normal(size=(2, 5))
        truth = base + generator.normal(size=(2, 5))
        memory = BroydenMemory(memory=max(pairs, 1))
        for _ in range(pairs):
            step = generator.normal(size=5)
            memory.add(step, truth @ step, scale=1.0)
        return base, truth, memory

    def test_an_empty_memory_gives_exactly_no_correction(self) -> None:
        """Section 22.1: with no pairs the element must be bit-for-bit unchanged."""

        base, _, memory = self._exact(0)
        result = build_correction(memory, base)

        assert np.all(result.correction == 0.0)
        assert result.rank == 0

    @pytest.mark.parametrize("pairs", [1, 2, 3, 4, 5])
    def test_the_secant_conditions_are_satisfied(self, pairs: int) -> None:
        """Sections 22.4 and 22.5, from one direction up to the full dimension."""

        base, _, memory = self._exact(pairs)
        result = build_correction(memory, base)
        steps = np.array(memory.steps).T
        increments = np.array(memory.increments).T

        assert result.rank == pairs
        assert np.abs((base + result.correction) @ steps - increments).max() < 1e-12
        assert result.secant_defect_after < 1e-12
        assert result.secant_defect_before > result.secant_defect_after

    def test_colinear_directions_reduce_the_rank_without_blowing_up(self) -> None:
        """Section 22.6: no NaN, no singular inverse, no arbitrary correction."""

        generator = np.random.default_rng(5)
        base = generator.normal(size=(2, 5))
        truth = base + generator.normal(size=(2, 5))
        direction = generator.normal(size=5)
        memory = BroydenMemory(memory=3)
        for factor in (1.0, 2.0, 3.0):
            memory.add(factor * direction, truth @ (factor * direction), scale=1.0)

        result = build_correction(memory, base)

        assert result.rank == 1
        assert np.isfinite(result.correction).all()
        assert np.linalg.norm(result.correction) < 10.0 * np.linalg.norm(truth - base)

    def test_the_minimum_norm_solution_is_the_one_returned(self) -> None:
        """`ZS^+` and not any other secant-satisfying correction."""

        base, _, memory = self._exact(2, seed=6)
        result = build_correction(memory, base)
        steps = np.array(memory.steps).T
        defects = np.array(memory.increments).T - base @ steps

        # Any correction of the form `ZS^+ + W` with `W S = 0` also satisfies the
        # secant conditions and must have a strictly larger Frobenius norm.
        null_direction = np.linalg.svd(steps.T, full_matrices=True)[2][-1]
        alternative = result.correction + np.outer(np.ones(2), null_direction)

        assert np.abs(alternative @ steps - defects).max() < 1e-12
        assert np.linalg.norm(result.correction) < np.linalg.norm(alternative)

    def test_a_non_finite_base_jacobian_falls_back_to_no_correction(self) -> None:
        """Section 28: deterministic fallback, never an exception."""

        _, _, memory = self._exact(2)
        result = build_correction(memory, np.full((2, 5), np.nan))

        assert np.all(result.correction == 0.0)
        assert result.rank == 0

    def test_a_misshapen_base_jacobian_is_refused(self) -> None:
        _, _, memory = self._exact(1)

        with pytest.raises(ValueError, match=r"\(2, 5\)"):
            build_correction(memory, np.zeros((3, 3)))

    def test_a_tighter_rank_tolerance_keeps_fewer_directions(self) -> None:
        generator = np.random.default_rng(7)
        base = generator.normal(size=(2, 5))
        truth = base + generator.normal(size=(2, 5))
        memory = BroydenMemory(memory=3)
        principal = generator.normal(size=5)
        for perturbation in (0.0, 1e-9, 1e-6):
            step = principal + perturbation * generator.normal(size=5)
            memory.add(step, truth @ step, scale=1.0)

        loose = build_correction(memory, base, rank_tolerance=1e-12).rank
        tight = build_correction(memory, base, rank_tolerance=1e-3).rank

        assert tight <= loose


class TestElementMemories:
    def test_every_element_has_its_own_memory(self) -> None:
        memories = ElementBroydenMemories(4, memory=3)
        generator = np.random.default_rng(8)
        steps = generator.normal(size=(4, 5))
        forces = generator.normal(size=(4, 2))

        accepted = memories.add_batch(steps, forces, np.ones(4))

        assert accepted == 4
        assert memories.total_pairs == 4
        assert len(memories) == 4

    def test_a_degenerate_element_is_skipped_without_affecting_the_others(self) -> None:
        """Section 28: the fallback is local."""

        memories = ElementBroydenMemories(3, memory=3)
        steps = np.array([np.ones(5), np.zeros(5), np.ones(5)])
        forces = np.ones((3, 2))

        assert memories.add_batch(steps, forces, np.ones(3)) == 2
        assert memories.total_rejected == 1

    def test_clearing_empties_every_element(self) -> None:
        """Section 17: the memory is a device of the increment, not material state."""

        memories = ElementBroydenMemories(3, memory=3)
        memories.add_batch(
            np.ones((3, 5)), np.ones((3, 2)), np.ones(3)
        )
        memories.clear()

        assert memories.total_pairs == 0

    def test_the_batch_build_reports_the_declared_diagnostics(self) -> None:
        memories = ElementBroydenMemories(3, memory=3)
        generator = np.random.default_rng(9)
        base = generator.normal(size=(3, 2, 5))
        for _ in range(2):
            memories.add_batch(
                generator.normal(size=(3, 5)), generator.normal(size=(3, 2)), np.ones(3)
            )

        corrections, diagnostics = memories.build_batch(base)

        assert corrections.shape == (3, 2, 5)
        assert set(diagnostics) >= {
            "broyden_pairs_total",
            "broyden_pairs_mean",
            "broyden_pairs_rejected",
            "broyden_rank_mean",
            "broyden_rank_max",
            "broyden_correction_norm",
            "broyden_secant_defect_before",
            "broyden_secant_defect_after",
            "broyden_elements_without_correction",
        }
        assert diagnostics["broyden_pairs_total"] == 6.0

    def test_an_empty_batch_gives_exactly_zero_corrections(self) -> None:
        memories = ElementBroydenMemories(2, memory=3)

        corrections, diagnostics = memories.build_batch(np.zeros((2, 2, 5)))

        assert np.all(corrections == 0.0)
        assert diagnostics["broyden_elements_without_correction"] == 2.0

    def test_a_non_positive_element_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="element_count"):
            ElementBroydenMemories(0, memory=3)
