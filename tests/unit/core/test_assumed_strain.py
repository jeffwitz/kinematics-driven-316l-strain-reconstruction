"""Element algebra of the assumed-strain stabilisation, after QUAS4.

Section 12 and section 20 of the 2026-08-04 specification. Everything here is
pure linear algebra on one element: no solver, no material model, no MFront.

The consistency check the whole construction rests on is
`test_the_quad4_projection_reproduces_the_fully_integrated_stiffness`. With
`(e1, e2, e3) = (1, 0, 1)` the assumed strain *is* the QUAD4 strain, so
`K_c + K_stab` must equal the 2x2-integrated stiffness exactly. If that fails,
the decomposition is wrong and nothing else in this file means anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.core.assumed_strain import (
    PROJECTION_COEFFICIENTS,
    AssumedStrainStabilisation,
    EnergyProjectedAssumedStrainStabilisation,
    central_operators,
    enrichment_operator,
    hourglass_amplitudes,
    make_stabilisation,
    projection_coefficients,
)
from fem_inhouse.core.element import (
    CPS4_QUADRATURE,
    plane_stress_elasticity,
    strain_displacement_matrix,
)

ELASTICITY = plane_stress_elasticity(205_000.0, 0.3)

#: Geometries the specification names in section 12.5, plus the unit square.
GEOMETRIES: dict[str, np.ndarray] = {
    "unit_square": np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    "rectangle": np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 0.5], [0.0, 0.5]]),
    "parallelogram": np.array([[0.0, 0.0], [2.0, 0.0], [2.7, 1.3], [0.7, 1.3]]),
    "trapezoid": np.array([[0.0, 0.0], [2.0, 0.0], [1.2, 1.0], [0.8, 1.0]]),
    "sheared": np.array([[0.0, 0.0], [1.0, 0.0], [1.9, 1.0], [0.9, 1.0]]),
    "distorted": np.array([[0.0, 0.0], [2.0, 0.3], [1.4, 1.6], [0.15, 0.9]]),
}

PROJECTIONS = sorted(PROJECTION_COEFFICIENTS)
#: Projections that actually stabilise. `quad4` is the consistency check.
STABILISED = ["asmd", "asoi", "asoi_half"]
#: Tensorial projections: the added strain transforms as a tensor, so the
#: element is invariant under a rotation of the mesh. Measured, see
#: `TestInvariance`. `asoi` and `asoi_half` are deliberately not in this list.
TENSORIAL = ["quad4", "asmd"]
FRAME_DEPENDENT = ["asoi", "asoi_half"]
#: Parallelograms. On these, and only on these, the gamma-mode is a genuine
#: zero-energy mode of the central point.
PARALLELOGRAMS = ["unit_square", "rectangle", "parallelogram", "sheared"]


def _full_stiffness(nodes: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    total = np.zeros((8, 8))
    for (xi, eta), weight in zip(
        CPS4_QUADRATURE.points, CPS4_QUADRATURE.weights, strict=True
    ):
        operator, determinant = strain_displacement_matrix(nodes, xi, eta)
        total += weight * determinant * (operator.T @ tangent @ operator)
    return total


def _central_stiffness(nodes: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    operators = central_operators(nodes)
    centre = operators.strain_displacement_centre
    return operators.area * (centre.T @ tangent @ centre)


def _nodal(nodes: np.ndarray, field) -> np.ndarray:
    return np.array([component for node in nodes for component in field(*node)])


class TestCentralOperators:
    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    def test_the_closed_form_matches_the_shape_function_derivatives(
        self, name: str
    ) -> None:
        """Hallquist's `b` vectors are the derivatives at the centre, exactly."""

        nodes = GEOMETRIES[name]
        expected, _ = strain_displacement_matrix(nodes, 0.0, 0.0)

        np.testing.assert_allclose(
            central_operators(nodes).strain_displacement_centre, expected, atol=1e-14
        )

    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    def test_gamma_is_orthogonal_to_every_affine_field(self, name: str) -> None:
        """The property the patch test needs, and the reason `gamma` is not `h`."""

        nodes = GEOMETRIES[name]
        gamma = central_operators(nodes).gamma

        assert abs(float(gamma @ np.ones(4))) < 1e-14
        assert abs(float(gamma @ nodes[:, 0])) < 1e-14
        assert abs(float(gamma @ nodes[:, 1])) < 1e-14

    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    def test_the_area_matches_the_integrated_jacobian(self, name: str) -> None:
        operators = central_operators(GEOMETRIES[name])

        assert operators.area == pytest.approx(float(operators.jacobian_weights.sum()))

    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    def test_the_cross_terms_cancel(self, name: str) -> None:
        """`sum_g JAC(g) B_n(g) = 0`, on every geometry and not only a parallelogram.

        This is what makes the stabilisation orthogonal to the constant-strain
        part, and what lets `K_stab` inherit the symmetry of the tangent.
        """

        operators = central_operators(GEOMETRIES[name])
        for projection in PROJECTIONS:
            total = sum(
                operators.jacobian_weights[point]
                * enrichment_operator(operators, point, projection)
                for point in range(operators.jacobian_weights.size)
            )
            assert np.abs(total).max() < 1e-14, (name, projection)

    def test_a_degenerate_element_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-positive area"):
            central_operators([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])

    def test_a_reversed_element_is_refused(self) -> None:
        reversed_nodes = GEOMETRIES["unit_square"][::-1]

        with pytest.raises(ValueError, match="non-positive"):
            central_operators(reversed_nodes)

    def test_a_misshapen_input_is_refused(self) -> None:
        with pytest.raises(ValueError, match="four nodes"):
            central_operators(np.zeros((3, 2)))


class TestDecomposition:
    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    def test_bc_plus_bn_reproduces_the_bilinear_operator(self, name: str) -> None:
        """With the QUAD4 coefficients the split is a regrouping, not a model."""

        nodes = GEOMETRIES[name]
        operators = central_operators(nodes)
        centre = operators.strain_displacement_centre

        for point, (xi, eta) in enumerate(CPS4_QUADRATURE.points):
            expected, _ = strain_displacement_matrix(nodes, xi, eta)
            enriched = enrichment_operator(operators, point, "quad4")
            np.testing.assert_allclose(centre + enriched, expected, atol=1e-13)

    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    def test_the_quad4_projection_reproduces_the_fully_integrated_stiffness(
        self, name: str
    ) -> None:
        """The load-bearing consistency check of the whole construction."""

        nodes = GEOMETRIES[name]
        result = AssumedStrainStabilisation(projection="quad4").evaluate(
            central_operators(nodes), np.zeros(8), ELASTICITY
        )
        assembled = _central_stiffness(nodes, ELASTICITY) + result.tangent
        expected = _full_stiffness(nodes, ELASTICITY)

        assert np.abs(assembled - expected).max() / np.abs(expected).max() < 1e-13

    @pytest.mark.parametrize("projection", FRAME_DEPENDENT)
    def test_the_oi_projections_cancel_the_hourglass_shear(self, projection: str) -> None:
        """`e3 = 0` is the bending-locking cure -- and the frame dependence."""

        assert projection_coefficients(projection)[2] == 0.0
        operators = central_operators(GEOMETRIES["unit_square"])
        for point in range(4):
            assert np.abs(enrichment_operator(operators, point, projection)[2]).max() == 0.0

    def test_asmd_is_the_deviatoric_projection(self) -> None:
        """`(1/2, -1/2, 1)` is `dev(sym(q (x) grad h))`, which is why it is a tensor."""

        e1, e2, e3 = projection_coefficients("asmd")

        assert (e1, e2, e3) == (0.5, -0.5, 1.0)
        assert e1 + e2 == 0.0

    def test_the_poisson_dependent_projection_is_refused_by_name(self) -> None:
        """ASBQI presumes an isotropic transverse coupling; a crystal has none."""

        with pytest.raises(ValueError, match="Poisson"):
            projection_coefficients("asbqi")

    def test_an_unknown_projection_lists_the_available_ones(self) -> None:
        with pytest.raises(ValueError, match="available:"):
            projection_coefficients("nope")


class TestRigidAndAffineFields:
    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    @pytest.mark.parametrize("projection", STABILISED)
    def test_a_rigid_translation_produces_no_stabilisation(
        self, name: str, projection: str
    ) -> None:
        nodes = GEOMETRIES[name]
        motion = _nodal(nodes, lambda x, y: (3.7e-4, -1.1e-4))
        result = AssumedStrainStabilisation(projection=projection).evaluate(
            central_operators(nodes), motion, ELASTICITY
        )

        assert np.abs(result.internal_force).max() < 1e-12
        assert abs(result.energy) < 1e-18

    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    @pytest.mark.parametrize("projection", STABILISED)
    def test_an_infinitesimal_rotation_produces_no_stabilisation(
        self, name: str, projection: str
    ) -> None:
        nodes = GEOMETRIES[name]
        motion = _nodal(nodes, lambda x, y: (-1e-4 * y, 1e-4 * x))
        result = AssumedStrainStabilisation(projection=projection).evaluate(
            central_operators(nodes), motion, ELASTICITY
        )

        assert np.abs(result.internal_force).max() < 1e-12
        assert abs(result.energy) < 1e-18

    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    @pytest.mark.parametrize("projection", STABILISED)
    def test_an_affine_field_produces_no_stabilisation(
        self, name: str, projection: str
    ) -> None:
        """Section 12.2: the elastic patch test, in its element-level form.

        `gamma` is orthogonal to every affine field by construction, so the
        stabilisation cannot see one. The physical part alone then carries the
        whole response, which is exactly the patch test.
        """

        nodes = GEOMETRIES[name]
        motion = _nodal(
            nodes, lambda x, y: (1.3e-3 * x - 4.0e-4 * y, 7.0e-4 * x + 2.2e-3 * y)
        )
        result = AssumedStrainStabilisation(projection=projection).evaluate(
            central_operators(nodes), motion, ELASTICITY
        )

        assert np.abs(result.internal_force).max() < 1e-11
        assert abs(result.energy) < 1e-18
        assert np.abs(result.modal_amplitudes).max() < 1e-16

    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    def test_the_patch_test_is_exact_against_full_integration(self, name: str) -> None:
        """The complete element, not just its stabilisation."""

        nodes = GEOMETRIES[name]
        motion = _nodal(
            nodes, lambda x, y: (1.3e-3 * x - 4.0e-4 * y, 7.0e-4 * x + 2.2e-3 * y)
        )
        operators = central_operators(nodes)
        centre = operators.strain_displacement_centre
        result = AssumedStrainStabilisation().evaluate(operators, motion, ELASTICITY)
        physical = operators.area * (centre.T @ (ELASTICITY @ (centre @ motion)))
        expected = _full_stiffness(nodes, ELASTICITY) @ motion

        assert np.abs(physical + result.internal_force - expected).max() < 1e-9


class TestHourglassModes:
    @staticmethod
    def _modes(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gamma = central_operators(nodes).gamma
        mode_x = np.zeros(8)
        mode_y = np.zeros(8)
        mode_x[0::2] = gamma
        mode_y[1::2] = gamma
        return mode_x, mode_y

    @pytest.mark.parametrize("name", PARALLELOGRAMS)
    @pytest.mark.parametrize("projection", STABILISED)
    def test_on_a_parallelogram_the_modes_are_pure_zero_energy_modes(
        self, name: str, projection: str
    ) -> None:
        """`b . gamma = 0` holds only when the Jacobian is constant.

        On a general quadrilateral the gamma-mode carries genuine constant-strain
        content -- measured at `b_x . gamma = 0.28` on the trapezoid here -- so it
        is not a zero-energy mode there and this claim would be false. The
        stabilisation is still positive on it; see the next test.
        """

        nodes = GEOMETRIES[name]
        operators = central_operators(nodes)
        centre = operators.strain_displacement_centre

        for mode in self._modes(nodes):
            physical = operators.area * float(
                (centre @ mode) @ ELASTICITY @ (centre @ mode)
            )
            assert abs(physical) < 1e-18, (name, projection)

    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    @pytest.mark.parametrize("projection", STABILISED)
    def test_every_geometry_gives_the_modes_a_positive_stabilisation(
        self, name: str, projection: str
    ) -> None:
        nodes = GEOMETRIES[name]
        operators = central_operators(nodes)
        strategy = AssumedStrainStabilisation(projection=projection)

        for mode in self._modes(nodes):
            assert strategy.evaluate(operators, mode, ELASTICITY).energy > 0.0

    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    def test_the_gamma_mode_is_a_kernel_mode_only_on_a_parallelogram(
        self, name: str
    ) -> None:
        """Recorded because it is easy to assume otherwise."""

        operators = central_operators(GEOMETRIES[name])
        coupling = max(
            abs(float(operators.b_x @ operators.gamma)),
            abs(float(operators.b_y @ operators.gamma)),
        )

        if name in PARALLELOGRAMS:
            assert coupling < 1e-15
        else:
            assert coupling > 1e-3

    @pytest.mark.parametrize("name", sorted(GEOMETRIES))
    @pytest.mark.parametrize("projection", STABILISED)
    def test_the_element_has_exactly_the_three_rigid_null_modes(
        self, name: str, projection: str
    ) -> None:
        """Section 12.3: rank 5, no extra spurious mode."""

        nodes = GEOMETRIES[name]
        result = AssumedStrainStabilisation(projection=projection).evaluate(
            central_operators(nodes), np.zeros(8), ELASTICITY
        )
        stiffness = _central_stiffness(nodes, ELASTICITY) + result.tangent
        eigenvalues = np.linalg.eigvalsh(0.5 * (stiffness + stiffness.T))
        rank = int(np.count_nonzero(eigenvalues > 1e-9 * eigenvalues.max()))

        assert rank == 5, (name, projection, eigenvalues)

    def test_the_unstabilised_element_would_have_rank_three(self) -> None:
        """The defect the stabilisation exists to remove, shown to be real."""

        stiffness = _central_stiffness(GEOMETRIES["unit_square"], ELASTICITY)
        eigenvalues = np.linalg.eigvalsh(stiffness)

        assert int(np.count_nonzero(eigenvalues > 1e-9 * eigenvalues.max())) == 3

    def test_the_amplitudes_are_the_projection_of_the_displacement(self) -> None:
        nodes = GEOMETRIES["unit_square"]
        mode_x, _ = self._modes(nodes)
        operators = central_operators(nodes)

        amplitudes = hourglass_amplitudes(operators, mode_x)

        assert amplitudes[0] == pytest.approx(float(operators.gamma @ operators.gamma))
        assert abs(amplitudes[1]) < 1e-18


class TestInvariance:
    """Section 12.4."""

    @staticmethod
    def _energy(nodes: np.ndarray, motion: np.ndarray, projection: str) -> float:
        return AssumedStrainStabilisation(projection=projection).evaluate(
            central_operators(nodes), motion, ELASTICITY
        ).energy

    @pytest.mark.parametrize("projection", STABILISED)
    def test_translating_the_mesh_changes_nothing(self, projection: str) -> None:
        nodes = GEOMETRIES["distorted"]
        motion = np.arange(8, dtype=float) * 1e-4
        shifted = nodes + np.array([17.0, -3.0])

        assert self._energy(shifted, motion, projection) == pytest.approx(
            self._energy(nodes, motion, projection), rel=1e-12
        )

    @staticmethod
    def _rotated(nodes, motion, angle=0.7):
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        return nodes @ rotation.T, (motion.reshape(4, 2) @ rotation.T).reshape(-1)

    @pytest.mark.parametrize("projection", TENSORIAL)
    def test_a_tensorial_projection_is_invariant_under_a_mesh_rotation(
        self, projection: str
    ) -> None:
        """`quad4` is `sym(q (x) grad h)` and `asmd` its deviator; both are tensors."""

        nodes = GEOMETRIES["distorted"]
        motion = np.arange(8, dtype=float) * 1e-4
        rotated_nodes, rotated_motion = self._rotated(nodes, motion)

        assert self._energy(rotated_nodes, rotated_motion, projection) == pytest.approx(
            self._energy(nodes, motion, projection), rel=1e-10
        )

    @pytest.mark.parametrize("projection", FRAME_DEPENDENT)
    def test_the_oi_projections_are_not_invariant_under_a_mesh_rotation(
        self, projection: str
    ) -> None:
        """A property of the projection, not a defect -- but a real restriction.

        Zeroing the shear row is done in the element's own axes and is not a
        tensor operation. Measured here at about 38 percent of energy change for
        a 0.7 rad rotation of a distorted quadrilateral. On this project's
        axis-aligned pixel meshes every element shares one frame and the
        objection has no bite; on a general mesh these variants must not be used.
        """

        nodes = GEOMETRIES["distorted"]
        motion = np.arange(8, dtype=float) * 1e-4
        rotated_nodes, rotated_motion = self._rotated(nodes, motion)

        reference = self._energy(nodes, motion, projection)
        rotated = self._energy(rotated_nodes, rotated_motion, projection)

        assert abs(rotated - reference) / reference > 0.1

    @pytest.mark.parametrize("projection", STABILISED)
    def test_a_cyclic_node_permutation_changes_nothing(self, projection: str) -> None:
        nodes = GEOMETRIES["distorted"]
        motion = np.arange(8, dtype=float) * 1e-4
        order = [1, 2, 3, 0]
        permuted_nodes = nodes[order]
        permuted_motion = motion.reshape(4, 2)[order].reshape(-1)

        assert self._energy(permuted_nodes, permuted_motion, projection) == pytest.approx(
            self._energy(nodes, motion, projection), rel=1e-10
        )

    @pytest.mark.parametrize("projection", STABILISED)
    def test_scaling_the_geometry_scales_the_energy_as_the_area(
        self, projection: str
    ) -> None:
        """Strain is dimensionless, so at fixed strain the energy scales with area."""

        nodes = GEOMETRIES["unit_square"]
        factor = 3.0
        motion = np.arange(8, dtype=float) * 1e-4

        scaled = self._energy(factor * nodes, factor * motion, projection)
        assert scaled == pytest.approx(
            factor**2 * self._energy(nodes, motion, projection), rel=1e-10
        )


class TestEnergyProjection:
    """Section 6.2."""

    def test_a_positive_definite_tangent_is_left_alone(self) -> None:
        operators = central_operators(GEOMETRIES["unit_square"])
        motion = np.arange(8, dtype=float) * 1e-4
        direct = AssumedStrainStabilisation().evaluate(operators, motion, ELASTICITY)
        projected = EnergyProjectedAssumedStrainStabilisation().evaluate(
            operators, motion, ELASTICITY
        )

        scale = np.abs(direct.tangent).max()
        assert np.abs(projected.tangent - direct.tangent).max() < 1e-9 * scale
        assert projected.diagnostics["tangent_eigenvalues_lifted"] == 0.0

    def test_a_softened_tangent_is_lifted_to_the_floor(self) -> None:
        """The case the projection exists for: a tangent losing definiteness."""

        softened = ELASTICITY.copy()
        softened[2, 2] = -1.0
        operators = central_operators(GEOMETRIES["unit_square"])

        result = EnergyProjectedAssumedStrainStabilisation().evaluate(
            operators, np.arange(8, dtype=float) * 1e-4, softened
        )

        assert result.diagnostics["tangent_smallest_eigenvalue"] < 0.0
        assert result.diagnostics["tangent_eigenvalues_lifted"] >= 1.0
        assert result.energy > 0.0

    def test_the_projected_stabilisation_stays_positive_semi_definite(self) -> None:
        softened = ELASTICITY.copy()
        softened[2, 2] = -1.0
        operators = central_operators(GEOMETRIES["distorted"])

        tangent = EnergyProjectedAssumedStrainStabilisation().evaluate(
            operators, np.zeros(8), softened
        ).tangent
        eigenvalues = np.linalg.eigvalsh(0.5 * (tangent + tangent.T))

        assert eigenvalues.min() > -1e-9 * eigenvalues.max()

    def test_an_asymmetric_tangent_is_reported_and_symmetrised_only_here(self) -> None:
        """The direct variant keeps the asymmetry; only the energy variant drops it."""

        # The extension-shear coupling, not the normal-normal one: ASMD's first
        # two rows of `B_n` are exact opposites, so a perturbation of `C[0, 1]`
        # multiplies a symmetric product and leaves `K_stab` symmetric. A crystal
        # tangent is asymmetric in the shear coupling, which is the case that
        # matters.
        asymmetric = ELASTICITY.copy()
        asymmetric[0, 2] += 5_000.0
        operators = central_operators(GEOMETRIES["distorted"])

        direct = AssumedStrainStabilisation().evaluate(operators, np.zeros(8), asymmetric)
        projected = EnergyProjectedAssumedStrainStabilisation().evaluate(
            operators, np.zeros(8), asymmetric
        )

        scale = np.abs(direct.tangent).max()
        assert np.abs(direct.tangent - direct.tangent.T).max() > 1e-6 * scale
        assert np.abs(projected.tangent - projected.tangent.T).max() < 1e-9 * scale
        assert projected.diagnostics["tangent_relative_asymmetry"] > 0.0

    def test_a_tangent_with_no_positive_eigenvalue_is_refused(self) -> None:
        operators = central_operators(GEOMETRIES["unit_square"])

        with pytest.raises(ValueError, match="no positive"):
            EnergyProjectedAssumedStrainStabilisation().evaluate(
                operators, np.zeros(8), -ELASTICITY
            )

    @pytest.mark.parametrize("floor", [0.0, 1.0, -1e-6])
    def test_an_out_of_range_floor_is_refused(self, floor: float) -> None:
        with pytest.raises(ValueError, match="relative_floor"):
            EnergyProjectedAssumedStrainStabilisation(relative_floor=floor)

    def test_the_floor_is_relative_and_therefore_scale_free(self) -> None:
        """A conditioning rule cannot depend on the unit the tangent is in."""

        operators = central_operators(GEOMETRIES["unit_square"])
        strategy = EnergyProjectedAssumedStrainStabilisation()

        small = strategy.evaluate(operators, np.zeros(8), ELASTICITY)
        large = strategy.evaluate(operators, np.zeros(8), 1000.0 * ELASTICITY)

        assert large.diagnostics["tangent_floor"] == pytest.approx(
            1000.0 * small.diagnostics["tangent_floor"], rel=1e-12
        )


class TestRegistry:
    @pytest.mark.parametrize(
        "strategy", ["assumed_strain_current", "assumed_strain_energy"]
    )
    def test_a_registered_strategy_builds_and_names_itself(self, strategy: str) -> None:
        assert make_stabilisation(strategy).name == strategy

    def test_an_unknown_strategy_lists_the_available_ones(self) -> None:
        with pytest.raises(ValueError, match="available:"):
            make_stabilisation("nope")

    def test_an_unknown_projection_is_refused_at_construction(self) -> None:
        """Before any solve, as section 22 requires."""

        with pytest.raises(ValueError, match="projection"):
            make_stabilisation("assumed_strain_current", projection="nope")

    def test_the_result_carries_every_declared_diagnostic(self) -> None:
        result = make_stabilisation("assumed_strain_energy").evaluate(
            central_operators(GEOMETRIES["unit_square"]),
            np.arange(8, dtype=float) * 1e-4,
            ELASTICITY,
        )

        assert set(result.diagnostics) >= {
            "stabilisation_energy",
            "stabilisation_force_norm",
            "stabilisation_tangent_norm",
            "hourglass_amplitude",
            "tangent_floor",
        }
        assert result.modal_amplitudes.shape == (2,)
        assert result.internal_force.shape == (8,)
        assert result.tangent.shape == (8, 8)


class TestConsistentDerivative:
    """Section 8: the tangent must differentiate the force it is shipped with."""

    @pytest.mark.parametrize("projection", STABILISED)
    @pytest.mark.parametrize("name", ["unit_square", "distorted"])
    def test_the_stabilisation_tangent_matches_finite_differences(
        self, projection: str, name: str
    ) -> None:
        nodes = GEOMETRIES[name]
        operators = central_operators(nodes)
        strategy = AssumedStrainStabilisation(projection=projection)
        base = np.arange(8, dtype=float) * 1.0e-4
        analytical = strategy.evaluate(operators, base, ELASTICITY).tangent

        step = 1e-8
        numerical = np.zeros((8, 8))
        for column in range(8):
            forward, backward = base.copy(), base.copy()
            forward[column] += step
            backward[column] -= step
            numerical[:, column] = (
                strategy.evaluate(operators, forward, ELASTICITY).internal_force
                - strategy.evaluate(operators, backward, ELASTICITY).internal_force
            ) / (2.0 * step)

        assert np.abs(numerical - analytical).max() / np.abs(analytical).max() < 1e-6
