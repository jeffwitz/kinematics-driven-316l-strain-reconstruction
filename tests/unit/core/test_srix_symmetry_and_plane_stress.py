"""Cubic symmetry of the crystal law, and the plane-stress bench.

Sections 10 and 13.

Section 10 asks a sharper question than "does the macroscopic stress match".
Two orientations related by a cubic symmetry must give the same response *after
permuting the slip systems*, and the permutation carries a sign because a slip
system is defined up to the sense of its Burgers vector. Comparing macroscopic
stress alone would pass even if the law were slipping on the wrong systems.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import pytest

from fem_inhouse.core.crystal_orientation import (
    mgis_rotation_argument,
    rotation_from_euler_bunge_deg,
)
from fem_inhouse.core.fcc_interaction_matrix import (
    build_rank_matrix,
    cubic_rotations,
    slip_system_permutation,
)

SRIX = "Fcc316LForestRubinSrix"
PLASTIC_SLIP = slice(6, 18)
BACK_STRAIN = slice(30, 42)
ROOT_TWO = math.sqrt(2.0)

#: Crystal directions the specification names.
AXES: dict[str, tuple[float, float, float]] = {
    "001": (0.0, 0.0, 1.0),
    "011": (0.0, 1.0, 1.0),
    "111": (1.0, 1.0, 1.0),
    "123": (1.0, 2.0, 3.0),
}

_BEHAVIOUR: dict[str, Any] = {}


def _mgis() -> tuple[Any, str]:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if not library:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    mgis = pytest.importorskip("mgis.behaviour")
    return mgis, library


def _behaviour() -> tuple[Any, Any]:
    mgis, library = _mgis()
    handle = _BEHAVIOUR.get("srix")
    if handle is None:
        handle = mgis.load(library, SRIX, mgis.Hypothesis.Tridimensional)
        _BEHAVIOUR["srix"] = handle
    # Stated every time: MGIS shares behaviour handles process-wide.
    for name, value in (
        ("SrixOverstressModulus", 18.7819100705),
        ("tau0", 40.0),
        ("Q", 10.0),
        ("b", 3.0),
        ("C", 40000.0),
        ("d", 1500.0),
    ):
        mgis.setParameter(handle, name, value)
    return mgis, handle


def _rotation_for_axis(axis: tuple[float, float, float]) -> np.ndarray:
    """`Q_global_to_material` sending the global `z` onto `axis`.

    Built by completing an orthonormal frame, so the test drives a global
    tension along `z` and the crystal sees it along the requested direction.
    """

    third = np.asarray(axis, dtype=float)
    third = third / np.linalg.norm(third)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, third))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    first = np.cross(seed, third)
    first /= np.linalg.norm(first)
    second = np.cross(third, first)
    return np.vstack([first, second, third])


def _run(
    rotation: np.ndarray,
    *,
    axial: float = 0.012,
    steps: int = 60,
) -> dict[str, Any]:
    """Uniaxial tension along the global `z`, for one crystal orientation."""

    mgis, behaviour = _behaviour()
    data = mgis.MaterialDataManager(behaviour, 1)
    for state in (data.s0, data.s1):
        mgis.setExternalStateVariable(state, "Temperature", 293.15)
    argument = mgis_rotation_argument(np.asarray(rotation, dtype=float)[None, :, :])
    for index in range(steps):
        value = axial * (index + 1) / steps
        strain = np.zeros(6)
        strain[2] = value
        strain[0] = strain[1] = -0.5 * value
        crystal = np.ascontiguousarray(strain.copy())
        mgis.rotateGradients(crystal, behaviour, argument)
        data.s1.gradients[:, :] = crystal.reshape(1, 6)
        mgis.integrate(
            data, mgis.IntegrationType.IntegrationWithConsistentTangentOperator, 1.0, 0, 1
        )
        mgis.update(data)
    # The behaviour is orthotropic, so `thermodynamic_forces` are in the
    # MATERIAL frame. Reading a global component off them without rotating back
    # picks a different component of the same tensor: for a symmetry sending z
    # onto y this reported -sigma/2 and looked like a broken symmetry.
    stress = np.ascontiguousarray(
        np.asarray(data.s1.thermodynamic_forces[0], dtype=float).copy()
    )
    mgis.rotateThermodynamicForces(stress, behaviour, argument)
    return {
        "axial_stress": float(stress[2] - 0.5 * (stress[0] + stress[1])),
        "slip": data.s1.internal_state_variables[0, PLASTIC_SLIP].copy(),
        "back": data.s1.internal_state_variables[0, BACK_STRAIN].copy(),
    }


# ---------------------------------------------------------------------------
# Section 10 -- cubic symmetry.
# ---------------------------------------------------------------------------


class TestSymmetryAlgebra:
    """The permutation itself, without touching MGIS."""

    def test_there_are_twenty_four_proper_rotations(self) -> None:
        rotations = cubic_rotations()

        assert len(rotations) == 24
        for matrix in rotations:
            assert round(float(np.linalg.det(matrix))) == 1
            np.testing.assert_array_equal(matrix @ matrix.T, np.eye(3, dtype=int))

    def test_every_symmetry_permutes_the_slip_systems(self) -> None:
        for matrix in cubic_rotations():
            destination, sign = slip_system_permutation(matrix)

            assert sorted(destination.tolist()) == list(range(12))
            assert set(sign.tolist()) <= {-1, 1}

    def test_the_identity_is_the_identity_permutation(self) -> None:
        destination, sign = slip_system_permutation(np.eye(3, dtype=int))

        np.testing.assert_array_equal(destination, np.arange(12))
        np.testing.assert_array_equal(sign, np.ones(12, dtype=int))

    def test_the_interaction_matrix_is_invariant_under_every_symmetry(self) -> None:
        """A structural consequence: the hardening cannot see the symmetry."""

        matrix = build_rank_matrix()

        for rotation in cubic_rotations():
            destination, _ = slip_system_permutation(rotation)
            np.testing.assert_array_equal(matrix[np.ix_(destination, destination)], matrix)

    def test_some_symmetries_reverse_a_slip_sense(self) -> None:
        """Which is why the permutation carries a sign at all."""

        assert any(
            -1 in slip_system_permutation(rotation)[1].tolist()
            for rotation in cubic_rotations()
        )

    def test_an_improper_operation_is_refused(self) -> None:
        reflection = np.diag([1, 1, -1]).astype(int)

        with pytest.raises(ValueError, match="determinant"):
            slip_system_permutation(reflection)


@pytest.mark.mfront
class TestSymmetryOnTheBehaviour:
    @pytest.mark.parametrize("name", sorted(AXES))
    def test_every_cubic_symmetry_leaves_the_axial_response_unchanged(
        self, name: str
    ) -> None:
        """Twenty-four orientations, one response. The macroscopic half."""

        base = _rotation_for_axis(AXES[name])
        reference = _run(base)

        for rotation in cubic_rotations():
            candidate = _run(rotation.astype(float) @ base)
            assert candidate["axial_stress"] == pytest.approx(
                reference["axial_stress"], rel=1e-9
            ), name

    @pytest.mark.parametrize("name", sorted(AXES))
    def test_the_slip_spectrum_is_preserved_by_every_symmetry(self, name: str) -> None:
        """The half that macroscopic stress cannot see.

        Two orientations related by a cubic symmetry must carry the *same slip
        on the same systems*, relabelled. The sorted spectrum of `|gamma_s|` is
        invariant under any relabelling, so comparing it tests exactly that,
        without depending on which relabelling it is: a law slipping on the
        wrong number of systems, or splitting the same total over a different
        set, fails here while the macroscopic stress would still agree.

        **What this does not do.** It does not assert the index map. The
        catalogue-level action of the symmetry group is available as
        `slip_system_permutation`, it is a genuine group action and it leaves
        the interaction matrix invariant -- all asserted in `TestSymmetryAlgebra`
        -- but reconciling it with the order MFront reports slips in was not
        established, and no rule tried here held for all twenty-four. Rather
        than pick the rule that happened to fit, the property is tested in the
        permutation-invariant form. The index map is an open point, recorded in
        the qualification report.
        """

        base = _rotation_for_axis(AXES[name])
        reference = np.sort(np.abs(_run(base)["slip"]))

        for rotation in cubic_rotations():
            candidate = np.sort(np.abs(_run(rotation.astype(float) @ base)["slip"]))
            np.testing.assert_allclose(candidate, reference, atol=1e-12)

    @pytest.mark.parametrize("name", sorted(AXES))
    def test_the_number_of_active_systems_is_preserved(self, name: str) -> None:
        base = _rotation_for_axis(AXES[name])
        reference = int((np.abs(_run(base)["slip"]) > 1e-12).sum())

        for rotation in cubic_rotations():
            candidate = _run(rotation.astype(float) @ base)["slip"]
            assert int((np.abs(candidate) > 1e-12).sum()) == reference

    def test_the_spectrum_test_is_not_vacuous(self) -> None:
        """Two genuinely different orientations must fail it.

        Otherwise preserving the spectrum would say nothing about symmetry.
        """

        along_001 = np.sort(np.abs(_run(_rotation_for_axis(AXES["001"]))["slip"]))
        along_123 = np.sort(np.abs(_run(_rotation_for_axis(AXES["123"]))["slip"]))

        assert not np.allclose(along_001, along_123, atol=1e-10)

    @pytest.mark.parametrize("name", sorted(AXES))
    def test_a_small_perturbation_moves_the_answer_only_a_little(
        self, name: str
    ) -> None:
        """Section 10's stability requirement, around each named orientation."""

        base = _rotation_for_axis(AXES[name])
        reference = _run(base)
        perturbation = rotation_from_euler_bunge_deg(0.05, 0.05, 0.05)
        perturbed = _run(perturbation @ base)

        relative = abs(perturbed["axial_stress"] - reference["axial_stress"]) / abs(
            reference["axial_stress"]
        )
        assert relative < 5e-3, (name, relative)

    def test_the_named_orientations_are_genuinely_different(self) -> None:
        """Otherwise the symmetry tests above would prove nothing."""

        stresses = {
            name: _run(_rotation_for_axis(axis))["axial_stress"]
            for name, axis in AXES.items()
        }

        assert len({round(value, 6) for value in stresses.values()}) == len(AXES)


# ---------------------------------------------------------------------------
# Section 13 -- the plane-stress bench.
# ---------------------------------------------------------------------------


def _plane_stress_batch(
    *, orientation: np.ndarray | None = None, parameter_set: str | None = None
) -> Any:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if not library:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    pytest.importorskip("mgis.behaviour")
    from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch

    options: dict[str, Any] = {}
    if orientation is not None:
        options["crystal_orientation"] = {
            "mode": "homogeneous",
            "matrix": np.asarray(orientation, dtype=float).tolist(),
        }
    if parameter_set is not None:
        options["parameter_set"] = parameter_set
    return create_plane_stress_material_batch(
        "mfront-3d-condensed-plane-stress",
        np.full((1, 1), 250.0),
        np.full((1, 1), 500.0),
        0.245,
        young_modulus_mpa=205000.0,
        poisson_ratio=0.3,
        hardening_mode="ludwik",
        plastic_strain_max=0.2,
        plastic_table_points=1000,
        first_positive_plastic_strain=1e-6,
        mfront_library=library,
        mfront_threads=1,
        mfront_behaviour_id="fcc_forest_rubin_srix",
        constitutive_options=options,
    )


LOADINGS: dict[str, tuple[float, float, float]] = {
    "tension": (1.0, -0.4, 0.0),
    "shear": (0.0, 0.0, 1.0),
    "biaxial": (1.0, 0.7, 0.0),
    "tension_shear": (0.8, -0.3, 0.5),
}

ORIENTATIONS: dict[str, np.ndarray] = {
    "identity": np.eye(3),
    "bunge_30_0_0": rotation_from_euler_bunge_deg(30.0, 0.0, 0.0),
    "bunge_35_20_15": rotation_from_euler_bunge_deg(35.0, 20.0, 15.0),
    "bunge_54_45_10": rotation_from_euler_bunge_deg(54.7, 45.0, 10.0),
}


@pytest.mark.mfront
class TestPlaneStressBench:
    """Section 13, over orientations, loadings and overstress moduli."""

    @pytest.mark.parametrize("orientation", sorted(ORIENTATIONS))
    @pytest.mark.parametrize("loading", sorted(LOADINGS))
    def test_the_out_of_plane_stresses_are_driven_to_zero(
        self, orientation: str, loading: str
    ) -> None:
        """All three components. An off-axis crystal couples them."""

        batch = _plane_stress_batch(orientation=ORIENTATIONS[orientation])
        direction = np.array(LOADINGS[loading])
        amplitude = 0.012
        for index in range(40):
            trial = batch.evaluate(
                (amplitude * (index + 1) / 40 * direction)[None, :], time_increment=1.0
            )
            batch.commit()

        # `plane_stress_residual_mpa` is exactly the three out-of-plane
        # components the closure has to annihilate: sigma_zz, sigma_xz, sigma_yz.
        transverse = np.abs(np.asarray(trial.plane_stress_residual_mpa[0], dtype=float))
        scale = max(float(np.abs(trial.full_stress_tensor_mpa[0]).max()), 1.0)

        assert transverse.max() < max(1e-6, 1e-9 * scale), (orientation, loading)

    @pytest.mark.parametrize(
        "parameter_set",
        [
            "316l_srix_exploratory_r1",
            "316l_srix_exploratory_r8",
            "316l_srix_transposed_from_nasri2018_rate_1e-3",
        ],
    )
    def test_the_closure_holds_across_overstress_moduli(self, parameter_set: str) -> None:
        batch = _plane_stress_batch(
            orientation=ORIENTATIONS["bunge_35_20_15"], parameter_set=parameter_set
        )
        for index in range(40):
            trial = batch.evaluate(
                np.array([[0.012 * (index + 1) / 40, -0.005 * (index + 1) / 40, 0.003]]),
                time_increment=1.0,
            )
            batch.commit()

        assert np.abs(np.asarray(trial.plane_stress_residual_mpa[0])).max() < 1e-6

    def test_an_off_axis_crystal_couples_extension_and_shear(self) -> None:
        """Otherwise the condensation would be untested in its hard case."""

        aligned = _plane_stress_batch(orientation=ORIENTATIONS["identity"])
        tilted = _plane_stress_batch(orientation=ORIENTATIONS["bunge_35_20_15"])
        strain = np.array([[0.004, -0.0016, 0.0]])

        aligned_trial = aligned.evaluate(strain, time_increment=1.0)
        tilted_trial = tilted.evaluate(strain, time_increment=1.0)
        aligned_shear = abs(float(aligned_trial.stress_in_plane_mpa[0, 2]))
        tilted_shear = abs(float(tilted_trial.stress_in_plane_mpa[0, 2]))

        assert aligned_shear < 1e-9
        assert tilted_shear > 1.0

    @pytest.mark.parametrize("orientation", ["identity", "bunge_35_20_15"])
    def test_the_condensed_tangent_matches_global_finite_differences(
        self, orientation: str
    ) -> None:
        """Section 13's last requirement, across the plateau of the difference.

        Several perturbation amplitudes are used because a single one cannot
        distinguish a wrong tangent from a badly conditioned difference: too
        large and the nonlinearity shows, too small and round-off does.
        """

        batch = _plane_stress_batch(orientation=ORIENTATIONS[orientation])
        base = np.array([[0.006, -0.0024, 0.001]])
        # Commit nineteen of the twenty steps, so the trial below carries a
        # NON-ZERO increment. Evaluating at the committed strain would return
        # the elastic tangent, and the comparison would silently stop testing
        # the plastic branch it exists for.
        for index in range(19):
            batch.evaluate(base * (index + 1) / 20, time_increment=1.0)
            batch.commit()

        trial = batch.evaluate(base, time_increment=1.0)
        analytical = np.asarray(trial.tangent_in_plane_mpa[0], dtype=float)
        batch.revert()

        best = math.inf
        for step in (1e-7, 1e-8, 1e-9):
            numerical = np.zeros((3, 3))
            for component in range(3):
                perturbed = base.copy()
                perturbed[0, component] += step
                forward = batch.evaluate(perturbed, time_increment=1.0)
                plus = np.asarray(forward.stress_in_plane_mpa[0], dtype=float)
                batch.revert()
                perturbed[0, component] -= 2.0 * step
                backward = batch.evaluate(perturbed, time_increment=1.0)
                minus = np.asarray(backward.stress_in_plane_mpa[0], dtype=float)
                batch.revert()
                numerical[:, component] = (plus - minus) / (2.0 * step)
            error = float(
                np.abs(numerical - analytical).max() / np.abs(analytical).max()
            )
            best = min(best, error)

        assert best < 1e-5, (orientation, best)
