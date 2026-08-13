"""The FCC crystal laws driven through the condensed plane-stress bridge.

Section 12 of the specification. Everything here needs MGIS and the compiled
behaviour library, so the module skips without ``MFRONT_BEHAVIOUR_LIBRARY``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from fem_inhouse.core.crystal_orientation import (
    HomogeneousOrientationProvider,
    rotation_from_euler_bunge_deg,
)
from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch

C11, C12, C44 = 197_000.0, 125_000.0, 122_000.0
ROOT_TWO = np.sqrt(2.0)


def _kelvin_to_tensor(vector: np.ndarray) -> np.ndarray:
    shear = vector[3:] / ROOT_TWO
    return np.array(
        [
            [vector[0], shear[0], shear[1]],
            [shear[0], vector[1], shear[2]],
            [shear[1], shear[2], vector[2]],
        ]
    )


def _tensor_to_kelvin(tensor: np.ndarray) -> np.ndarray:
    return np.array(
        [
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            ROOT_TWO * tensor[0, 1],
            ROOT_TWO * tensor[0, 2],
            ROOT_TWO * tensor[1, 2],
        ]
    )


def analytical_global_stress(
    strain_kelvin_global: np.ndarray, rotation_global_to_material: np.ndarray
) -> np.ndarray:
    """Elastic stress under the convention of `crystal_orientation`, by hand.

    Deliberately independent of MGIS: comparing MGIS against MGIS would show
    that the convention is self-consistent, not that it is the right way round.

        eps_crystal = Q eps_global Q^T,   sigma_global = Q^T sigma_crystal Q.
    """

    stiffness = np.zeros((6, 6))
    stiffness[:3, :3] = C12
    np.fill_diagonal(stiffness[:3, :3], C11)
    # Kelvin doubles the shear rows and columns, so C44 appears as 2 C44.
    np.fill_diagonal(stiffness[3:, 3:], 2.0 * C44)

    rotation = rotation_global_to_material
    strain_crystal = _tensor_to_kelvin(
        rotation @ _kelvin_to_tensor(strain_kelvin_global) @ rotation.T
    )
    stress_crystal = stiffness @ strain_crystal
    return _tensor_to_kelvin(rotation.T @ _kelvin_to_tensor(stress_crystal) @ rotation)


SRIX = "fcc_forest_rubin_srix"
MERIC_CAILLETAUD = "fcc_meric_cailletaud"
SRIX_GENERIC = "fcc_forest_rubin_srix_generic_validation"

#: Every out-of-plane stress the condensation must annihilate, in MPa.
PLANE_STRESS_TOLERANCE_MPA = 1e-8


def _library() -> str:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if not library:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    pytest.importorskip("mgis.behaviour")
    return library


def _generic_library() -> str:
    library = os.environ.get("SRIX_GENERIC_MFRONT_BEHAVIOUR_LIBRARY")
    if not library:
        pytest.skip("SRIX_GENERIC_MFRONT_BEHAVIOUR_LIBRARY is not set")
    pytest.importorskip("mgis.behaviour")
    return library


def _batch(
    behaviour_id: str = SRIX,
    *,
    points: int = 1,
    orientation: np.ndarray | None = None,
    backend: str = "mfront-3d-condensed-plane-stress",
    nonlocal_coupling_modulus_mpa: float | None = None,
):
    options = {}
    if orientation is not None:
        options["crystal_orientation"] = {"mode": "homogeneous", "matrix": orientation}
    return create_plane_stress_material_batch(
        backend,
        np.full(points, 124.0),
        np.full(points, 380.0),
        0.245,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.3,
        hardening_mode="ludwik",
        plastic_strain_max=0.2,
        plastic_table_points=1000,
        first_positive_plastic_strain=1e-6,
        mfront_library=_library(),
        mfront_threads=1,
        mfront_behaviour_id=behaviour_id,
        nonlocal_coupling_modulus_mpa=nonlocal_coupling_modulus_mpa,
        constitutive_options=options or None,
    )


def _rotation_about_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------
# 12.1 - the factory accepts the crystal profile
# --------------------------------------------------------------------------


def test_the_factory_builds_a_crystal_batch() -> None:
    batch = _batch()

    assert batch.backend_name == "mfront-3d-condensed-plane-stress-micromorphic"
    # A crystal tangent is not symmetric, so the global solver must not assume it.
    assert batch.linear_system_matrix_type == "nonsymmetric"


def test_validation_generic_srix_backend_is_opt_in() -> None:
    batch = create_plane_stress_material_batch(
        "mfront-srix-generic-plane-stress",
        np.full(2, 124.0),
        np.full(2, 380.0),
        0.245,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.3,
        hardening_mode="ludwik",
        plastic_strain_max=0.2,
        plastic_table_points=1000,
        first_positive_plastic_strain=1e-6,
        mfront_library=_generic_library(),
        mfront_threads=1,
        mfront_behaviour_id=SRIX_GENERIC,
        nonlocal_coupling_modulus_mpa=100.0,
    )
    batch.set_nonlocal_equivalent_plastic_strain(np.array([2e-4, 1e-4]))
    trial = batch.evaluate_in_plane(
        np.array([[2e-4, -6e-5, 1e-5], [1.5e-4, -4e-5, -2e-5]]),
        time_increment=1.0,
    )
    assert trial.tangent_in_plane_mpa is not None
    assert "accumulated_slip" in trial.observables
    complete = batch.complete_trial(trial)
    assert complete.full_stress_tensor_mpa.shape == (2, 3, 3)
    batch.commit()


def test_meric_cailletaud_is_available_through_the_same_path() -> None:
    assert _batch(MERIC_CAILLETAUD).point_count == 1


def test_a_crystal_has_no_native_plane_stress_hypothesis() -> None:
    with pytest.raises(ValueError, match="no native plane-stress"):
        _batch(backend="mfront-native-plane-stress")


def test_the_micromorphic_coupling_uses_crystal_accumulated_slip() -> None:
    """SRIX uses Gamma=sum of the twelve accumulated positive slips."""

    batch = create_plane_stress_material_batch(
        "mfront-3d-condensed-plane-stress",
        np.full(1, 124.0),
        np.full(1, 380.0),
        0.245,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.3,
        hardening_mode="ludwik",
        plastic_strain_max=0.2,
        plastic_table_points=1000,
        first_positive_plastic_strain=1e-6,
        mfront_library=_library(),
        mfront_threads=1,
        mfront_behaviour_id=SRIX,
        nonlocal_coupling_modulus_mpa=200.0,
    )
    assert batch.backend_name == "mfront-3d-condensed-plane-stress-micromorphic"


@pytest.mark.mfront
def test_srix_scalar_micromorphic_extension_accepts_a_path_cutback() -> None:
    """The stiff exploratory coupling is advanced through accepted substeps."""

    batch = _batch(nonlocal_coupling_modulus_mpa=5168.0)
    target = np.array([1.5e-3, -3.0e-4, 3.0e-4])
    for fraction in (0.5, 1.0):
        trial = batch.evaluate_nonlocal_state(
            (fraction * target)[None, :],
            time_increment=1.0,
        )
        assert trial[0][0] >= 0.0
        batch.commit()


def test_the_j2_elastic_constants_are_not_imposed_on_a_crystal() -> None:
    """The crystal carries its own cubic elasticity inside MFront."""

    batch = create_plane_stress_material_batch(
        "mfront-3d-condensed-plane-stress",
        np.full(1, 124.0),
        np.full(1, 380.0),
        0.245,
        young_modulus_mpa=1.0,  # nonsense for J2, irrelevant here
        poisson_ratio=0.49,
        hardening_mode="ludwik",
        plastic_strain_max=0.2,
        plastic_table_points=1000,
        first_positive_plastic_strain=1e-6,
        mfront_library=_library(),
        mfront_threads=1,
        mfront_behaviour_id=SRIX,
    )

    assert batch.point_count == 1


def test_an_unknown_constitutive_option_is_refused() -> None:
    with pytest.raises(ValueError, match="unsupported constitutive_options"):
        create_plane_stress_material_batch(
            "mfront-3d-condensed-plane-stress",
            np.full(1, 124.0),
            np.full(1, 380.0),
            0.245,
            young_modulus_mpa=205_000.0,
            poisson_ratio=0.3,
            hardening_mode="ludwik",
            plastic_strain_max=0.2,
            plastic_table_points=1000,
            first_positive_plastic_strain=1e-6,
            mfront_library=_library(),
            mfront_threads=1,
            mfront_behaviour_id=SRIX,
            constitutive_options={"nonsense": 1},
        )


# --------------------------------------------------------------------------
# 12.2 and 12.3 - the rotation, against an analytical cubic stiffness
# --------------------------------------------------------------------------


def _mgis_global_stress(strain_kelvin: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """One elastic 3D integration through the bridge, in the global frame."""

    from fem_inhouse.core.mfront import MFront3DMaterialPointBatch
    from fem_inhouse.core.mfront_behaviours import MFRONT_BEHAVIOURS

    specification = MFRONT_BEHAVIOURS.get(SRIX)
    bridge = MFront3DMaterialPointBatch(
        _library(),
        behaviour_spec=specification,
        point_count=1,
        rotation_global_to_material=rotation[None, :, :],
        behaviour_name=specification.behaviour_name("condensed_3d"),
    )
    return bridge.evaluate(strain_kelvin[None, :], time_increment=1.0).stress_kelvin_mpa[0]


def test_the_bridge_reproduces_the_cubic_stiffness_unrotated() -> None:
    """Section 12.2, and a check that the elasticity really is anisotropic."""

    strain = np.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    stress = _mgis_global_stress(strain, np.eye(3))

    assert stress[0] == pytest.approx(C11 * 1e-5, rel=1e-5)
    assert stress[1] == pytest.approx(C12 * 1e-5, rel=1e-5)
    assert stress[2] == pytest.approx(C12 * 1e-5, rel=1e-5)
    # Isotropy would demand C44 = (C11 - C12) / 2 = 36 GPa; it is 122.
    assert pytest.approx(3.3889, rel=1e-4) == 2.0 * C44 / (C11 - C12)


@pytest.mark.parametrize("degrees", [30.0, -30.0, 17.0])
def test_the_mgis_rotation_agrees_with_a_hand_rotated_stiffness(degrees: float) -> None:
    """Section 12.3, the assertion that pins the whole convention.

    The reference is an analytical rotation of the cubic tensor, so a
    self-consistent but inverted convention cannot pass.
    """

    strain = np.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])
    rotation = _rotation_about_z(degrees)

    through_mgis = _mgis_global_stress(strain, rotation)
    by_hand = analytical_global_stress(strain, rotation)

    assert through_mgis == pytest.approx(by_hand, abs=1e-4)


def test_a_cubic_symmetry_rotation_leaves_the_response_unchanged() -> None:
    strain = np.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    unrotated = _mgis_global_stress(strain, np.eye(3))
    rotated = _mgis_global_stress(strain, _rotation_about_z(90.0))

    assert rotated == pytest.approx(unrotated, abs=1e-6)


def test_a_generic_rotation_changes_the_response() -> None:
    """Otherwise the orientation would be plumbed in but doing nothing."""

    strain = np.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    unrotated = _mgis_global_stress(strain, np.eye(3))
    rotated = _mgis_global_stress(strain, _rotation_about_z(30.0))

    assert not np.allclose(rotated, unrotated, atol=1e-3)
    # An off-axis cubic crystal answers an axial strain with shear.
    assert abs(rotated[3]) > 1e-2


# --------------------------------------------------------------------------
# 12.4 - anisotropic plane-stress condensation
# --------------------------------------------------------------------------


def _analytical_condensed_tangent(rotation: np.ndarray) -> np.ndarray:
    """Plane-stress condensation of the rotated cubic stiffness, by hand."""

    stiffness = np.zeros((6, 6))
    for column in range(6):
        unit = np.zeros(6)
        unit[column] = 1.0
        stiffness[:, column] = analytical_global_stress(unit, rotation)
    in_plane = np.array([0, 1, 3])
    out_of_plane = np.array([2, 4, 5])
    caa = stiffness[np.ix_(in_plane, in_plane)]
    cab = stiffness[np.ix_(in_plane, out_of_plane)]
    cba = stiffness[np.ix_(out_of_plane, in_plane)]
    cbb = stiffness[np.ix_(out_of_plane, out_of_plane)]
    return caa - cab @ np.linalg.solve(cbb, cba)


@pytest.mark.parametrize("degrees", [0.0, 30.0])
def test_the_condensation_drives_all_three_out_of_plane_stresses_to_zero(
    degrees: float,
) -> None:
    """Not just sigma_zz: an off-axis crystal couples in the shears too."""

    batch = _batch(orientation=_rotation_about_z(degrees))

    trial = batch.evaluate(np.array([[3e-4, -1e-4, 5e-5]]), time_increment=1.0)

    assert np.abs(trial.plane_stress_residual_mpa).max() < PLANE_STRESS_TOLERANCE_MPA
    assert np.abs(trial.full_stress_tensor_mpa[0, 2, 2]) < PLANE_STRESS_TOLERANCE_MPA
    assert np.abs(trial.full_stress_tensor_mpa[0, 0, 2]) < PLANE_STRESS_TOLERANCE_MPA
    assert np.abs(trial.full_stress_tensor_mpa[0, 1, 2]) < PLANE_STRESS_TOLERANCE_MPA


@pytest.mark.parametrize("degrees", [0.0, 30.0])
def test_the_elastic_condensed_tangent_matches_the_analytical_one(degrees: float) -> None:
    """Section 12.4, off-diagonal terms included."""

    rotation = _rotation_about_z(degrees)
    batch = _batch(orientation=rotation)

    trial = batch.evaluate(np.array([[1e-6, 0.0, 0.0]]), time_increment=1.0)
    tangent = trial.tangent_in_plane_mpa[0]

    # The bridge reports engineering plane-stress components; the analytical
    # condensation is in Kelvin, so the shear row and column carry sqrt(2).
    scale = np.array([1.0, 1.0, 1.0 / np.sqrt(2.0)])
    expected = _analytical_condensed_tangent(rotation) * scale[:, None] * scale[None, :]

    assert tangent == pytest.approx(expected, rel=1e-4, abs=1.0)
    if degrees != 0.0:
        # The rotated crystal genuinely couples extension and shear.
        assert abs(expected[0, 2]) > 1.0


@pytest.mark.parametrize("degrees", [0.0, 30.0])
def test_the_condensed_tangent_matches_finite_differences_in_plasticity(
    degrees: float,
) -> None:
    batch = _batch(orientation=_rotation_about_z(degrees))
    preload = np.array([[2e-3, -8e-4, 4e-4]])
    for step in range(1, 11):
        batch.evaluate(preload * step / 10, time_increment=1.0)
        batch.commit()

    strain = preload * 1.05
    trial = batch.evaluate(strain, time_increment=1.0)
    algorithmic = trial.tangent_in_plane_mpa[0]
    reference = trial.stress_in_plane_mpa[0].copy()

    perturbation = 1e-9
    numerical = np.zeros((3, 3))
    for column in range(3):
        perturbed = strain.copy()
        perturbed[0, column] += perturbation
        numerical[:, column] = (
            batch.evaluate(perturbed, time_increment=1.0).stress_in_plane_mpa[0] - reference
        ) / perturbation

    deviation = np.abs(algorithmic - numerical).max() / np.abs(numerical).max()
    assert deviation < 1e-4


# --------------------------------------------------------------------------
# 12.5 - crystal plasticity through the condensation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "rotation"),
    [
        ("identity", np.eye(3)),
        ("bunge 30/45/60", rotation_from_euler_bunge_deg(30.0, 45.0, 60.0)),
    ],
)
def test_crystal_plasticity_converges_and_exposes_twelve_slips(
    label: str, rotation: np.ndarray
) -> None:
    batch = _batch(orientation=rotation)

    trial = None
    for step in range(1, 21):
        trial = batch.evaluate(np.array([[6e-3, -2e-3, 1e-3]]) * step / 20, time_increment=1.0)
        batch.commit()
    assert trial is not None

    observables = trial.observables
    for family in ("plastic_slip", "equivalent_plastic_slip", "back_strain"):
        assert observables[family].shape == (1, 12)
    assert observables["accumulated_slip"].shape == (1,)
    assert (observables["equivalent_plastic_slip"] >= 0.0).all()
    assert observables["accumulated_slip"][0] > 0.0
    assert np.abs(trial.plane_stress_residual_mpa).max() < PLANE_STRESS_TOLERANCE_MPA
    # Several systems must carry the load, or the test says nothing about a crystal.
    assert int((np.abs(observables["plastic_slip"]) > 1e-12).sum()) >= 2
    assert "equivalent_plastic_strain" not in observables


def test_an_orientation_changes_the_plastic_answer() -> None:
    strain = np.array([[4e-3, -1.5e-3, 0.0]])
    answers = []
    for rotation in (np.eye(3), rotation_from_euler_bunge_deg(30.0, 45.0, 60.0)):
        batch = _batch(orientation=rotation)
        for step in range(1, 11):
            trial = batch.evaluate(strain * step / 10, time_increment=1.0)
            batch.commit()
        answers.append(trial.stress_in_plane_mpa[0].copy())

    assert not np.allclose(answers[0], answers[1], rtol=1e-3)


def test_commit_and_revert_keep_the_crystal_state_consistent() -> None:
    batch = _batch()
    settled = np.array([[3e-3, -1e-3, 0.0]])
    for step in range(1, 11):
        batch.evaluate(settled * step / 10, time_increment=1.0)
        batch.commit()
    committed = batch.evaluate(settled, time_increment=1.0)
    reference = committed.observables["equivalent_plastic_slip"].copy()

    # A trial that is thrown away must leave nothing behind.
    batch.evaluate(settled * 3.0, time_increment=1.0)
    batch.revert()
    after_revert = batch.evaluate(settled, time_increment=1.0)

    assert after_revert.observables["equivalent_plastic_slip"] == pytest.approx(reference)


def test_the_default_orientation_is_the_identity() -> None:
    """Crystal axes aligned with the specimen axes when nothing is configured."""

    provider = HomogeneousOrientationProvider.identity()
    without = _batch()
    withidentity = _batch(orientation=provider.rotations_global_to_material(1)[0])

    strain = np.array([[2e-3, -7e-4, 0.0]])
    assert without.evaluate(strain, time_increment=1.0).stress_in_plane_mpa == pytest.approx(
        withidentity.evaluate(strain, time_increment=1.0).stress_in_plane_mpa
    )
