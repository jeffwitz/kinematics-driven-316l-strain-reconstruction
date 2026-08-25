"""Orientation handling, pinned against analytical rotations of a cubic tensor.

The reference here is never MGIS. A test that checks MGIS against MGIS proves
that the convention is self-consistent, not that it is right; these compare
against a stiffness rotated by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.core.crystal_orientation import (
    HomogeneousOrientationProvider,
    PixelOrientationProvider,
    mgis_rotation_argument,
    orientation_provider_from_mapping,
    rotation_from_euler_bunge_deg,
    validate_rotations,
)
from fem_inhouse.core.mesh import StructuredMesh, flatten_element_field_like_mesh

C11, C12, C44 = 197_000.0, 125_000.0, 122_000.0
ROOT_TWO = np.sqrt(2.0)


def _cubic_stiffness_kelvin() -> np.ndarray:
    """316L single-crystal stiffness in the crystal frame, Kelvin notation."""

    stiffness = np.zeros((6, 6))
    stiffness[:3, :3] = C12
    np.fill_diagonal(stiffness[:3, :3], C11)
    # Kelvin doubles the shear rows and columns, so C44 appears as 2 C44.
    np.fill_diagonal(stiffness[3:, 3:], 2.0 * C44)
    return stiffness


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
    """Stress from the convention of the module, computed by hand.

    eps_crystal = Q eps_global Q^T, then sigma_global = Q^T sigma_crystal Q.
    """

    rotation = rotation_global_to_material
    strain_crystal = _tensor_to_kelvin(
        rotation @ _kelvin_to_tensor(strain_kelvin_global) @ rotation.T
    )
    stress_crystal = _cubic_stiffness_kelvin() @ strain_crystal
    return _tensor_to_kelvin(rotation.T @ _kelvin_to_tensor(stress_crystal) @ rotation)


def _rotation_about_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_the_identity_is_accepted_and_returned_unchanged() -> None:
    validated = validate_rotations(np.eye(3)[None, :, :])

    assert validated.shape == (1, 3, 3)
    assert np.array_equal(validated[0], np.eye(3))


def test_a_wrong_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"shape \(n_points, 3, 3\)"):
        validate_rotations(np.eye(3))


def test_a_mismatched_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected 4 orientations, got 2"):
        validate_rotations(np.broadcast_to(np.eye(3), (2, 3, 3)), point_count=4)


def test_a_non_orthogonal_matrix_is_rejected() -> None:
    skewed = np.eye(3)[None, :, :].copy()
    skewed[0, 0, 1] = 0.1

    with pytest.raises(ValueError, match="not orthogonal"):
        validate_rotations(skewed)


def test_a_reflection_is_rejected_even_though_it_is_orthogonal() -> None:
    """det = -1 passes every orthogonality check and mirrors the crystal."""

    reflection = np.diag([1.0, 1.0, -1.0])[None, :, :]

    assert np.abs(reflection[0] @ reflection[0].T - np.eye(3)).max() == 0.0
    with pytest.raises(ValueError, match="reflection"):
        validate_rotations(reflection)


def test_a_non_finite_rotation_is_rejected() -> None:
    broken = np.eye(3)[None, :, :].copy()
    broken[0, 2, 2] = np.nan

    with pytest.raises(ValueError, match="finite"):
        validate_rotations(broken)


def test_validation_copies_rather_than_aliasing_the_caller_array() -> None:
    """The MGIS rotation calls mutate their argument in place."""

    source = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()

    validated = validate_rotations(source)
    validated[0, 0, 0] = 99.0

    assert source[0, 0, 0] == 1.0


# --------------------------------------------------------------------------
# The MGIS convention
# --------------------------------------------------------------------------


def test_the_mgis_argument_is_the_transpose_flattened() -> None:
    """MGIS wants material-to-global; this module speaks global-to-material.

    Measured, not assumed: passing the untransposed matrix reproduces the
    inverse rotation. This assertion is what keeps that discovery from being
    silently undone.
    """

    rotation = _rotation_about_z(30.0)

    flat = mgis_rotation_argument(rotation[None, :, :])

    assert flat.shape == (9,)
    assert np.allclose(flat.reshape(3, 3), rotation.T)


def test_the_mgis_argument_validates_before_converting() -> None:
    with pytest.raises(ValueError, match="not orthogonal"):
        mgis_rotation_argument(np.full((1, 3, 3), 0.5))


def test_several_points_are_laid_out_one_after_another() -> None:
    rotations = np.stack([np.eye(3), _rotation_about_z(90.0)])

    flat = mgis_rotation_argument(rotations)

    assert flat.shape == (18,)
    assert np.allclose(flat[:9].reshape(3, 3), np.eye(3))
    assert np.allclose(flat[9:].reshape(3, 3), _rotation_about_z(90.0).T)


# --------------------------------------------------------------------------
# The homogeneous provider
# --------------------------------------------------------------------------


def test_the_homogeneous_provider_repeats_one_matrix() -> None:
    provider = HomogeneousOrientationProvider(_rotation_about_z(30.0))

    rotations = provider.rotations_global_to_material(5)

    assert rotations.shape == (5, 3, 3)
    assert np.allclose(rotations, _rotation_about_z(30.0))


def test_the_provider_returns_a_writable_array_per_call() -> None:
    """A broadcast view would let one point's rotation change all of them."""

    provider = HomogeneousOrientationProvider.identity()

    rotations = provider.rotations_global_to_material(3)
    rotations[0, 0, 0] = 5.0

    assert provider.rotations_global_to_material(3)[0, 0, 0] == 1.0


def test_an_invalid_homogeneous_orientation_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="not orthogonal"):
        HomogeneousOrientationProvider(np.full((3, 3), 0.5))


def test_the_provider_rejects_a_nonsensical_point_count() -> None:
    provider = HomogeneousOrientationProvider.identity()

    with pytest.raises(ValueError, match="at least one"):
        provider.rotations_global_to_material(0)


# --------------------------------------------------------------------------
# Euler angles
# --------------------------------------------------------------------------


def test_zero_euler_angles_give_the_identity() -> None:
    assert np.allclose(rotation_from_euler_bunge_deg(0.0, 0.0, 0.0), np.eye(3))


def test_euler_angles_produce_proper_rotations() -> None:
    for angles in ((30.0, 45.0, 60.0), (10.0, 0.0, 350.0), (123.0, 47.0, 291.0)):
        validate_rotations(rotation_from_euler_bunge_deg(*angles)[None, :, :])


def test_the_first_bunge_angle_rotates_about_z() -> None:
    assert np.allclose(rotation_from_euler_bunge_deg(30.0, 0.0, 0.0), _rotation_about_z(30.0))


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_a_configuration_matrix_builds_a_provider() -> None:
    provider = orientation_provider_from_mapping(
        {"mode": "homogeneous", "matrix": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]}
    )

    assert np.allclose(
        provider.rotations_global_to_material(1)[0], _rotation_about_z(90.0)
    )


def test_a_configuration_may_use_euler_angles_instead() -> None:
    provider = orientation_provider_from_mapping(
        {"mode": "homogeneous", "euler_bunge_deg": [30.0, 0.0, 0.0]}
    )

    assert np.allclose(
        provider.rotations_global_to_material(1)[0], _rotation_about_z(30.0)
    )


def test_ebsd_configuration_expands_one_orientation_per_pixel_to_material_states() -> None:
    angles = np.zeros((2, 3, 3), dtype=float)
    angles[1, 2] = (90.0, 0.0, 0.0)
    provider = orientation_provider_from_mapping(
        {"mode": "ebsd", "euler_bunge_deg": angles}
    )
    rotations = provider.rotations_global_to_material(12)
    target = 2 * (1 * 3 + 2)
    assert rotations.shape == (12, 3, 3)
    np.testing.assert_allclose(rotations[0], np.eye(3))
    np.testing.assert_allclose(rotations[target], rotation_from_euler_bunge_deg(90, 0, 0))
    np.testing.assert_allclose(rotations[target + 1], rotations[target])


def test_rectangular_ebsd_f_order_matches_structured_mesh_element_ids() -> None:
    mesh = StructuredMesh(3.0, 5.0, 1.0, 1.0)
    markers = 1000 * np.indices((3, 5))[0] + np.indices((3, 5))[1]
    angles = np.zeros((3, 5, 3), dtype=float)
    angles[..., 0] = markers * 0.01
    provider = orientation_provider_from_mapping(
        {"mode": "ebsd", "euler_bunge_deg": angles, "element_order": "F"}
    )
    rotations = provider.rotations_global_to_material(mesh.n_elems)
    source = np.empty((3, 5, 3, 3), dtype=float)
    for i, j in np.ndindex((3, 5)):
        source[i, j] = rotation_from_euler_bunge_deg(angles[i, j, 0], 0.0, 0.0)
    expected = flatten_element_field_like_mesh(source, mesh.elem_ids)
    np.testing.assert_allclose(rotations, expected)


def test_rectangular_ebsd_default_c_order_is_distinguishable_from_mesh_f_order() -> None:
    mesh = StructuredMesh(3.0, 5.0, 1.0, 1.0)
    markers = 1000 * np.indices((3, 5))[0] + np.indices((3, 5))[1]
    angles = np.zeros((3, 5, 3), dtype=float)
    angles[..., 0] = markers * 0.01
    provider = PixelOrientationProvider.from_euler_bunge_deg(angles)
    rotations = provider.rotations_global_to_material(mesh.n_elems)
    expected = flatten_element_field_like_mesh(
        np.asarray(
            [
                [rotation_from_euler_bunge_deg(angles[i, j, 0], 0.0, 0.0) for j in range(5)]
                for i in range(3)
            ]
        ), mesh.elem_ids
    )
    assert not np.allclose(rotations, expected)


def test_a_configuration_must_choose_one_form() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        orientation_provider_from_mapping(
            {"mode": "homogeneous", "matrix": np.eye(3), "euler_bunge_deg": [0, 0, 0]}
        )
    with pytest.raises(ValueError, match="exactly one"):
        orientation_provider_from_mapping({"mode": "homogeneous"})


def test_an_unknown_mode_names_the_supported_ones() -> None:
    with pytest.raises(ValueError, match="homogeneous, ebsd"):
        orientation_provider_from_mapping({"mode": "unsupported"})


# --------------------------------------------------------------------------
# The convention itself, against an analytical rotation
# --------------------------------------------------------------------------


def test_a_cubic_symmetry_rotation_leaves_the_response_alone() -> None:
    """Ninety degrees about a cube axis maps the crystal onto itself."""

    strain = np.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    unrotated = analytical_global_stress(strain, np.eye(3))
    rotated = analytical_global_stress(strain, _rotation_about_z(90.0))

    assert np.allclose(rotated, unrotated, atol=1e-9)


def test_a_generic_rotation_changes_the_response() -> None:
    """Otherwise the whole orientation machinery would be doing nothing."""

    strain = np.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    unrotated = analytical_global_stress(strain, np.eye(3))
    rotated = analytical_global_stress(strain, _rotation_about_z(30.0))

    assert not np.allclose(rotated, unrotated, atol=1e-6)
    # A cubic crystal loaded off-axis develops shear from an axial strain.
    assert abs(rotated[3]) > 1e-2


def test_opposite_rotations_mirror_each_other() -> None:
    """A cube has a mirror plane normal to x, so +30 and -30 are mirror images."""

    strain = np.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    plus = analytical_global_stress(strain, _rotation_about_z(30.0))
    minus = analytical_global_stress(strain, _rotation_about_z(-30.0))

    assert plus[:3] == pytest.approx(minus[:3], abs=1e-9)
    assert plus[3] == pytest.approx(-minus[3], abs=1e-9)


def test_the_zener_ratio_is_the_one_the_stiffness_encodes() -> None:
    assert pytest.approx(3.3889, rel=1e-4) == 2.0 * C44 / (C11 - C12)
