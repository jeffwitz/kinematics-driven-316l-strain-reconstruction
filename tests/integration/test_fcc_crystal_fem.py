"""A small plane-stress finite-element run driven by the SRIX crystal law.

Section 12.6. Deliberately tiny so it can run in CI: no experimental ROI is
needed to establish that the crystal law goes through the whole solver.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from fem_inhouse.core.crystal_orientation import rotation_from_euler_bunge_deg
from fem_inhouse.examples import reduced_biaxial_case
from fem_inhouse.solver import run_case_study

SRIX = "fcc_forest_rubin_srix"
SRIX_GENERIC = "fcc_forest_rubin_srix_generic_validation"


def _library() -> str:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    pytest.importorskip("mgis.behaviour")
    return library


def _generic_library() -> str:
    library = os.environ.get("SRIX_GENERIC_MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("SRIX_GENERIC_MFRONT_BEHAVIOUR_LIBRARY is not set")
    pytest.importorskip("mgis.behaviour")
    return library


def _run(
    orientation: np.ndarray | None = None,
    *,
    nx: int = 3,
    ny: int = 3,
    element_formulation: str = "cps4",
):
    case = reduced_biaxial_case(nx=nx, ny=ny)
    options: dict[str, object] = {}
    if orientation is not None:
        options["crystal_orientation"] = {
            "mode": "homogeneous",
            "matrix": np.asarray(orientation).tolist(),
        }
    solver = replace(
        case.config.solver,
        constitutive_backend="mfront-3d-condensed-plane-stress",
        mfront_library=_library(),
        mfront_behaviour_id=SRIX,
        constitutive_options=options,
        mfront_threads=1,
        increments=8,
        element_formulation=element_formulation,
    )
    return run_case_study(
        replace(case.config, solver=solver),
        displacement_x_mm=case.displacement_x_mm,
        displacement_y_mm=case.displacement_y_mm,
        yield_stress_mpa=case.yield_stress_mpa,
        hardening_coefficient_mpa=case.hardening_coefficient_mpa,
    )


@pytest.mark.mfront
def test_a_small_crystal_case_assembles_and_converges() -> None:
    result = _run()

    assert result.diagnostics is not None
    assert result.diagnostics.cutbacks == 0
    assert np.isfinite(result.displacement_mm).all()
    assert np.isfinite(result.stress_mpa).all()
    assert np.abs(result.displacement_mm).max() > 0.0


@pytest.mark.mfront
def test_generic_srix_matches_historical_srix_on_homogeneous_case() -> None:
    case = reduced_biaxial_case(nx=3, ny=3)
    common = dict(increments=8, residual_tolerance=1e-6, mfront_threads=1)
    historical = replace(
        case.config.solver,
        constitutive_backend="mfront-3d-condensed-plane-stress",
        mfront_library=_library(),
        mfront_behaviour_id=SRIX,
        **common,
    )
    generic = replace(
        case.config.solver,
        constitutive_backend="mfront-srix-generic-plane-stress",
        mfront_library=_generic_library(),
        mfront_behaviour_id=SRIX_GENERIC,
        **common,
    )
    results = [
        run_case_study(
            replace(case.config, solver=solver),
            displacement_x_mm=case.displacement_x_mm,
            displacement_y_mm=case.displacement_y_mm,
            yield_stress_mpa=case.yield_stress_mpa,
            hardening_coefficient_mpa=case.hardening_coefficient_mpa,
        )
        for solver in (historical, generic)
    ]
    historical_result, generic_result = results
    np.testing.assert_allclose(
        generic_result.displacement_mm, historical_result.displacement_mm, rtol=1e-10, atol=1e-14
    )
    np.testing.assert_allclose(
        generic_result.stress_mpa, historical_result.stress_mpa, rtol=1e-10, atol=1e-8
    )
    np.testing.assert_allclose(
        generic_result.equivalent_plastic_strain,
        historical_result.equivalent_plastic_strain,
        rtol=1e-10,
        atol=1e-14,
    )
    assert generic_result.diagnostics is not None
    assert generic_result.diagnostics.maximum_gauss_point_plane_stress_residual_mpa < 1e-6


@pytest.mark.mfront
def test_srix_scalar_nonlocal_source_runs_through_the_solver() -> None:
    """The generic nested driver accepts SRIX accumulated slip as its source."""

    case = reduced_biaxial_case(nx=3, ny=3)
    solver = replace(
        case.config.solver,
        constitutive_backend="mfront-3d-condensed-plane-stress",
        mfront_library=_library(),
        mfront_behaviour_id=SRIX,
        increments=4,
        max_newton_iterations=20,
        residual_tolerance=1e-6,
        minimum_step_divisor=32,
        mfront_threads=1,
    )
    nonlocal_config = replace(
        case.config.nonlocal_plasticity,
        enabled=True,
        length_scale_mm=0.05888,
        coupling_modulus_mpa=100.0,
        criterion="accumulated_slip_helmholtz",
        relative_tolerance=1e-6,
        maximum_iterations=15,
    )
    result = run_case_study(
        replace(case.config, solver=solver, nonlocal_plasticity=nonlocal_config),
        displacement_x_mm=case.displacement_x_mm,
        displacement_y_mm=case.displacement_y_mm,
        yield_stress_mpa=case.yield_stress_mpa,
        hardening_coefficient_mpa=case.hardening_coefficient_mpa,
    )

    assert result.cumulated_slip is not None
    assert result.nonlocal_equivalent_plastic_strain is not None
    assert np.all(result.cumulated_slip >= 0.0)
    assert np.all(result.nonlocal_equivalent_plastic_strain >= 0.0)
    assert result.diagnostics.converged_increments == 4
    assert result.diagnostics.cutbacks == 0
    assert result.diagnostics.nonlocal_coupling_failures == 0
    assert result.diagnostics.total_nonlocal_iterations > 0
    assert result.diagnostics.maximum_gauss_point_plane_stress_residual_mpa < 1e-6


@pytest.mark.mfront
def test_generic_srix_scalar_nonlocal_source_runs_through_the_solver() -> None:
    case = reduced_biaxial_case(nx=3, ny=3)
    solver = replace(
        case.config.solver,
        constitutive_backend="mfront-srix-generic-plane-stress",
        mfront_library=_generic_library(),
        mfront_behaviour_id=SRIX_GENERIC,
        increments=4,
        max_newton_iterations=20,
        residual_tolerance=1e-6,
        minimum_step_divisor=32,
        mfront_threads=1,
    )
    nonlocal_config = replace(
        case.config.nonlocal_plasticity,
        enabled=True,
        length_scale_mm=0.05888,
        coupling_modulus_mpa=100.0,
        criterion="accumulated_slip_helmholtz",
        relative_tolerance=1e-6,
        maximum_iterations=15,
    )
    result = run_case_study(
        replace(case.config, solver=solver, nonlocal_plasticity=nonlocal_config),
        displacement_x_mm=case.displacement_x_mm,
        displacement_y_mm=case.displacement_y_mm,
        yield_stress_mpa=case.yield_stress_mpa,
        hardening_coefficient_mpa=case.hardening_coefficient_mpa,
    )
    assert result.cumulated_slip is not None
    assert result.nonlocal_equivalent_plastic_strain is not None
    assert result.diagnostics.converged_increments == 4
    assert result.diagnostics.nonlocal_coupling_failures == 0
    assert result.diagnostics.total_nonlocal_iterations > 0
    assert result.diagnostics.maximum_gauss_point_plane_stress_residual_mpa < 1e-6


@pytest.mark.mfront
def test_srix_scalar_nonlocal_source_accepts_a_spatial_orientation_map() -> None:
    """The scalar coupling survives the first heterogeneous EBSD-like map."""

    case = reduced_biaxial_case(nx=3, ny=3)
    angles = np.zeros((3, 3, 3), dtype=float)
    angles[1:, :, :] = np.array([30.0, 45.0, 60.0])
    options = {
        "crystal_orientation": {
            "mode": "ebsd",
            "euler_bunge_deg": angles.tolist(),
        }
    }
    solver = replace(
        case.config.solver,
        constitutive_backend="mfront-3d-condensed-plane-stress",
        mfront_library=_library(),
        mfront_behaviour_id=SRIX,
        constitutive_options=options,
        increments=4,
        max_newton_iterations=20,
        residual_tolerance=1e-6,
        minimum_step_divisor=32,
        mfront_threads=1,
    )
    nonlocal_config = replace(
        case.config.nonlocal_plasticity,
        enabled=True,
        length_scale_mm=0.05888,
        coupling_modulus_mpa=100.0,
        criterion="accumulated_slip_helmholtz",
        relative_tolerance=1e-6,
        maximum_iterations=15,
    )
    result = run_case_study(
        replace(case.config, solver=solver, nonlocal_plasticity=nonlocal_config),
        displacement_x_mm=case.displacement_x_mm,
        displacement_y_mm=case.displacement_y_mm,
        yield_stress_mpa=case.yield_stress_mpa,
        hardening_coefficient_mpa=case.hardening_coefficient_mpa,
    )

    assert result.cumulated_slip is not None
    assert result.nonlocal_equivalent_plastic_strain is not None
    assert np.isfinite(result.stress_mpa).all()
    assert result.diagnostics.converged_increments >= 4
    assert result.diagnostics.cutbacks == 3
    assert result.diagnostics.nonlocal_coupling_failures == result.diagnostics.cutbacks
    assert result.diagnostics.maximum_gauss_point_plane_stress_residual_mpa < 1e-6
    assert not np.allclose(result.stress_mpa[0, 0], result.stress_mpa[2, 0])


@pytest.mark.mfront
def test_generic_srix_scalar_nonlocal_source_accepts_a_spatial_orientation_map() -> None:
    case = reduced_biaxial_case(nx=3, ny=3)
    angles = np.zeros((3, 3, 3), dtype=float)
    angles[1:, :, :] = np.array([30.0, 45.0, 60.0])
    options = {
        "crystal_orientation": {
            "mode": "ebsd",
            "euler_bunge_deg": angles.tolist(),
        }
    }
    solver = replace(
        case.config.solver,
        constitutive_backend="mfront-srix-generic-plane-stress",
        mfront_library=_generic_library(),
        mfront_behaviour_id=SRIX_GENERIC,
        constitutive_options=options,
        increments=4,
        max_newton_iterations=20,
        residual_tolerance=1e-6,
        minimum_step_divisor=32,
        mfront_threads=1,
    )
    nonlocal_config = replace(
        case.config.nonlocal_plasticity,
        enabled=True,
        length_scale_mm=0.05888,
        coupling_modulus_mpa=100.0,
        criterion="accumulated_slip_helmholtz",
        relative_tolerance=1e-6,
        maximum_iterations=15,
    )
    result = run_case_study(
        replace(case.config, solver=solver, nonlocal_plasticity=nonlocal_config),
        displacement_x_mm=case.displacement_x_mm,
        displacement_y_mm=case.displacement_y_mm,
        yield_stress_mpa=case.yield_stress_mpa,
        hardening_coefficient_mpa=case.hardening_coefficient_mpa,
    )
    assert result.cumulated_slip is not None
    assert result.nonlocal_equivalent_plastic_strain is not None
    assert np.isfinite(result.stress_mpa).all()
    assert result.diagnostics.converged_increments >= 4
    assert result.diagnostics.nonlocal_coupling_failures == result.diagnostics.cutbacks
    assert result.diagnostics.maximum_gauss_point_plane_stress_residual_mpa < 1e-6
    assert not np.allclose(result.stress_mpa[0, 0], result.stress_mpa[2, 0])


@pytest.mark.mfront
def test_generic_srix_realises_the_same_heterogeneous_nonlocal_solution_as_legacy() -> None:
    """The validation bridge must preserve the legacy heterogeneous solution."""

    case = reduced_biaxial_case(nx=3, ny=3)
    angles = np.zeros((3, 3, 3), dtype=float)
    angles[0, :, :] = np.array([17.0, 31.0, 43.0])
    angles[1, :, :] = np.array([35.0, 20.0, 15.0])
    angles[2, :, :] = np.array([62.0, 11.0, 78.0])
    options = {
        "crystal_orientation": {
            "mode": "ebsd",
            "euler_bunge_deg": angles.tolist(),
        }
    }
    nonlocal_config = replace(
        case.config.nonlocal_plasticity,
        enabled=True,
        length_scale_mm=0.05888,
        coupling_modulus_mpa=100.0,
        criterion="accumulated_slip_helmholtz",
        relative_tolerance=1e-6,
        maximum_iterations=15,
    )
    solvers = [
        replace(
            case.config.solver,
            constitutive_backend=backend,
            mfront_library=library,
            mfront_behaviour_id=behaviour,
            constitutive_options=options,
            increments=4,
            max_newton_iterations=20,
            residual_tolerance=1e-6,
            minimum_step_divisor=32,
            mfront_threads=1,
        )
        for backend, library, behaviour in (
            ("mfront-3d-condensed-plane-stress", _library(), SRIX),
            ("mfront-srix-generic-plane-stress", _generic_library(), SRIX_GENERIC),
        )
    ]
    results = [
        run_case_study(
            replace(case.config, solver=solver, nonlocal_plasticity=nonlocal_config),
            displacement_x_mm=case.displacement_x_mm,
            displacement_y_mm=case.displacement_y_mm,
            yield_stress_mpa=case.yield_stress_mpa,
            hardening_coefficient_mpa=case.hardening_coefficient_mpa,
        )
        for solver in solvers
    ]
    legacy, generic = results
    assert legacy.diagnostics.converged_increments == 4
    assert generic.diagnostics.converged_increments == 4
    assert generic.diagnostics.maximum_gauss_point_plane_stress_residual_mpa < 1e-6
    np.testing.assert_allclose(generic.displacement_mm, legacy.displacement_mm, rtol=1e-5)
    np.testing.assert_allclose(generic.stress_mpa, legacy.stress_mpa, rtol=2e-5, atol=1e-8)
    np.testing.assert_allclose(generic.cumulated_slip, legacy.cumulated_slip, rtol=1e-5)
    np.testing.assert_allclose(
        generic.nonlocal_equivalent_plastic_strain,
        legacy.nonlocal_equivalent_plastic_strain,
        rtol=1e-5,
    )


@pytest.mark.mfront
@pytest.mark.parametrize(
    ("backend", "behaviour_id", "library_factory"),
    [
        ("mfront-3d-condensed-plane-stress", SRIX, _library),
        ("mfront-srix-generic-plane-stress", SRIX_GENERIC, _generic_library),
    ],
)
def test_the_requested_coupling_modulus_actually_reaches_the_law(
    backend: str, behaviour_id: str, library_factory: Callable[[], str]
) -> None:
    """Asking for a non-local coupling must change the solution.

    The equivalence test above compares two backends against each other, which
    says nothing about whether either of them is coupled at all. It passed for
    a year of commits while BOTH sides ran with Hchi = 0: the crystal factory
    put the requested modulus in `material_property_values`, and the condensed
    batch overwrote it with zeros on its way to MGIS. The request was accepted,
    recorded in the manifest, and discarded.

    An agreement test cannot catch that; only a sensitivity test can. This one
    asserts the one thing a silently inert coupling cannot fake -- that turning
    the modulus on moves the answer.
    """

    case = reduced_biaxial_case(nx=3, ny=3)
    angles = np.zeros((3, 3, 3), dtype=float)
    angles[0, :, :] = np.array([17.0, 31.0, 43.0])
    angles[1, :, :] = np.array([35.0, 20.0, 15.0])
    angles[2, :, :] = np.array([62.0, 11.0, 78.0])
    options = {
        "crystal_orientation": {"mode": "ebsd", "euler_bunge_deg": angles.tolist()}
    }
    solver = replace(
        case.config.solver,
        constitutive_backend=backend,
        mfront_library=library_factory(),
        mfront_behaviour_id=behaviour_id,
        constitutive_options=options,
        increments=4,
        max_newton_iterations=20,
        residual_tolerance=1e-6,
        minimum_step_divisor=32,
        mfront_threads=1,
    )

    def solve(coupling_modulus_mpa: float) -> np.ndarray:
        nonlocal_config = replace(
            case.config.nonlocal_plasticity,
            enabled=True,
            length_scale_mm=0.05888,
            coupling_modulus_mpa=coupling_modulus_mpa,
            criterion="accumulated_slip_helmholtz",
            relative_tolerance=1e-6,
            maximum_iterations=15,
        )
        return run_case_study(
            replace(case.config, solver=solver, nonlocal_plasticity=nonlocal_config),
            displacement_x_mm=case.displacement_x_mm,
            displacement_y_mm=case.displacement_y_mm,
            yield_stress_mpa=case.yield_stress_mpa,
            hardening_coefficient_mpa=case.hardening_coefficient_mpa,
        ).displacement_mm

    uncoupled = solve(0.0)
    coupled = solve(100.0)
    relative = float(np.abs(coupled - uncoupled).max() / np.abs(uncoupled).max())
    # Measured at 5.4e-3 on this case for both backends. The bound is loose on
    # purpose: the point is that the effect exists and is far above solver
    # noise, not that it has a particular value.
    assert relative > 1e-4, f"the coupling modulus left the solution unchanged ({relative:.3e})"


@pytest.mark.mfront
def test_the_out_of_plane_stresses_are_driven_to_zero() -> None:
    """The plane-stress condition, imposed in the global frame.

    All three components, not just sigma_zz: an off-axis crystal couples the
    normal and the out-of-plane shears.
    """

    result = _run(rotation_from_euler_bunge_deg(30.0, 45.0, 60.0))

    assert np.abs(result.plane_stress_residual_mpa).max() < 1e-6
    # The vector form carries sigma_zz, sigma_xz and sigma_yz separately.
    assert np.abs(result.plane_stress_residual_vector_mpa).max() < 1e-6


@pytest.mark.mfront
def test_the_out_of_plane_strain_is_reconstructed() -> None:
    """A crystal contracts out of plane by its own anisotropy, not by nu."""

    result = _run()

    transverse = result.total_strain_tensor[..., 2, 2]
    assert np.isfinite(transverse).all()
    assert np.abs(transverse).max() > 0.0


@pytest.mark.mfront
def test_no_j2_equivalent_plastic_strain_is_manufactured() -> None:
    """The crystal exposes twelve slips; the solver must not invent a scalar.

    The FEMResult field stays at zero rather than being filled with a plausible
    substitute. The twelve-component families are reachable from the material
    batch, which test_fcc_plane_stress.py exercises directly; plumbing them
    through FEMResult is a separate change and is not claimed here.
    """

    result = _run()

    assert np.abs(result.equivalent_plastic_strain).max() == 0.0


@pytest.mark.mfront
def test_an_orientation_changes_the_computed_stress() -> None:
    """The orientation must reach the constitutive evaluation, and be visible.

    Not on the displacement: this case prescribes an affine field on all four
    edges, and an affine field satisfies equilibrium exactly for ANY homogeneous
    material, so the displacement is orientation independent by construction.
    The stress it produces is not.
    """

    aligned = _run()
    tilted = _run(rotation_from_euler_bunge_deg(30.0, 45.0, 60.0))

    assert not np.allclose(aligned.stress_mpa, tilted.stress_mpa, rtol=1e-6)
    np.testing.assert_allclose(
        tilted.displacement_mm, aligned.displacement_mm, rtol=1e-8, atol=1e-14
    )


@pytest.mark.mfront
def test_a_cubic_symmetry_rotation_leaves_the_field_alone() -> None:
    """Ninety degrees about a cube axis maps the crystal onto itself."""

    aligned = _run()
    quarter_turn = _run(np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))

    np.testing.assert_allclose(
        quarter_turn.displacement_mm, aligned.displacement_mm, rtol=1e-6, atol=1e-12
    )


# --------------------------------------------------------------------------
# Section 15.9 - the crystal law under reduced integration
# --------------------------------------------------------------------------

TILTED = rotation_from_euler_bunge_deg(30.0, 45.0, 60.0)


@pytest.mark.mfront
@pytest.mark.parametrize(("label", "orientation"), [("identity", None), ("tilted", TILTED)])
def test_the_crystal_runs_with_one_material_point_per_element(
    label: str, orientation: np.ndarray | None
) -> None:
    """One constitutive state per element, and four times fewer of them.

    This is the whole reason the reduced element exists: a crystal point costs
    roughly sixteen times a J2 point, so removing three of the four is worth
    more here than anywhere else in the solver.
    """

    full = _run(orientation)
    reduced = _run(orientation, element_formulation="cps4r")

    assert full.diagnostics.gauss_points_per_element == 4
    assert reduced.diagnostics.gauss_points_per_element == 1
    assert (
        full.diagnostics.constitutive_material_point_count
        == 4 * reduced.diagnostics.constitutive_material_point_count
    )
    assert reduced.diagnostics.cutbacks == 0


@pytest.mark.mfront
@pytest.mark.parametrize(("label", "orientation"), [("identity", None), ("tilted", TILTED)])
def test_the_reduced_crystal_agrees_with_the_full_one_on_an_affine_load(
    label: str, orientation: np.ndarray | None
) -> None:
    """Every Gauss point sees the same strain, so the two must coincide.

    Including for a tilted crystal, which is the case that would expose a
    stabilisation built on the wrong elasticity.
    """

    full = _run(orientation)
    reduced = _run(orientation, element_formulation="cps4r")

    np.testing.assert_allclose(reduced.displacement_mm, full.displacement_mm, rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(reduced.stress_mpa, full.stress_mpa, rtol=1e-7, atol=1e-7)
    assert np.abs(reduced.plane_stress_residual_mpa).max() < 1e-6


@pytest.mark.mfront
@pytest.mark.parametrize(("label", "orientation"), [("identity", None), ("tilted", TILTED)])
def test_the_reduced_crystal_excites_no_hourglass_energy_on_an_affine_load(
    label: str, orientation: np.ndarray | None
) -> None:
    reduced = _run(orientation, element_formulation="cps4r")

    assert reduced.diagnostics.hourglass_energy_ratio < 1e-9


@pytest.mark.mfront
def test_the_stabilisation_uses_the_rotated_cubic_elasticity() -> None:
    """Section 11, asserted rather than trusted.

    The reference tangent is measured from the behaviour itself, so it carries
    the orientation. Were it the isotropic matrix instead, the two orientations
    would return the same operator.
    """

    from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch

    def reference(orientation: np.ndarray) -> np.ndarray:
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
            constitutive_options={
                "crystal_orientation": {
                    "mode": "homogeneous",
                    "matrix": orientation.tolist(),
                }
            },
        )
        return batch.reference_in_plane_tangent_mpa()

    aligned = reference(np.eye(3))
    tilted = reference(TILTED)

    # Cubic in plane stress, crystal axes aligned: the shear modulus is C44.
    assert aligned[2, 2] == pytest.approx(122_000.0, rel=1e-6)
    assert aligned[0, 0] == pytest.approx(197_000.0 - 125_000.0**2 / 197_000.0, rel=1e-6)
    assert aligned[0, 2] == pytest.approx(0.0, abs=1e-6)
    # Tilted: extension and shear couple, which an isotropic reference cannot do.
    assert abs(tilted[0, 2]) > 1e3
    assert np.abs(tilted - tilted.T).max() < 1e-9 * np.abs(tilted).max()


@pytest.mark.mfront
def test_the_crystal_state_reaches_the_typed_result() -> None:
    """Section 14. Twelve slips cannot be flattened into a scalar PEEQ."""

    result = _run(nx=3, ny=3)

    assert result.plastic_slip is not None
    assert result.plastic_slip.shape == (3, 3, 12)
    assert result.equivalent_plastic_slip.shape == (3, 3, 12)
    assert result.back_strain.shape == (3, 3, 12)
    assert result.cumulated_slip.shape == (3, 3)
    assert result.active_slip_systems.shape == (3, 3)


@pytest.mark.mfront
def test_the_equivalent_plastic_strain_is_never_filled_with_cumulated_slip() -> None:
    """Section 14 and prohibition 16.

    The sum of twelve accumulated slips is a different scalar with a different
    definition and a different magnitude. Reporting it as a J2 equivalent
    plastic strain would make two incomparable campaigns look comparable, so the
    field stays at exactly zero and the slip travels under its own name.
    """

    result = _run(nx=3, ny=3)

    assert float(np.abs(result.equivalent_plastic_strain).max()) == 0.0
    assert float(result.cumulated_slip.max()) > 1e-3


@pytest.mark.mfront
def test_the_accumulated_slips_are_non_negative_and_sum_to_the_scalar() -> None:
    result = _run(nx=3, ny=3)

    assert float(result.equivalent_plastic_slip.min()) >= 0.0
    np.testing.assert_allclose(
        result.equivalent_plastic_slip.sum(axis=2), result.cumulated_slip, rtol=1e-12
    )


@pytest.mark.mfront
def test_the_active_count_matches_the_nonzero_slips() -> None:
    result = _run(nx=3, ny=3)

    expected = np.count_nonzero(np.abs(result.plastic_slip) > 1e-12, axis=2)
    np.testing.assert_array_equal(result.active_slip_systems.astype(int), expected)


@pytest.mark.mfront
def test_a_crystal_campaign_declares_the_slip_fields_for_archiving(tmp_path) -> None:
    """Section 14: archivable, not merely present in memory."""

    from dataclasses import replace as _replace

    from fem_inhouse.partitioning import PartitionLayout
    from fem_inhouse.workflows.partitioned import PartitionWorkflow

    case = reduced_biaxial_case(nx=4, ny=4)
    solver = _replace(
        case.config.solver,
        constitutive_backend="mfront-3d-condensed-plane-stress",
        mfront_behaviour_id=SRIX,
        mfront_library=_library(),
    )
    workflow = PartitionWorkflow(
        config=_replace(case.config, solver=solver),
        layout=PartitionLayout(global_shape=(4, 4), partition_shape=(2, 2), padding=0),
        displacement_x_mm=case.displacement_x_mm,
        displacement_y_mm=case.displacement_y_mm,
        yield_stress_mpa=case.yield_stress_mpa,
        hardening_coefficient_mpa=case.hardening_coefficient_mpa,
        output_directory=tmp_path / "campaign",
    )

    manifest = workflow._manifest_data()

    assert "PLASTIC_SLIP" in manifest["result_fields"]
    assert "CUMULATED_SLIP" in manifest["result_fields"]
    assert (
        "NOT an equivalent plastic strain"
        in manifest["result_field_metadata"]["CUMULATED_SLIP"]["components"]
    )
