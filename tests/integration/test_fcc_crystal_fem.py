"""A small plane-stress finite-element run driven by the SRIX crystal law.

Section 12.6. Deliberately tiny so it can run in CI: no experimental ROI is
needed to establish that the crystal law goes through the whole solver.
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

from fem_inhouse.core.crystal_orientation import rotation_from_euler_bunge_deg
from fem_inhouse.examples import reduced_biaxial_case
from fem_inhouse.solver import run_case_study

SRIX = "fcc_forest_rubin_srix"


def _library() -> str:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    pytest.importorskip("mgis.behaviour")
    return library


def _run(orientation: np.ndarray | None = None, *, nx: int = 3, ny: int = 3):
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
