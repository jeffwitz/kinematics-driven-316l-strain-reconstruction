from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

from fem_inhouse.examples import reduced_biaxial_case
from fem_inhouse.solver import run_case_study


@pytest.mark.mfront
def test_mfront_backend_matches_python_through_the_complete_newton_loop() -> None:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")

    case = reduced_biaxial_case(nx=4, ny=4)
    python_solver = replace(
        case.config.solver,
        hardening_mode="ludwik",
        residual_tolerance=1e-8,
        constitutive_backend="python",
    )
    mfront_solver = replace(
        python_solver,
        constitutive_backend="mfront",
        mfront_library=library,
        mfront_threads=2,
    )

    results = {}
    for backend, solver in (("python", python_solver), ("mfront", mfront_solver)):
        results[backend] = run_case_study(
            replace(case.config, solver=solver),
            displacement_x_mm=case.displacement_x_mm,
            displacement_y_mm=case.displacement_y_mm,
            yield_stress_mpa=case.yield_stress_mpa,
            hardening_coefficient_mpa=case.hardening_coefficient_mpa,
        )

    python = results["python"]
    mfront = results["mfront"]
    assert mfront.diagnostics is not None
    assert mfront.diagnostics.cutbacks == 0
    assert "constitutive=mfront" in mfront.diagnostics.backend
    assert np.max(mfront.equivalent_plastic_strain) > 0
    np.testing.assert_allclose(
        mfront.displacement_mm, python.displacement_mm, rtol=1e-8, atol=1e-12
    )
    np.testing.assert_allclose(mfront.stress_mpa, python.stress_mpa, rtol=1e-8, atol=1e-7)
    np.testing.assert_allclose(
        mfront.plastic_strain,
        python.plastic_strain,
        rtol=1e-8,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        mfront.equivalent_plastic_strain,
        python.equivalent_plastic_strain,
        rtol=1e-8,
        atol=1e-12,
    )
    np.testing.assert_allclose(mfront.reaction_force, python.reaction_force, rtol=1e-8, atol=1e-10)
    assert python.diagnostics is not None
    assert python.diagnostics.tensor_reconstruction_source == "python_analytical"
    assert mfront.diagnostics.tensor_reconstruction_source == "mfront_native_axial_strain"

    tensor_fields = (
        "stress_tensor_mpa",
        "total_strain_tensor",
        "elastic_strain_tensor",
        "plastic_strain_tensor",
    )
    for field_name in tensor_fields:
        python_field = getattr(python, field_name)
        mfront_field = getattr(mfront, field_name)
        assert python_field is not None
        assert mfront_field is not None
        np.testing.assert_allclose(
            mfront_field,
            python_field,
            rtol=1e-6,
            atol=1e-6 if field_name == "stress_tensor_mpa" else 1e-10,
        )

    assert python.plane_stress_residual_mpa is not None
    assert mfront.plane_stress_residual_mpa is not None
    np.testing.assert_array_equal(python.plane_stress_residual_mpa, 0.0)
    assert np.max(np.abs(mfront.plane_stress_residual_mpa)) <= 1e-9
    for result in (python, mfront):
        assert result.total_strain_tensor is not None
        assert result.elastic_strain_tensor is not None
        assert result.plastic_strain_tensor is not None
        np.testing.assert_allclose(
            result.total_strain_tensor,
            result.elastic_strain_tensor + result.plastic_strain_tensor,
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            np.trace(result.plastic_strain_tensor, axis1=-2, axis2=-1),
            0.0,
            rtol=0.0,
            atol=1e-12,
        )
