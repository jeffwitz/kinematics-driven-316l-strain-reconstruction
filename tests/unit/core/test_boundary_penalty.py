"""Penalty enforcement of the measured boundary, and its misfit indicator.

Elimination stays the default and the reference. Penalty keeps the prescribed
degrees of freedom in the system with a finite spring, so the gap between the
measurement and the value the solver actually imposes becomes an output field.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fem_inhouse.core import nonlinear
from fem_inhouse.core.nonlinear import _csr_diagonal_indices
from fem_inhouse.examples import reduced_biaxial_case


def _kwargs():
    case = reduced_biaxial_case(nx=5, ny=5, constitutive_backend="python")
    mesh = case.config.mesh
    return {
        "disp_x": np.asarray(case.displacement_x_mm),
        "disp_y": np.asarray(case.displacement_y_mm),
        "yield_map": np.asarray(case.yield_stress_mpa),
        "K_map": np.asarray(case.hardening_coefficient_mpa),
        "n_exp": case.config.material.hardening_exponent,
        "x_size": mesh.nx * mesh.base_pixel_size_mm,
        "y_size": mesh.ny * mesh.base_pixel_size_mm,
        "element_size": mesh.base_pixel_size_mm,
        "scale_factor": mesh.scale_factor,
        "N_inc": case.config.solver.increments,
        "constitutive_backend": "python",
        "verbose": False,
    }


def test_csr_diagonal_indices_finds_the_stored_diagonal() -> None:
    matrix = csr_matrix(np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]]))

    positions = _csr_diagonal_indices(matrix, np.array([0, 2]))

    np.testing.assert_allclose(matrix.data[positions], [2.0, 4.0])


def test_csr_diagonal_indices_rejects_a_missing_diagonal() -> None:
    matrix = csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))

    with pytest.raises(ValueError, match="no stored diagonal"):
        _csr_diagonal_indices(matrix, np.array([0]))


def test_penalty_rejects_an_unknown_enforcement() -> None:
    with pytest.raises(ValueError, match="elimination or penalty"):
        nonlinear.run_fem(**_kwargs(), boundary_enforcement="lagrange")


def test_penalty_rejects_a_nonpositive_stiffness() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        nonlinear.run_fem(
            **_kwargs(), boundary_enforcement="penalty", boundary_penalty_stiffness=0.0
        )


def test_penalty_converges_to_elimination_as_the_spring_stiffens() -> None:
    shared = _kwargs()
    reference = nonlinear.run_fem(**shared)

    errors = {}
    for stiffness in (1.0e8, 1.0e10):
        result = nonlinear.run_fem(
            **shared,
            boundary_enforcement="penalty",
            boundary_penalty_stiffness=stiffness,
        )
        errors[stiffness] = float(np.max(np.abs(result["U"] - reference["U"])))

    # The consistency error of a penalty method scales as one over the spring.
    assert errors[1.0e8] < 1.0e-6
    assert errors[1.0e10] < errors[1.0e8] / 10.0


def test_penalty_exposes_the_boundary_misfit_as_a_field() -> None:
    shared = _kwargs()
    stiffness = 1.0e8
    result = nonlinear.run_fem(
        **shared,
        boundary_enforcement="penalty",
        boundary_penalty_stiffness=stiffness,
    )

    misfit = result["BOUNDARY_MISFIT"]
    interior = misfit[1:-1, 1:-1, :]

    assert misfit.shape == result["U"].shape
    assert np.max(np.abs(misfit)) > 0.0
    np.testing.assert_array_equal(interior, np.zeros_like(interior))
    # The reaction is the spring force conjugate to that gap.
    np.testing.assert_allclose(
        result["RF"], stiffness * misfit, rtol=1.0e-9, atol=1.0e-12
    )


def test_elimination_stays_the_default_and_reports_no_misfit() -> None:
    result = nonlinear.run_fem(**_kwargs())

    np.testing.assert_array_equal(
        result["BOUNDARY_MISFIT"], np.zeros_like(result["BOUNDARY_MISFIT"])
    )
