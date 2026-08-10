from __future__ import annotations

import numpy as np

from fem_inhouse.spectral2d.coupled_blocks import (
    make_dct_helmholtz_inverse,
    make_dst_b0_inverse,
)
from fem_inhouse.spectral2d.green import B0Green2D
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.spectral2d.transform_factory import create_full_dirichlet_dsti_plan
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig


def test_dct_helmholtz_inverse_preserves_constant_field() -> None:
    inverse = make_dct_helmholtz_inverse(
        (5, 4), length_scale=0.7, spacing_x=1.0, spacing_y=1.0
    )
    values = np.full(20, 3.25)
    np.testing.assert_allclose(inverse(values), values, rtol=0.0, atol=1.0e-13)


def test_dst_b0_inverse_has_the_expected_flat_vector_contract() -> None:
    grid = StructuredGrid2D(5, 4, 5.0, 4.0)
    plan = create_full_dirichlet_dsti_plan(grid, SpectralTransformConfig())
    symbols = TwoSubcellDiagnostic2D(grid).reference_operator_symbols(plan)
    green = B0Green2D(symbols, lambda_0=2.0, mu_0=3.0)
    inverse = make_dst_b0_inverse(plan, green)
    values = np.arange(4 * 3 * 2, dtype=float)
    result = inverse(values)
    assert result.shape == values.shape
    assert np.isfinite(result).all()
