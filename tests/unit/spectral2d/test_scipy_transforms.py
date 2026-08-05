from __future__ import annotations

import numpy as np

from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.transform_factory import create_full_dirichlet_dsti_plan
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig
from fem_inhouse.spectral2d.transforms_scipy import FullDirichletDSTIPlan2D


def test_scipy_plan_is_batched_and_orthonormal() -> None:
    plan = FullDirichletDSTIPlan2D(StructuredGrid2D(12, 8, 12.0, 8.0))
    field = np.random.default_rng(12).normal(size=(*plan.interior_shape, 2))
    transformed = plan.forward_displacement(field)

    np.testing.assert_allclose(plan.inverse_displacement(transformed), field, atol=1.0e-13)
    for component in range(2):
        np.testing.assert_allclose(
            transformed[..., component],
            plan.forward_displacement(field[..., component])[..., 0]
            if field[..., component].ndim == 3
            else plan.forward_displacement(field[..., component]),
            atol=1.0e-13,
        )


def test_scipy_buffered_and_factory_contract() -> None:
    grid = StructuredGrid2D(8, 8, 1.0, 1.0)
    plan = create_full_dirichlet_dsti_plan(grid, SpectralTransformConfig())
    field = np.random.default_rng(3).normal(size=(*grid.interior_shape, 2))
    transformed = np.empty_like(field)
    recovered = np.empty_like(field)

    plan.forward_into(field, transformed)
    plan.inverse_into(transformed, recovered)

    np.testing.assert_allclose(recovered, field, atol=1.0e-13)
    assert plan.diagnostics.backend == "scipy"
    assert plan.diagnostics.batch_components == 2
