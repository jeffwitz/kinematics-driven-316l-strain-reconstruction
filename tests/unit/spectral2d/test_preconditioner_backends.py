from __future__ import annotations

import numpy as np

from fem_inhouse.spectral2d.green import B0Green2D, ReferenceOperatorSymbols


def test_green_apply_into_matches_compatibility_wrapper() -> None:
    values = np.linspace(0.1, 2.0, 12).reshape(2, 2, 3)
    symbols = ReferenceOperatorSymbols(values[..., 0], values[..., 1], values[..., 2])
    green = B0Green2D(symbols, lambda_0=2.0, mu_0=3.0)
    polarization = np.random.default_rng(8).normal(size=(2, 2, 2))
    destination = np.empty_like(polarization)

    green.apply_into(polarization, destination)

    np.testing.assert_allclose(destination, green.apply(polarization))
