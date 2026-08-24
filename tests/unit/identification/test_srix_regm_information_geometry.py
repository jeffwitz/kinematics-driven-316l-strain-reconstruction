from __future__ import annotations

import numpy as np

from scripts.qualify_srix_regm_information_geometry import _geometry


def test_geometry_reports_fisher_svd_and_correlations() -> None:
    matrix = np.diag([4.0, 2.0, 1.0, 0.1])
    result = _geometry(matrix)
    np.testing.assert_allclose(result["singular_values"], [4.0, 2.0, 1.0, 0.1])
    assert result["numerical_rank"] == 4
    np.testing.assert_allclose(result["fisher"], matrix.T @ matrix)
    assert np.all(np.isfinite(result["correlation"]))
