from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.core.srix_parameters import get_parameter_set
from fem_inhouse.identification.srix_parameter_coordinates import SrixTheta9


def test_srix_theta9_round_trip_default_preset():
    preset = get_parameter_set("316l_srix_transposed_from_nasri2018_rate_1e-3")
    theta = SrixTheta9.from_parameter_set(preset)
    recovered = SrixTheta9.from_log_coordinates(theta.log_coordinates())
    np.testing.assert_allclose(recovered.as_physical_array(), theta.as_physical_array())


def test_srix_theta9_runtime_overrides_are_complete():
    preset = get_parameter_set("316l_srix_transposed_from_nasri2018_rate_1e-3")
    overrides = SrixTheta9.from_parameter_set(preset).as_runtime_overrides()
    assert set(overrides) == {
        "C11_mpa", "C12_mpa", "C44_mpa", "tau0_mpa", "R_mpa", "Q_mpa",
        "b", "C_mpa", "d",
    }


def test_srix_theta9_log_coordinates_preserve_cubic_stability():
    eta = np.array([10.0, 11.0, 12.0, 3.0, 2.0, 1.0, 0.5, 8.0, 6.0])
    theta = SrixTheta9.from_log_coordinates(eta)
    assert theta.c11_mpa > theta.c12_mpa
    assert theta.c11_mpa + 2.0 * theta.c12_mpa > 0.0
    with pytest.raises(ValueError):
        SrixTheta9.from_log_coordinates(np.full(8, 1.0))
