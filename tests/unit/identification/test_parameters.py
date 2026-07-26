from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.identification.parameters import (
    NonlocalIdentificationPoint,
    from_h_chi_and_a_chi,
)


def test_parameter_coordinates_round_trip_exactly() -> None:
    point = NonlocalIdentificationPoint.from_alpha_and_length_um(
        alpha=2.5,
        length_scale_um=40.0,
        h_ref_mpa=5_000.0,
    )
    reconstructed = from_h_chi_and_a_chi(
        h_chi_mpa=point.h_chi_mpa,
        a_chi_mpa_mm2=point.a_chi_mpa_mm2 or 0.0,
        h_ref_mpa=point.h_ref_mpa,
    )

    assert reconstructed.alpha == pytest.approx(point.alpha)
    assert reconstructed.length_scale_mm == pytest.approx(point.length_scale_mm)
    assert reconstructed.a_chi_mpa_mm2 == pytest.approx(point.a_chi_mpa_mm2)
    assert reconstructed.a_chi_mpa_um2 == pytest.approx(
        point.h_chi_mpa * 40.0**2
    )


def test_local_point_is_unique_and_independent_of_length() -> None:
    first = NonlocalIdentificationPoint.from_alpha_and_length_um(
        alpha=0.0,
        length_scale_um=20.0,
        h_ref_mpa=4_000.0,
    )
    second = NonlocalIdentificationPoint.from_alpha_and_length_um(
        alpha=0.0,
        length_scale_um=60.0,
        h_ref_mpa=4_000.0,
    )

    assert first == second
    assert first.length_scale_mm is None
    assert first.a_chi_mpa_mm2 is None
    assert first.theta_h is None
    assert first.theta_a is None


@pytest.mark.parametrize(
    ("alpha", "length_um", "h_ref"),
    [
        (-1.0, 20.0, 5_000.0),
        (1.0, None, 5_000.0),
        (1.0, 0.0, 5_000.0),
        (1.0, 20.0, 0.0),
        (np.nan, 20.0, 5_000.0),
    ],
)
def test_invalid_parameter_coordinates_are_rejected(
    alpha: float,
    length_um: float | None,
    h_ref: float,
) -> None:
    with pytest.raises(ValueError):
        NonlocalIdentificationPoint.from_alpha_and_length_um(
            alpha=alpha,
            length_scale_um=length_um,
            h_ref_mpa=h_ref,
        )
