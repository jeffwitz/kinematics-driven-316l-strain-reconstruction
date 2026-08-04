import numpy as np
import pytest

from fem_inhouse.spectral2d import (
    Spectral2DConfig,
    Spectral2DDiagnostics,
    Spectral2DResult,
)


def test_spectral_config_validates_registered_choices() -> None:
    assert Spectral2DConfig().green_operator == "b0"
    with pytest.raises(ValueError, match="spatial_scheme"):
        Spectral2DConfig(spatial_scheme="bad")  # type: ignore[arg-type]


def test_result_pixel_average_is_explicit_for_tri2() -> None:
    diagnostics = Spectral2DDiagnostics(
        spatial_scheme="tri2",
        green_operator="b0",
        pixels=(2, 3),
        material_points=12,
        points_per_pixel=2,
        spacing_x=1.0,
        spacing_y=1.0,
    )
    result = Spectral2DResult(
        displacement=np.zeros((3, 4, 2)),
        applied_displacement=np.zeros((3, 4, 2)),
        fluctuation_displacement=np.zeros((3, 4, 2)),
        strain_in_plane=np.zeros((2, 3, 2, 3)),
        stress_in_plane_mpa=np.zeros((2, 3, 2, 3)),
        full_stress_tensor_mpa=None,
        full_strain_tensor=None,
        elastic_strain_tensor=None,
        plastic_strain_tensor=None,
        observables={},
        reaction_forces=np.zeros((3, 4, 2)),
        diagnostics=diagnostics,
    )
    raw = np.arange(12.0).reshape(2, 3, 2)
    np.testing.assert_allclose(result.pixel_average(raw), raw.mean(axis=2))
    with pytest.raises(ValueError, match="axis 2"):
        result.pixel_average(np.zeros((2, 3, 3)))
