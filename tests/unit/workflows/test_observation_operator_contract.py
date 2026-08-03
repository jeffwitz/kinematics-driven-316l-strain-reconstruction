"""Contract tests for the symmetric FEM-to-DIC observation operator.

These cover the synthetic validation cases of the observed-EVM comparison
specification that were not already exercised elsewhere: affine and shear
deformation, an integrated band in both orientations, absence of an implicit
transpose or flip, determinism, and refusal of an unusable measurement
configuration.

Rigid translation, canonical/image inverses and the historical U/V mapping are
covered in `tests/unit/measurement/`; they are not duplicated here.
"""

from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.measurement import (
    canonical_to_image_flow,
    disflow_profile,
    image_flow_to_canonical,
    require_native_finest_scale,
    warp_forward_displacement,
)
from fem_inhouse.measurement.disflow import DISFlowConfig
from fem_inhouse.measurement.warp import WARP_BORDER_MODE, WARP_INTERPOLATION
from fem_inhouse.workflows.nonlocality_diagnostic import reconstruct_historical_evm

PIXEL = 0.00184
#: Deliberately non-square so a transpose cannot pass unnoticed.
SHAPE = (48, 37)


def _speckle(shape: tuple[int, int]) -> np.ndarray:
    generator = np.random.default_rng(11)
    coarse = generator.random((shape[0] // 2 + 2, shape[1] // 2 + 2))
    fine = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)[: shape[0], : shape[1]]
    return np.clip(fine * 255.0, 0, 255).astype(np.uint8)


def _grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(shape[0], dtype=np.float64)[:, None] * PIXEL
    y = np.arange(shape[1], dtype=np.float64)[None, :] * PIXEL
    return np.broadcast_to(x, shape), np.broadcast_to(y, shape)


def _affine_displacement(shape, exx=0.0, eyy=0.0, gxy=0.0):
    x, y = _grid(shape)
    field = np.empty((*shape, 2))
    field[..., 0] = exx * x + 0.5 * gxy * y
    field[..., 1] = 0.5 * gxy * x + eyy * y
    return field


def test_affine_deformation_reconstructs_its_imposed_evm() -> None:
    imposed = _affine_displacement(SHAPE, exx=2.0e-3, eyy=-6.0e-4)

    evm = reconstruct_historical_evm(
        imposed, spacing_x_mm=PIXEL, spacing_y_mm=PIXEL, poisson_ratio=0.3
    )

    # A homogeneous affine field must give a spatially uniform EVM.
    assert float(np.std(evm)) == pytest.approx(0.0, abs=1.0e-12)
    assert float(np.mean(evm)) > 0.0


def test_simple_shear_is_not_confused_with_extension() -> None:
    shear = _affine_displacement(SHAPE, gxy=1.5e-3)
    extension = _affine_displacement(SHAPE, exx=1.5e-3)

    shear_evm = reconstruct_historical_evm(
        shear, spacing_x_mm=PIXEL, spacing_y_mm=PIXEL, poisson_ratio=0.3
    )
    extension_evm = reconstruct_historical_evm(
        extension, spacing_x_mm=PIXEL, spacing_y_mm=PIXEL, poisson_ratio=0.3
    )

    assert float(np.std(shear_evm)) == pytest.approx(0.0, abs=1.0e-12)
    assert not np.isclose(float(np.mean(shear_evm)), float(np.mean(extension_evm)))


def _banded_displacement(shape, *, vertical: bool, width: float = 4.0):
    """A localised shear band, as a smooth step in the transverse displacement."""

    axis = 0 if vertical else 1
    n = np.arange(shape[axis], dtype=np.float64)
    profile = np.tanh((n - shape[axis] / 2.0) / width) * 2.0e-3 * PIXEL * shape[axis]
    field = np.zeros((*shape, 2))
    if vertical:
        field[..., 0] = profile[:, None]
    else:
        field[..., 1] = profile[None, :]
    return field


@pytest.mark.parametrize("vertical", [True, False])
def test_integrated_band_is_localised_in_the_expected_direction(vertical: bool) -> None:
    field = _banded_displacement(SHAPE, vertical=vertical)

    evm = reconstruct_historical_evm(
        field, spacing_x_mm=PIXEL, spacing_y_mm=PIXEL, poisson_ratio=0.3
    )

    along = 1 if vertical else 0
    across = 0 if vertical else 1
    # Uniform along the band, peaked across it.
    assert float(np.std(evm.mean(axis=across))) < 1.0e-12
    assert float(np.std(evm.mean(axis=along))) > 0.0
    peak = int(np.argmax(evm.mean(axis=along)))
    assert abs(peak - SHAPE[across] // 2) <= 1


def test_the_two_orientations_are_mirror_images_not_the_same_field() -> None:
    vertical = reconstruct_historical_evm(
        _banded_displacement(SHAPE, vertical=True),
        spacing_x_mm=PIXEL,
        spacing_y_mm=PIXEL,
        poisson_ratio=0.3,
    )
    horizontal = reconstruct_historical_evm(
        _banded_displacement(SHAPE, vertical=False),
        spacing_x_mm=PIXEL,
        spacing_y_mm=PIXEL,
        poisson_ratio=0.3,
    )

    assert vertical.shape == horizontal.shape
    # A transpose bug would make one the transpose of the other.
    assert not np.allclose(vertical, horizontal)


def test_flow_conversion_does_not_transpose_or_flip() -> None:
    field = np.zeros((*SHAPE, 2))
    field[3, 5, 0] = 7.0 * PIXEL  # a single ux spike at a known asymmetric spot

    flow = canonical_to_image_flow(field, pixel_size_mm=PIXEL)

    assert flow.shape == (*SHAPE, 2)
    # canonical ux must land on the image ROW component, at the same index.
    assert flow[3, 5, 1] == pytest.approx(7.0)
    assert flow[3, 5, 0] == pytest.approx(0.0)
    assert flow[5, 3, 1] == pytest.approx(0.0)
    np.testing.assert_allclose(
        image_flow_to_canonical(flow, pixel_size_mm=PIXEL), field, atol=1.0e-15
    )


@pytest.mark.measurement
def test_warp_is_deterministic_bit_for_bit() -> None:
    # The only test in this file that reaches OpenCV: `warp_forward_displacement`
    # imports `cv2`. Everything else here is a pure-numpy contract check, so the
    # marker belongs on this test rather than on the module. The marker routes it
    # into the measurement job; `importorskip` is what keeps it out of the locked
    # quality environment, which deliberately installs no OpenCV.
    pytest.importorskip("cv2")

    reference = _speckle(SHAPE)
    flow = canonical_to_image_flow(
        _affine_displacement(SHAPE, exx=1.0e-3), pixel_size_mm=PIXEL
    )

    first = warp_forward_displacement(reference, flow, mode="iterative_forward_inverse")
    second = warp_forward_displacement(reference, flow, mode="iterative_forward_inverse")

    np.testing.assert_array_equal(first.image, second.image)
    assert first.residual_pixels == second.residual_pixels
    assert first.iterations == second.iterations


def test_resampling_contract_is_named_for_the_manifest() -> None:
    # These strings are recorded in every observation manifest; a silent change
    # to the resampling would otherwise be invisible in archived reports.
    assert WARP_INTERPOLATION == "cv2.INTER_LINEAR"
    assert WARP_BORDER_MODE == "cv2.BORDER_REFLECT101"


def test_metrological_use_refuses_a_coarse_finest_scale() -> None:
    with pytest.raises(ValueError, match="finest_scale=0"):
        require_native_finest_scale(DISFlowConfig(finest_scale=1))


def test_metrological_use_refuses_an_unset_finest_scale() -> None:
    with pytest.raises(ValueError, match="finest_scale=0"):
        require_native_finest_scale(DISFlowConfig(finest_scale=None))


@pytest.mark.parametrize("name", ["legacy_script_2021", "declared_medium_v4"])
def test_archived_profiles_remain_metrologically_usable(name: str) -> None:
    require_native_finest_scale(disflow_profile(name).config)
