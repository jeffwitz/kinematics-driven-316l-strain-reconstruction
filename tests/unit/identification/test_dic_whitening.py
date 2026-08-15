from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.identification.dic_whitening import (
    DICSpectralTransfer,
    DICSpectralWhitener,
)


def _correlated_noise(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((32, 12, 10, 2))
    frequencies_x = np.fft.fftfreq(12)[:, None]
    frequencies_y = np.fft.fftfreq(10)[None, :]
    low_pass = np.exp(-30.0 * (frequencies_x**2 + frequencies_y**2))
    transformed = np.fft.fftn(white, axes=(1, 2), norm="ortho")
    return np.fft.ifftn(
        transformed * low_pass[None, :, :, None],
        axes=(1, 2),
        norm="ortho",
    ).real


def test_spectral_whitener_is_self_adjoint() -> None:
    whitener = DICSpectralWhitener.from_noise_realisations(_correlated_noise())
    rng = np.random.default_rng(20260814)
    left = rng.standard_normal(whitener.field_shape)
    right = rng.standard_normal(whitener.field_shape)

    lhs = np.vdot(whitener.apply(left), right).real
    rhs = np.vdot(left, whitener.adjoint(right)).real

    assert lhs == pytest.approx(rhs, rel=2.0e-13, abs=1.0e-12)


def test_normal_action_and_quadratic_misfit_are_consistent() -> None:
    whitener = DICSpectralWhitener.from_noise_realisations(_correlated_noise())
    rng = np.random.default_rng(7)
    field = rng.standard_normal(whitener.field_shape)

    normal = whitener.normal_action(field)
    expected = np.vdot(field, normal).real

    assert expected > 0.0
    assert 2.0 * whitener.quadratic_misfit(field) == pytest.approx(
        expected, rel=5.0e-13
    )


def test_low_power_modes_are_penalised_more_than_measured_noise_modes() -> None:
    whitener = DICSpectralWhitener.from_noise_realisations(_correlated_noise())
    nx, ny, components = whitener.field_shape
    low_frequency = np.ones((nx, ny, components))
    checkerboard = (-1.0) ** (
        np.arange(nx)[:, None, None] + np.arange(ny)[None, :, None]
    )
    checkerboard = np.broadcast_to(checkerboard, (nx, ny, components))

    low_penalty = np.linalg.norm(whitener.apply(low_frequency)) / np.linalg.norm(
        low_frequency
    )
    high_penalty = np.linalg.norm(whitener.apply(checkerboard)) / np.linalg.norm(
        checkerboard
    )

    assert high_penalty > 10.0 * low_penalty


def test_zero_noise_and_incompatible_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="zero spectral power"):
        DICSpectralWhitener.from_noise_realisations(np.zeros((2, 4, 5, 2)))

    whitener = DICSpectralWhitener.from_noise_realisations(_correlated_noise())
    with pytest.raises(ValueError, match="field must have shape"):
        whitener.apply(np.zeros((4, 5, 2)))


def test_stationary_field_builds_a_reproducible_target_grid_whitener() -> None:
    field = _correlated_noise()[0]
    first = DICSpectralWhitener.from_stationary_noise_field(
        field,
        target_shape=(7, 6),
        sample_count=12,
        seed=19,
    )
    second = DICSpectralWhitener.from_stationary_noise_field(
        field,
        target_shape=(7, 6),
        sample_count=12,
        seed=19,
    )

    assert first.field_shape == (7, 6, 2)
    np.testing.assert_array_equal(
        first.power_spectral_density,
        second.power_spectral_density,
    )


def test_stationary_field_rejects_an_impossible_target_shape() -> None:
    with pytest.raises(ValueError, match="target_shape"):
        DICSpectralWhitener.from_stationary_noise_field(
            _correlated_noise()[0],
            target_shape=(100, 3),
        )


def test_spatial_mean_removal_explicitly_controls_the_dc_uncertainty() -> None:
    noise = _correlated_noise() + np.array([2.0, -3.0])
    retained = DICSpectralWhitener.from_noise_realisations(
        noise,
        remove_spatial_mean=False,
    )
    removed = DICSpectralWhitener.from_noise_realisations(
        noise,
        remove_spatial_mean=True,
    )

    assert np.all(retained.power_spectral_density[0, 0] > 1.0)
    assert np.all(removed.power_spectral_density[0, 0] < 1.0e-25)


def test_masked_whitener_has_the_exact_non_self_adjoint_pair() -> None:
    noise = _correlated_noise()
    mask = np.ones(noise.shape[1:])
    mask[[0, -1], :, :] = 0.0
    mask[:, [0, -1], :] = 0.0
    whitener = DICSpectralWhitener.from_noise_realisations(
        noise,
        remove_spatial_mean=True,
        support_mask=mask,
    )
    rng = np.random.default_rng(88)
    left = rng.normal(size=whitener.field_shape)
    right = rng.normal(size=whitener.field_shape)

    lhs = float(np.vdot(whitener.apply(left), right).real)
    rhs = float(np.vdot(left, whitener.adjoint(right)).real)

    np.testing.assert_allclose(lhs, rhs, rtol=2.0e-13, atol=1.0e-11)
    np.testing.assert_array_equal(whitener.normal_action(left)[0], 0.0)
    np.testing.assert_array_equal(whitener.normal_action(left)[-1], 0.0)


def test_spectral_transfer_is_self_adjoint_and_suppresses_short_wavelengths() -> None:
    transfer = DICSpectralTransfer(
        wavelengths_pixels=np.array([4.0, 8.0, 16.0, 64.0]),
        gains=np.array([0.01, 0.1, 0.5, 0.95]),
    )
    rng = np.random.default_rng(8)
    left = rng.standard_normal((32, 32, 2))
    right = rng.standard_normal((32, 32, 2))
    np.testing.assert_allclose(
        np.vdot(transfer.apply(left), right),
        np.vdot(left, transfer.adjoint(right)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    long_wave = np.ones((32, 32, 2))
    short_wave = (-1.0) ** (
        np.arange(32)[:, None, None] + np.arange(32)[None, :, None]
    )
    short_wave = np.broadcast_to(short_wave, long_wave.shape)
    assert np.linalg.norm(transfer.apply(short_wave)) < np.linalg.norm(
        transfer.apply(long_wave)
    )


def test_the_periodic_transfer_distorts_an_affine_field_and_the_wrap_free_one_does_not() -> None:
    """A low-pass must leave an affine field alone. The plain FFT does not.

    `apply` filters through `fftn`, which treats the crop as periodic, and a
    displacement ramp is discontinuous across the wrap. On a 100x100 crop at one
    per cent strain the error reaches nine DIC sigma, concentrated in a border
    band -- and measured against the P43 history that artefact carried most of
    the residual to an elastic model. The regression guards both halves: that
    the plain transfer really does distort, so the motivation cannot quietly
    disappear, and that the wrap-free one is exact.
    """

    import numpy as np

    from fem_inhouse.identification.dic_whitening import DICSpectralTransfer

    transfer = DICSpectralTransfer(
        wavelengths_pixels=np.array([2.0, 4.0, 8.0, 16.0, 64.0, 512.0]),
        gains=np.array([0.05, 0.35, 0.72, 0.9, 0.98, 1.0]),
    )
    pixel_size_mm = 0.00184
    nodes = 101
    x, y = np.meshgrid(
        np.arange(nodes) * pixel_size_mm, np.arange(nodes) * pixel_size_mm, indexing="ij"
    )
    affine = np.zeros((nodes, nodes, 2))
    affine[:, :, 0] = 1.0e-2 * x
    affine[:, :, 1] = -3.0e-3 * y

    periodic_error = np.abs(np.asarray(transfer.apply(affine)) - affine).max()
    assert periodic_error > 20.0 * 9.4e-5 / 10.0  # several DIC sigma, comfortably

    wrap_free = np.asarray(transfer.apply_without_wrap(affine))
    np.testing.assert_allclose(wrap_free, affine, rtol=0.0, atol=1e-14)

    # A genuinely high-frequency field must still be filtered the same way.
    generator = np.random.default_rng(3)
    rough = generator.normal(size=affine.shape) * 1.0e-4
    np.testing.assert_allclose(
        transfer.apply_without_wrap(rough), transfer.apply(rough), rtol=0.0, atol=2e-6
    )
