"""Matrix-free spectral whitening from measured DIC noise realisations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DICSpectralTransfer:
    """Self-adjoint approximation of the measured DIC spatial transfer."""

    wavelengths_pixels: FloatArray
    gains: FloatArray

    def __post_init__(self) -> None:
        wavelengths = np.asarray(self.wavelengths_pixels, dtype=np.float64)
        gains = np.asarray(self.gains, dtype=np.float64)
        if wavelengths.ndim != 1 or gains.shape != wavelengths.shape:
            raise ValueError("wavelengths and gains must be one-dimensional and matched")
        if (
            wavelengths.size < 2
            or np.any(wavelengths <= 0.0)
            or not np.all(np.diff(wavelengths) > 0.0)
        ):
            raise ValueError("wavelengths must be strictly increasing and positive")
        if not np.isfinite(gains).all() or np.any(gains < 0.0):
            raise ValueError("gains must be finite and non-negative")
        object.__setattr__(self, "wavelengths_pixels", wavelengths.copy())
        object.__setattr__(self, "gains", gains.copy())

    @classmethod
    def from_sinusoidal_csv(cls, path: str | Path) -> DICSpectralTransfer:
        """Load an isotropic average of the archived horizontal/vertical gains."""

        rows = np.genfromtxt(path, delimiter=",", names=True)
        wavelengths = np.asarray(rows["wavelength_pixels"], dtype=np.float64)
        gains = np.asarray(rows["gain"], dtype=np.float64)
        unique = np.unique(wavelengths)
        averaged = np.array([np.mean(gains[wavelengths == value]) for value in unique])
        return cls(unique, averaged)

    def _multiplier(self, shape: tuple[int, int, int]) -> FloatArray:
        frequencies_x = np.fft.fftfreq(shape[0])[:, None]
        frequencies_y = np.fft.fftfreq(shape[1])[None, :]
        frequency = np.sqrt(frequencies_x**2 + frequencies_y**2)
        wavelength = np.divide(
            1.0,
            frequency,
            out=np.full_like(frequency, np.inf),
            where=frequency > 0.0,
        )
        multiplier = np.interp(
            np.log(np.maximum(wavelength, self.wavelengths_pixels[0])),
            np.log(self.wavelengths_pixels),
            self.gains,
            left=float(self.gains[0]),
            right=float(self.gains[-1]),
        )
        multiplier[frequency == 0.0] = 1.0
        return np.broadcast_to(multiplier[..., None], shape)

    def apply(self, values: ArrayLike) -> FloatArray:
        """Apply the transfer function to a displacement field."""

        field = np.asarray(values, dtype=np.float64)
        if field.ndim != 3 or not np.isfinite(field).all():
            raise ValueError("values must be a finite field with shape (nx, ny, components)")
        transformed = np.fft.fftn(field, axes=(0, 1), norm="ortho")
        return np.asarray(
            np.fft.ifftn(
                transformed * self._multiplier(field.shape), axes=(0, 1), norm="ortho"
            ).real,
            dtype=np.float64,
        )

    def adjoint(self, values: ArrayLike) -> FloatArray:
        """Apply the exact adjoint (the fitted transfer is real and symmetric)."""

        return self.apply(values)


def _opposite_frequency_indices(size: int) -> NDArray[np.int64]:
    return (-np.arange(size, dtype=np.int64)) % size


@dataclass(frozen=True, slots=True)
class DICSpectralWhitener:
    """Component-wise stationary DIC covariance inverse square root.

    The first P0 intentionally models each displacement component with its own
    spatial PSD and neglects cross-component covariance.  The operator uses
    orthonormal FFTs, is self-adjoint, and never forms a dense covariance.
    """

    power_spectral_density: FloatArray
    spectral_floor: float
    support_mask: FloatArray | None = None

    def __post_init__(self) -> None:
        psd = np.asarray(self.power_spectral_density, dtype=np.float64)
        if psd.ndim != 3 or psd.shape[-1] < 1:
            raise ValueError("power_spectral_density must have shape (nx, ny, components)")
        if not np.isfinite(psd).all() or np.any(psd < 0.0):
            raise ValueError("power_spectral_density must be finite and non-negative")
        if not np.isfinite(self.spectral_floor) or self.spectral_floor <= 0.0:
            raise ValueError("spectral_floor must be finite and positive")
        object.__setattr__(self, "power_spectral_density", psd.copy())
        self.power_spectral_density.setflags(write=False)
        if self.support_mask is not None:
            mask = np.asarray(self.support_mask, dtype=np.float64)
            try:
                mask = np.broadcast_to(mask, psd.shape).copy()
            except ValueError as error:
                raise ValueError("support_mask must broadcast to the field shape") from error
            if not np.isfinite(mask).all() or np.any((mask < 0.0) | (mask > 1.0)):
                raise ValueError("support_mask must be finite and lie between zero and one")
            if not np.any(mask > 0.0):
                raise ValueError("support_mask must contain a non-zero value")
            mask.setflags(write=False)
            object.__setattr__(self, "support_mask", mask)

    @classmethod
    def from_noise_realisations(
        cls,
        noise_realisations: ArrayLike,
        *,
        relative_floor: float = 1.0e-6,
        absolute_floor: float = 0.0,
        remove_spatial_mean: bool = False,
        support_mask: ArrayLike | None = None,
    ) -> DICSpectralWhitener:
        """Estimate a stationary component-wise PSD from repeated-frame noise."""

        noise = np.asarray(noise_realisations, dtype=np.float64)
        if noise.ndim != 4 or noise.shape[0] < 2 or noise.shape[-1] < 1:
            raise ValueError(
                "noise_realisations must have shape (samples, nx, ny, components)"
            )
        if not np.isfinite(noise).all():
            raise ValueError("noise_realisations must be finite")
        if not np.isfinite(relative_floor) or relative_floor <= 0.0:
            raise ValueError("relative_floor must be finite and positive")
        if not np.isfinite(absolute_floor) or absolute_floor < 0.0:
            raise ValueError("absolute_floor must be finite and non-negative")
        values = noise.copy()
        mask: FloatArray | None = None
        if support_mask is not None:
            raw_mask = np.asarray(support_mask, dtype=np.float64)
            try:
                mask = np.broadcast_to(raw_mask, noise.shape[1:]).copy()
            except ValueError as error:
                raise ValueError("support_mask must broadcast to one noise field") from error
            if not np.isfinite(mask).all() or np.any((mask < 0.0) | (mask > 1.0)):
                raise ValueError("support_mask must be finite and lie between zero and one")
        if remove_spatial_mean:
            if mask is None:
                values -= values.mean(axis=(1, 2), keepdims=True)
            else:
                denominator = np.sum(mask, axis=(0, 1), keepdims=True)
                if np.any(denominator <= 0.0):
                    raise ValueError("support_mask must retain every displacement component")
                means = np.sum(values * mask[None, ...], axis=(1, 2), keepdims=True)
                values -= means / denominator[None, ...]
        if mask is not None:
            values *= mask[None, ...]
        transformed = np.fft.fftn(values, axes=(1, 2), norm="ortho")
        psd = np.mean(np.abs(transformed) ** 2, axis=0)
        opposite_x = _opposite_frequency_indices(psd.shape[0])
        opposite_y = _opposite_frequency_indices(psd.shape[1])
        conjugate_partner = psd[np.ix_(opposite_x, opposite_y, np.arange(psd.shape[2]))]
        psd = 0.5 * (psd + conjugate_partner)
        positive = psd[psd > 0.0]
        if positive.size == 0:
            raise ValueError("noise realisations have zero spectral power")
        floor = max(float(absolute_floor), float(relative_floor * np.median(positive)))
        return cls(
            power_spectral_density=psd,
            spectral_floor=floor,
            support_mask=mask,
        )

    @classmethod
    def from_stationary_noise_field(
        cls,
        noise_field: ArrayLike,
        *,
        target_shape: tuple[int, int],
        sample_count: int = 64,
        seed: int = 20260814,
        relative_floor: float = 1.0e-6,
        absolute_floor: float = 0.0,
        remove_spatial_mean: bool = True,
        support_mask: ArrayLike | None = None,
    ) -> DICSpectralWhitener:
        """Estimate a target-grid whitener from random contiguous noise windows.

        This is the bridge for the single large repeat-frame P43 residual: its
        spatial stationarity is used to extract several same-size realisations
        without wrapping or interpolating the measured noise.
        """

        field = np.asarray(noise_field, dtype=np.float64)
        if field.ndim != 3 or field.shape[-1] < 1 or not np.isfinite(field).all():
            raise ValueError("noise_field must have finite shape (nx, ny, components)")
        nx, ny = (int(value) for value in target_shape)
        if nx < 1 or ny < 1 or nx > field.shape[0] or ny > field.shape[1]:
            raise ValueError("target_shape must be positive and fit inside noise_field")
        if sample_count < 2:
            raise ValueError("sample_count must be at least two")
        rng = np.random.default_rng(seed)
        origins_x = rng.integers(0, field.shape[0] - nx + 1, size=sample_count)
        origins_y = rng.integers(0, field.shape[1] - ny + 1, size=sample_count)
        realisations = np.empty((sample_count, nx, ny, field.shape[-1]))
        for index, (origin_x, origin_y) in enumerate(
            zip(origins_x, origins_y, strict=True)
        ):
            realisations[index] = field[
                origin_x : origin_x + nx,
                origin_y : origin_y + ny,
            ]
        return cls.from_noise_realisations(
            realisations,
            relative_floor=relative_floor,
            absolute_floor=absolute_floor,
            remove_spatial_mean=remove_spatial_mean,
            support_mask=support_mask,
        )

    @property
    def field_shape(self) -> tuple[int, int, int]:
        return self.power_spectral_density.shape

    @property
    def spectral_weights(self) -> FloatArray:
        return 1.0 / np.sqrt(
            np.maximum(self.power_spectral_density, self.spectral_floor)
        )

    def _field(self, values: ArrayLike) -> FloatArray:
        field = np.asarray(values, dtype=np.float64)
        if field.shape != self.field_shape:
            raise ValueError(f"field must have shape {self.field_shape}")
        if not np.isfinite(field).all():
            raise ValueError("field must be finite")
        return field

    def apply(self, values: ArrayLike) -> FloatArray:
        """Apply ``C_D^{-1/2}`` to a displacement residual."""

        field = self._field(values)
        if self.support_mask is not None:
            field = field * self.support_mask
        transformed = np.fft.fftn(field, axes=(0, 1), norm="ortho")
        whitened = np.fft.ifftn(
            transformed * self.spectral_weights,
            axes=(0, 1),
            norm="ortho",
        )
        return np.asarray(whitened.real, dtype=np.float64)

    def adjoint(self, values: ArrayLike) -> FloatArray:
        """Apply the exact Euclidean adjoint, including the support mask."""

        field = self._field(values)
        transformed = np.fft.fftn(field, axes=(0, 1), norm="ortho")
        adjoint = np.fft.ifftn(
            transformed * self.spectral_weights,
            axes=(0, 1),
            norm="ortho",
        ).real
        if self.support_mask is not None:
            adjoint *= self.support_mask
        return np.asarray(adjoint, dtype=np.float64)

    def normal_action(self, values: ArrayLike) -> FloatArray:
        """Apply ``C_D^{-1}`` without assembling it."""

        return self.adjoint(self.apply(values))

    def quadratic_misfit(self, residual: ArrayLike) -> float:
        whitened = self.apply(residual)
        return 0.5 * float(np.vdot(whitened, whitened).real)
