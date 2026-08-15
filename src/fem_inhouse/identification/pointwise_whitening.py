"""Whitening for a noise field that is not spatially uniform.

`DICSpectralWhitener` models one spectral density for the whole window. That is
right for the raw DIC noise and wrong for the strain of an elastic-closure
residual: the pointwise standard deviation of the latter varies by a factor of
six across the window, and a stationary model reproduces the global norm of the
noise it was fitted to while failing to return noise for noise on data it has
not seen.

This whitens in two stages. The first divides by the standard deviation
estimated at each point and component, which removes the non-uniformity a
stationary model cannot see. The second applies a spectral whitener to what is
left, which removes the spatial correlation the first stage ignores. Neither
stage alone passes the test; the composition is what is validated.

The acceptance criterion is the one that matters and the one a self-consistency
check cannot fake: **held-out** noise must come back with unit norm per
component *and* unit variance along directions fixed in advance. Samples used to
fit the whitener are never used to test it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.identification.dic_whitening import DICSpectralWhitener

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PointwiseFieldWhitener:
    """Pointwise scaling, optionally followed by a spectral stage."""

    inverse_deviation: FloatArray
    spectral: DICSpectralWhitener | None

    @classmethod
    def fit(
        cls,
        samples: ArrayLike,
        *,
        with_spectral_stage: bool = True,
        relative_floor: float = 1.0e-3,
    ) -> PointwiseFieldWhitener:
        """Estimate the whitener from noise realisations, shape `(n, ...)`.

        The floor is relative to the median deviation: a point whose estimated
        deviation is far below the rest would otherwise be amplified without
        bound by its own estimation error.
        """

        values = np.asarray(samples, dtype=np.float64)
        if values.ndim < 2:
            raise ValueError("samples must have shape (realisations, ...)")
        if values.shape[0] < 8:
            raise ValueError("a pointwise deviation needs several realisations")
        deviation = values.std(axis=0)
        floor = relative_floor * float(np.median(deviation))
        inverse = 1.0 / np.maximum(deviation, floor)
        scaled = values * inverse[None, ...]
        spectral = (
            DICSpectralWhitener.from_noise_realisations(scaled)
            if with_spectral_stage
            else None
        )
        return cls(inverse_deviation=inverse, spectral=spectral)

    def apply(self, values: ArrayLike) -> FloatArray:
        field = np.asarray(values, dtype=np.float64) * self.inverse_deviation
        if self.spectral is None:
            return field
        return np.asarray(self.spectral.apply(field), dtype=np.float64)

    def adjoint(self, values: ArrayLike) -> FloatArray:
        field = np.asarray(values, dtype=np.float64)
        if self.spectral is not None:
            field = np.asarray(self.spectral.adjoint(field), dtype=np.float64)
        return field * self.inverse_deviation


def null_test(
    whitener: PointwiseFieldWhitener,
    held_out: ArrayLike,
    *,
    directions: int = 64,
    seed: int = 20260816,
) -> dict[str, float]:
    """Return noise for noise? Norm and directional variance, on unseen samples.

    The norm alone is not enough: a whitener can carry the right global norm and
    still concentrate the noise along a few directions, which is exactly how a
    projection onto a mode acquires a spurious significance. The directions are
    drawn before the samples are seen.
    """

    samples = np.asarray(held_out, dtype=np.float64)
    whitened = np.asarray([whitener.apply(sample).reshape(-1) for sample in samples])
    components = whitened.shape[1]
    generator = np.random.default_rng(seed)
    probes = generator.normal(size=(components, directions))
    probes /= np.linalg.norm(probes, axis=0, keepdims=True)
    projections = whitened @ probes
    return {
        "norm_over_expected": float(
            np.mean(np.linalg.norm(whitened, axis=1)) / np.sqrt(components)
        ),
        "directional_standard_deviation": float(projections.std(axis=0).mean()),
        "worst_directional_standard_deviation": float(projections.std(axis=0).max()),
        "maximum_absolute_projection": float(np.abs(projections).max()),
    }
