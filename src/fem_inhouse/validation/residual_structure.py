"""Structure of the signed residual between a reference and a candidate field.

A residual with the same RMSE can be white noise or a coherent dipole across
every band, and only the second says something about the model. These
diagnostics describe where the residual energy sits and how it is organised.

Radial and directional autocorrelation and the coherence lengths come from
`postprocessing.spatial_correlation`, which already implements the
mask-corrected estimator this project uses; they are not reimplemented here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

#: Heuristic labels for the shape of a residual across a band.
TOO_NARROW = "candidate_band_too_narrow"
TOO_WIDE = "candidate_band_too_wide"
SHIFTED = "candidate_band_shifted"
UNDER_AMPLITUDE = "candidate_amplitude_too_low"
OVER_AMPLITUDE = "candidate_amplitude_too_high"
UNSTRUCTURED = "no_dominant_structure"


@dataclass(frozen=True, slots=True)
class EnergyPartition:
    """Where the squared residual sits, relative to the band corridors."""

    total: float
    corridor: float
    background: float
    corridor_fraction: float


def signed_residual(
    reference: NDArray[np.generic],
    candidate: NDArray[np.generic],
    *,
    valid_mask: NDArray[np.bool_] | None = None,
) -> FloatArray:
    """Return ``reference - candidate``, with excluded pixels set to nan.

    The sign convention is fixed here once: positive means the candidate is
    below the reference, so a positive residual is missing strain.
    """

    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("the two fields must share a shape")
    residual = a - b
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != a.shape:
            raise ValueError("valid_mask must match the field shape")
        residual = np.where(mask, residual, np.nan)
    return np.ascontiguousarray(residual)


def energy_partition(
    residual: NDArray[np.generic],
    *,
    corridor: NDArray[np.bool_],
) -> EnergyPartition:
    """Split the squared residual between the band corridors and the rest."""

    values = np.asarray(residual, dtype=np.float64)
    mask = np.asarray(corridor, dtype=bool)
    if mask.shape != values.shape:
        raise ValueError("corridor must match the residual shape")
    finite = np.isfinite(values)
    squared = np.where(finite, values**2, 0.0)
    total = float(np.sum(squared))
    inside = float(np.sum(squared[mask & finite]))
    outside = total - inside
    return EnergyPartition(
        total=total,
        corridor=inside,
        background=outside,
        corridor_fraction=float(inside / total) if total > 0.0 else float("nan"),
    )


def radial_power_spectrum(
    residual: NDArray[np.generic],
    *,
    maximum_bins: int = 64,
) -> tuple[FloatArray, FloatArray]:
    """Return radially binned power against spatial frequency in cycles/pixel."""

    values = np.asarray(residual, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("residual must be two-dimensional")
    filled = np.where(np.isfinite(values), values, 0.0)
    filled = filled - float(np.mean(filled))
    power = np.abs(np.fft.fftshift(np.fft.fft2(filled))) ** 2
    rows, columns = values.shape
    fx = np.fft.fftshift(np.fft.fftfreq(rows))[:, None]
    fy = np.fft.fftshift(np.fft.fftfreq(columns))[None, :]
    radius = np.sqrt(fx**2 + fy**2)
    edges = np.linspace(0.0, float(radius.max()), maximum_bins + 1)
    index = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, maximum_bins - 1)
    counts = np.bincount(index, minlength=maximum_bins)
    sums = np.bincount(index, weights=power.ravel(), minlength=maximum_bins)
    spectrum = np.divide(
        sums, counts, out=np.full(maximum_bins, np.nan), where=counts > 0
    )
    return np.asarray(0.5 * (edges[:-1] + edges[1:])), np.asarray(spectrum)


def directional_variogram(
    residual: NDArray[np.generic],
    *,
    axis: int,
    maximum_lag_pixels: int = 64,
) -> tuple[FloatArray, FloatArray]:
    """Return the semivariogram along one array axis.

    A residual that is organised along the bands and a residual that is
    organised across them have different variograms even at equal RMSE.
    """

    values = np.asarray(residual, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("residual must be two-dimensional")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    if maximum_lag_pixels < 1:
        raise ValueError("maximum_lag_pixels must be positive")
    limit = min(maximum_lag_pixels, values.shape[axis] - 1)
    lags = np.arange(1, limit + 1, dtype=np.float64)
    gamma = np.empty(limit, dtype=np.float64)
    for position, lag in enumerate(range(1, limit + 1)):
        if axis == 0:
            difference = values[lag:, :] - values[:-lag, :]
        else:
            difference = values[:, lag:] - values[:, :-lag]
        finite = np.isfinite(difference)
        gamma[position] = (
            0.5 * float(np.mean(difference[finite] ** 2)) if finite.any() else np.nan
        )
    return lags, gamma


def residual_associations(
    residual: NDArray[np.generic],
    reference: NDArray[np.generic],
    *,
    spacing_pixels: float = 1.0,
) -> dict[str, float]:
    """Correlate the residual with the reference and with its derivatives.

    A residual correlated with the reference is an amplitude error. A residual
    correlated with a **signed** directional derivative is a placement error,
    because shifting a field by a small delta gives a residual close to
    ``delta * df/dx``.

    The gradient **magnitude** is reported too, since it says where the residual
    sits relative to the steep parts of the field, but it cannot detect a
    displacement on its own: a shift residual is antisymmetric across the band
    while the magnitude is symmetric and positive, so their correlation cancels
    to zero regardless of the shift.
    """

    r = np.asarray(residual, dtype=np.float64)
    f = np.asarray(reference, dtype=np.float64)
    if r.shape != f.shape:
        raise ValueError("residual and reference must share a shape")
    if not np.isfinite(spacing_pixels) or spacing_pixels <= 0.0:
        raise ValueError("spacing_pixels must be finite and positive")
    gx, gy = np.gradient(f, spacing_pixels)
    magnitude = np.sqrt(gx**2 + gy**2)
    finite = np.isfinite(r) & np.isfinite(f)

    def _pearson(a: FloatArray, b: FloatArray) -> float:
        if a.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "with_reference": _pearson(r[finite], f[finite]),
        "with_reference_gradient_magnitude": _pearson(r[finite], magnitude[finite]),
        "with_reference_derivative_x": _pearson(r[finite], gx[finite]),
        "with_reference_derivative_y": _pearson(r[finite], gy[finite]),
    }


def classify_residual(
    residual: NDArray[np.generic],
    *,
    corridor: NDArray[np.bool_],
    flanks: NDArray[np.bool_],
    relative_tolerance: float = 0.2,
) -> dict[str, object]:
    """Name the dominant shape of the residual around the bands.

    This is a **diagnostic heuristic, not a demonstrated result**. It reads the
    sign of the residual at the band centre against its flanks and reports the
    pattern that best matches, together with the numbers behind it so the
    reader can disagree.
    """

    values = np.asarray(residual, dtype=np.float64)
    inside = np.asarray(corridor, dtype=bool)
    outside = np.asarray(flanks, dtype=bool)
    if inside.shape != values.shape or outside.shape != values.shape:
        raise ValueError("corridor and flanks must match the residual shape")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must lie strictly between zero and one")

    finite = np.isfinite(values)
    centre_samples = values[inside & finite]
    flank_samples = values[outside & finite]
    if centre_samples.size == 0 or flank_samples.size == 0:
        return {"label": UNSTRUCTURED, "centre": float("nan"), "flank": float("nan")}

    centre = float(np.mean(centre_samples))
    flank = float(np.mean(flank_samples))
    scale = float(np.mean(np.abs(values[finite]))) or 1.0
    quiet = relative_tolerance * scale

    if abs(centre) <= quiet and abs(flank) <= quiet:
        label = UNSTRUCTURED
    elif centre > quiet and flank < -quiet:
        label = TOO_NARROW
    elif centre < -quiet and flank > quiet:
        label = TOO_WIDE
    elif centre > quiet and flank > quiet:
        label = UNDER_AMPLITUDE
    elif centre < -quiet and flank < -quiet:
        label = OVER_AMPLITUDE
    else:
        label = SHIFTED

    return {
        "label": label,
        "centre": centre,
        "flank": flank,
        "scale": scale,
        "interpretation": (
            "heuristic diagnostic from residual sign at band centre against "
            "flanks; not a demonstrated conclusion"
        ),
    }
