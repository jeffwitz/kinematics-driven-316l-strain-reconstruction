"""Known defects, applied to a reference field to test the metrics themselves.

A metric that cannot rank a deliberately broken field is not usable for ranking
models. These generators produce defects whose severity is known in advance, so
a metric's response can be checked before it is trusted on real candidates.

Minimal set for lot 2: the perturbations that position, width, amplitude and
continuity are supposed to separate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PerturbedField:
    """A deliberately broken field and the defect it carries."""

    name: str
    field: FloatArray
    defect: str
    magnitude: float


def translate_field(
    field: NDArray[np.generic],
    *,
    rows: float,
    columns: float,
) -> FloatArray:
    """Shift a field by a sub-pixel amount, keeping its support."""

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("field must be two-dimensional")
    return np.ascontiguousarray(
        ndimage.shift(values, (rows, columns), order=1, mode="nearest")
    )


def scale_amplitude(field: NDArray[np.generic], *, factor: float) -> FloatArray:
    """Multiply the field, leaving its geometry untouched."""

    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("factor must be finite and positive")
    return np.ascontiguousarray(np.asarray(field, dtype=np.float64) * factor)


def change_band_width(
    field: NDArray[np.generic],
    *,
    factor: float,
    background: float = 0.0,
) -> FloatArray:
    """Widen or contract features while preserving their peak amplitude.

    Implemented as a Gaussian blur followed by renormalisation to the original
    peak, so the defect is width alone rather than width and amplitude
    together — which is the whole point of testing them separately.
    """

    values = np.asarray(field, dtype=np.float64)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("factor must be finite and positive")
    excess = values - background
    peak = float(np.max(excess))
    if factor > 1.0:
        widened = ndimage.gaussian_filter(excess, sigma=(factor - 1.0) * 2.0)
    else:
        sharpened = excess - ndimage.gaussian_filter(excess, sigma=(1.0 - factor) * 4.0)
        widened = np.clip(excess + sharpened, 0.0, None)
    new_peak = float(np.max(widened))
    if new_peak > 0.0 and peak > 0.0:
        widened = widened * (peak / new_peak)
    return np.ascontiguousarray(widened + background)


def remove_region(
    field: NDArray[np.generic],
    *,
    region: NDArray[np.bool_],
    background: float = 0.0,
) -> FloatArray:
    """Delete a band by flattening it to the background level."""

    values = np.asarray(field, dtype=np.float64).copy()
    mask = np.asarray(region, dtype=bool)
    if mask.shape != values.shape:
        raise ValueError("region must match the field shape")
    values[mask] = background
    return np.ascontiguousarray(values)


def interrupt_region(
    field: NDArray[np.generic],
    *,
    region: NDArray[np.bool_],
    fraction: float = 0.3,
    background: float = 0.0,
) -> FloatArray:
    """Blank a contiguous fraction of a band along its longer extent."""

    values = np.asarray(field, dtype=np.float64).copy()
    mask = np.asarray(region, dtype=bool)
    if mask.shape != values.shape:
        raise ValueError("region must match the field shape")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie strictly between zero and one")
    rows, columns = np.where(mask)
    if rows.size == 0:
        raise ValueError("region is empty")
    axis_rows = rows.max() - rows.min() >= columns.max() - columns.min()
    along = rows if axis_rows else columns
    start = along.min()
    span = round(fraction * float(along.max() - start + 1))
    cut = (along >= start) & (along < start + span)
    values[rows[cut], columns[cut]] = background
    return np.ascontiguousarray(values)


def add_spurious_band(
    field: NDArray[np.generic],
    *,
    centre: tuple[float, float],
    orientation_degrees: float,
    amplitude: float,
    half_width_pixels: float = 4.0,
) -> FloatArray:
    """Add a band the reference does not contain."""

    values = np.asarray(field, dtype=np.float64)
    rows = np.arange(values.shape[0], dtype=np.float64)[:, None]
    columns = np.arange(values.shape[1], dtype=np.float64)[None, :]
    angle = np.radians(orientation_degrees)
    distance = np.abs(
        -np.sin(angle) * (rows - centre[0]) + np.cos(angle) * (columns - centre[1])
    )
    return np.ascontiguousarray(
        values + amplitude * np.exp(-0.5 * (distance / half_width_pixels) ** 2)
    )


def standard_cases(
    reference: NDArray[np.generic],
    *,
    band_region: NDArray[np.bool_] | None = None,
) -> list[PerturbedField]:
    """Build the minimal falsification set from a reference field.

    Severity order registered by the specification, most severe first: a missing
    band, a spurious band, a shift comparable with the band width, a 20 % width
    error, a 10 % amplitude error.
    """

    values = np.asarray(reference, dtype=np.float64)
    cases = [
        PerturbedField("shift_1px", translate_field(values, rows=1.0, columns=0.0),
                       "position", 1.0),
        PerturbedField("shift_4px", translate_field(values, rows=4.0, columns=0.0),
                       "position", 4.0),
        PerturbedField("shift_16px", translate_field(values, rows=16.0, columns=0.0),
                       "position", 16.0),
        PerturbedField("amplitude_0p90", scale_amplitude(values, factor=0.90),
                       "amplitude", 0.10),
        PerturbedField("amplitude_1p50", scale_amplitude(values, factor=1.50),
                       "amplitude", 0.50),
        PerturbedField("width_1p20", change_band_width(values, factor=1.20),
                       "width", 0.20),
        PerturbedField("width_0p80", change_band_width(values, factor=0.80),
                       "width", 0.20),
    ]
    if band_region is not None:
        cases.append(
            PerturbedField("band_removed", remove_region(values, region=band_region),
                           "missing_band", 1.0)
        )
        cases.append(
            PerturbedField(
                "band_interrupted",
                interrupt_region(values, region=band_region, fraction=0.3),
                "continuity",
                0.3,
            )
        )
    return cases
