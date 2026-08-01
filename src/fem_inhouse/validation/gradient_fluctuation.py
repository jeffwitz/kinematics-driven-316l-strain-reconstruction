"""Scalar criteria on displacement gradients, for fluctuation diagnostics.

Exploratory tooling. These criteria are **not** part of the v1 or v2 decision
sets and select nothing; see
`validation/gradient_fluctuation_criteria_diagnostic.md`.

Why gradients rather than the equivalent strain: EVM is one scalar invariant of
the symmetric part, so two fields can agree on it while disagreeing on which
component carries the strain. The full gradient keeps that information, and
splitting it into symmetric and antisymmetric parts separates deformation from
local rotation — a rigid rotation moves the gradient distance and must leave
the strain distance untouched.

The differentiation, the support, the mask and the edge handling are those of
the historical EVM operator: `np.gradient` with array axis 0 = canonical x,
then `cell_average` to element centres, then the core crop. Nothing here
introduces a second convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from fem_inhouse.postprocessing.kinematics import cell_average

FloatArray = NDArray[np.float64]

#: Registered scales of the multiscale analysis, in pixels. 49 is the measured
#: MTF-50 of the chain, so it separates what the optics resolve from what it
#: does not.
SCALES_PIXELS: tuple[int, ...] = (8, 16, 32, 49, 96)

#: The declared high-pass filter is ``H_s(f) = f - G_s * f`` with a Gaussian of
#: this standard deviation. Fixed once for every field and every scale: no
#: filter is ever tuned per candidate.
GAUSSIAN_SIGMA_FACTOR = 0.5


@dataclass(frozen=True, slots=True)
class GradientFields:
    """The gradient of one displacement field and its standard parts."""

    label: str
    gradient: FloatArray
    symmetric: FloatArray
    frobenius: FloatArray


def displacement_gradient(
    displacement_mm: NDArray[np.generic],
    *,
    spacing_x_mm: float,
    spacing_y_mm: float,
    average_to_cells: bool = True,
) -> FloatArray:
    """Return ``grad[..., i, j] = du_i / dx_j`` on the element centres.

    Array axis 0 is canonical x and axis 1 is canonical y, matching
    `strain_from_displacement`; the same `np.gradient` call is used, so the
    edge handling is one-sided at the grid boundary and central everywhere
    else. Averaging to cell centres reproduces the support of the historical
    EVM operator, which is what makes these criteria comparable with the
    archived EVM metrics.
    """

    values = np.asarray(displacement_mm, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("displacement_mm must have shape (rows, columns, 2)")
    if spacing_x_mm <= 0.0 or spacing_y_mm <= 0.0:
        raise ValueError("grid spacings must be positive")
    if not np.isfinite(values).all():
        raise ValueError("displacement_mm must contain only finite values")

    dux_dx, dux_dy = np.gradient(values[..., 0], spacing_x_mm, spacing_y_mm)
    duy_dx, duy_dy = np.gradient(values[..., 1], spacing_x_mm, spacing_y_mm)
    gradient = np.stack(
        (
            np.stack((dux_dx, dux_dy), axis=-1),
            np.stack((duy_dx, duy_dy), axis=-1),
        ),
        axis=-2,
    )
    if not average_to_cells:
        return np.ascontiguousarray(gradient)
    averaged = np.empty((gradient.shape[0] - 1, gradient.shape[1] - 1, 2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            averaged[..., i, j] = cell_average(gradient[..., i, j])
    return averaged


def symmetric_part(gradient: NDArray[np.generic]) -> FloatArray:
    """The small-strain tensor, ``(grad + grad^T) / 2``."""

    values = _as_tensor_field(gradient)
    return np.ascontiguousarray(0.5 * (values + np.swapaxes(values, -1, -2)))


def antisymmetric_part(gradient: NDArray[np.generic]) -> FloatArray:
    """The local rotation tensor, ``(grad - grad^T) / 2``."""

    values = _as_tensor_field(gradient)
    return np.ascontiguousarray(0.5 * (values - np.swapaxes(values, -1, -2)))


def frobenius_norm(tensor: NDArray[np.generic]) -> FloatArray:
    """Pointwise Frobenius norm of a tensor field."""

    values = _as_tensor_field(tensor)
    return np.ascontiguousarray(np.sqrt(np.sum(values**2, axis=(-2, -1))))


def remove_mean(
    tensor: NDArray[np.generic], *, valid_mask: NDArray[np.bool_] | None = None
) -> FloatArray:
    """Subtract the domain-average tensor, leaving the fluctuation."""

    values = _as_tensor_field(tensor)
    mask = _resolve_mask(valid_mask, values.shape[:2], values)
    mean = np.zeros((2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            mean[i, j] = float(np.mean(values[..., i, j][mask]))
    return np.ascontiguousarray(values - mean)


def highpass(field: NDArray[np.generic], *, scale_pixels: float) -> FloatArray:
    """``H_s(f) = f - G_s * f`` with the declared Gaussian.

    Applied componentwise to a tensor field, with the same sigma for every
    component, every scale and every candidate.
    """

    values = np.asarray(field, dtype=np.float64)
    if scale_pixels <= 0.0:
        raise ValueError("scale_pixels must be positive")
    if not np.isfinite(values).all():
        raise ValueError("field must contain only finite values")
    sigma = GAUSSIAN_SIGMA_FACTOR * float(scale_pixels)
    axes = (0, 1)
    smoothed = ndimage.gaussian_filter(values, sigma=sigma, axes=axes, mode="nearest")
    return np.ascontiguousarray(values - smoothed)


def relative_frobenius_distance(
    candidate: NDArray[np.generic],
    reference: NDArray[np.generic],
    *,
    valid_mask: NDArray[np.bool_] | None = None,
) -> float:
    """``||candidate - reference||_{L2,F} / ||reference||_{L2,F}``."""

    a = _as_tensor_field(candidate)
    b = _as_tensor_field(reference)
    if a.shape != b.shape:
        raise ValueError("both tensor fields must share the same support")
    mask = _resolve_mask(valid_mask, a.shape[:2], a, b)
    numerator = float(np.sqrt(np.sum(((a - b) ** 2).sum(axis=(-2, -1))[mask])))
    denominator = float(np.sqrt(np.sum((b**2).sum(axis=(-2, -1))[mask])))
    return numerator / denominator if denominator > 0.0 else float("nan")


def relative_l2_distance(
    candidate: NDArray[np.generic],
    reference: NDArray[np.generic],
    *,
    valid_mask: NDArray[np.bool_] | None = None,
) -> float:
    """Relative L2 distance between two scalar maps."""

    a = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("both maps must be two-dimensional and share a support")
    mask = _resolve_scalar_mask(valid_mask, a, b)
    numerator = float(np.sqrt(np.sum((a[mask] - b[mask]) ** 2)))
    denominator = float(np.sqrt(np.sum(b[mask] ** 2)))
    return numerator / denominator if denominator > 0.0 else float("nan")


def map_comparison(
    candidate: NDArray[np.generic],
    reference: NDArray[np.generic],
    *,
    valid_mask: NDArray[np.bool_] | None = None,
) -> dict[str, float]:
    """Relative distance, both correlations, bias and upper-quantile ratios."""

    a = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("both maps must be two-dimensional and share a support")
    mask = _resolve_scalar_mask(valid_mask, a, b)
    x = a[mask]
    y = b[mask]
    result = {
        "relative_l2": relative_l2_distance(a, b, valid_mask=mask),
        "pearson": _correlation(x, y),
        "spearman": _correlation(_ranks(x), _ranks(y)),
        "mean_bias": float(np.mean(x - y)),
    }
    for quantile in (0.90, 0.95):
        denominator = float(np.quantile(y, quantile))
        result[f"quantile_ratio_q{int(quantile * 100)}"] = (
            float(np.quantile(x, quantile)) / denominator if denominator != 0.0 else float("nan")
        )
    return result


def gradient_criteria(
    candidate: FloatArray,
    reference: FloatArray,
    *,
    valid_mask: NDArray[np.bool_] | None = None,
) -> dict[str, float]:
    """The four registered criteria of the specification, plus the sym variant."""

    candidate_symmetric = symmetric_part(candidate)
    reference_symmetric = symmetric_part(reference)
    result = {
        "J_gradient": relative_frobenius_distance(candidate, reference, valid_mask=valid_mask),
        "J_strain": relative_frobenius_distance(
            candidate_symmetric, reference_symmetric, valid_mask=valid_mask
        ),
        "J_norm_map": relative_l2_distance(
            frobenius_norm(candidate), frobenius_norm(reference), valid_mask=valid_mask
        ),
        "J_fluctuation": relative_frobenius_distance(
            remove_mean(candidate, valid_mask=valid_mask),
            remove_mean(reference, valid_mask=valid_mask),
            valid_mask=valid_mask,
        ),
        "J_fluctuation_symmetric": relative_frobenius_distance(
            remove_mean(candidate_symmetric, valid_mask=valid_mask),
            remove_mean(reference_symmetric, valid_mask=valid_mask),
            valid_mask=valid_mask,
        ),
    }
    result |= {
        f"norm_map_{key}": value
        for key, value in map_comparison(
            frobenius_norm(candidate), frobenius_norm(reference), valid_mask=valid_mask
        ).items()
    }
    return result


def multiscale_fluctuation(
    candidate: FloatArray,
    reference: FloatArray,
    *,
    scales_pixels: tuple[int, ...] = SCALES_PIXELS,
    valid_mask: NDArray[np.bool_] | None = None,
) -> dict[int, float]:
    """``J_fluct(s)`` on the high-passed symmetric part, scale by scale."""

    candidate_symmetric = symmetric_part(candidate)
    reference_symmetric = symmetric_part(reference)
    return {
        int(scale): relative_frobenius_distance(
            highpass(candidate_symmetric, scale_pixels=scale),
            highpass(reference_symmetric, scale_pixels=scale),
            valid_mask=valid_mask,
        )
        for scale in scales_pixels
    }


def highpass_energy_ratio(
    candidate: FloatArray,
    reference: FloatArray,
    *,
    scales_pixels: tuple[int, ...] = SCALES_PIXELS,
) -> dict[int, float]:
    """How much high-pass strain energy a field carries, relative to the DIC.

    This is what makes a fluctuation distance readable. ``J_fluct`` saturates
    near 1 for any candidate with no content at the scale considered, because
    the residual then reduces to the reference itself. A ratio well below 1
    means the candidate is smooth there, so its score of about 1 is the price
    of predicting nothing rather than evidence of agreement.
    """

    candidate_symmetric = symmetric_part(candidate)
    reference_symmetric = symmetric_part(reference)
    result: dict[int, float] = {}
    for scale in scales_pixels:
        denominator = float(np.sqrt(np.sum(highpass(reference_symmetric, scale_pixels=scale) ** 2)))
        numerator = float(np.sqrt(np.sum(highpass(candidate_symmetric, scale_pixels=scale) ** 2)))
        result[int(scale)] = numerator / denominator if denominator > 0.0 else float("nan")
    return result


def _as_tensor_field(tensor: NDArray[np.generic]) -> FloatArray:
    values = np.asarray(tensor, dtype=np.float64)
    if values.ndim != 4 or values.shape[-2:] != (2, 2):
        raise ValueError("tensor field must have shape (rows, columns, 2, 2)")
    return values


def _resolve_mask(
    valid_mask: NDArray[np.bool_] | None,
    shape: tuple[int, ...],
    *fields: FloatArray,
) -> NDArray[np.bool_]:
    """Points kept: declared valid and finite in every field involved."""

    mask = np.ones(shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if mask.shape != tuple(shape):
        raise ValueError("valid_mask must match the field support")
    for field in fields:
        mask = mask & np.isfinite(field).all(axis=(-2, -1))
    if not mask.any():
        raise ValueError("no valid point remains")
    return mask


def _resolve_scalar_mask(
    valid_mask: NDArray[np.bool_] | None,
    *fields: FloatArray,
) -> NDArray[np.bool_]:
    mask = (
        np.ones(fields[0].shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if mask.shape != fields[0].shape:
        raise ValueError("valid_mask must match the field support")
    for field in fields:
        mask = mask & np.isfinite(field)
    if not mask.any():
        raise ValueError("no valid point remains")
    return mask


def _correlation(x: FloatArray, y: FloatArray) -> float:
    x_centred = x - np.mean(x)
    y_centred = y - np.mean(y)
    denominator = float(np.sqrt(np.sum(x_centred**2) * np.sum(y_centred**2)))
    return float(np.sum(x_centred * y_centred) / denominator) if denominator > 0.0 else float("nan")


def _ranks(values: FloatArray) -> FloatArray:
    """Average ranks, so ties do not bias Spearman."""

    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if unique.size != values.size:
        sums = np.zeros(unique.size, dtype=np.float64)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks
