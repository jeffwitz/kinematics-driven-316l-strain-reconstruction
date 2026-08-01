"""The four registered defects of the P43 (ell, alpha) selection campaign.

Protocol: `validation/p0043_small_parameter_matrix_preregistration.md`.

All four read one observable, the high-passed strain magnitude
``g_s = ||H_s(sym(grad u))||_F`` of `gradient_fluctuation`, at the principal
scale of 49 px, the measured MTF-50 of the chain.

`D_presence` exists because the Frobenius distances of the earlier diagnostic
could not reject a structureless field: a fluctuation distance saturates near 1
for a candidate with no content, so predicting nothing was cheaper than
predicting something in the wrong place. Presence is measured separately and
never merged into the others.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.validation.fractions_skill_score import fractions_skill_score
from fem_inhouse.validation.gradient_fluctuation import (
    frobenius_norm,
    highpass,
    symmetric_part,
)

FloatArray = NDArray[np.float64]

#: Principal scale, the measured MTF-50 of the chain. Selection happens here.
PRINCIPAL_SCALE_PIXELS = 49

#: Reported as sensitivities, never used to break a tie.
SENSITIVITY_SCALES_PIXELS: tuple[int, ...] = (32, 96)

#: Registered names, in the order they enter the Pareto front and the minimax.
DEFECT_NAMES: tuple[str, ...] = (
    "D_shape",
    "D_amplitude",
    "D_localisation",
    "D_presence",
)


@dataclass(frozen=True, slots=True)
class Defects:
    """The four defects of one candidate, all lower-is-better."""

    label: str
    scale_pixels: int
    shape: float
    amplitude: float
    localisation: float
    presence: float

    def as_dict(self) -> dict[str, float]:
        return {
            "D_shape": self.shape,
            "D_amplitude": self.amplitude,
            "D_localisation": self.localisation,
            "D_presence": self.presence,
        }


def fluctuation_magnitude(
    gradient: NDArray[np.generic],
    *,
    scale_pixels: int,
) -> FloatArray:
    """``g_s = ||H_s(sym(grad u))||_F``, the scalar fluctuation field."""

    return frobenius_norm(highpass(symmetric_part(gradient), scale_pixels=scale_pixels))


def shape_defect(
    candidate: FloatArray,
    reference: FloatArray,
    *,
    valid_mask: NDArray[np.bool_] | None = None,
) -> float:
    """``1 - pearson``: are the fluctuations in the same places."""

    x, y = _paired(candidate, reference, valid_mask)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denominator <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum(x * y) / denominator)


def amplitude_defect(
    candidate: FloatArray,
    reference: FloatArray,
    *,
    quantile: float = 0.95,
    valid_mask: NDArray[np.bool_] | None = None,
) -> float:
    """``abs(log)`` of the upper-quantile ratio, so both errors cost the same."""

    x, y = _paired(candidate, reference, valid_mask)
    numerator = float(np.quantile(x, quantile))
    denominator = float(np.quantile(y, quantile))
    if not (numerator > 0.0 and denominator > 0.0):
        return float("nan")
    return abs(float(np.log(numerator / denominator)))


def localisation_defect(
    candidate: FloatArray,
    reference: FloatArray,
    *,
    scale_pixels: int = PRINCIPAL_SCALE_PIXELS,
    threshold_quantile: float = 0.90,
    valid_mask: NDArray[np.bool_] | None = None,
) -> float:
    """``1 - FSS`` at the DIC's own upper quantile, threshold frozen there.

    The threshold comes from the reference and is applied unchanged to the
    candidate, so no candidate can move the boundary it is judged against.
    """

    reference_values = np.asarray(reference, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    mask = _mask(valid_mask, reference_values, candidate_values)
    threshold = float(np.quantile(reference_values[mask], threshold_quantile))
    score = fractions_skill_score(
        reference_values >= threshold,
        candidate_values >= threshold,
        scale_pixels=scale_pixels,
        valid_mask=mask,
    )
    return 1.0 - score


def presence_defect(
    candidate: FloatArray,
    reference: FloatArray,
    *,
    valid_mask: NDArray[np.bool_] | None = None,
) -> float:
    """``abs(log R)`` with ``R`` the ratio of high-pass strain energies.

    The guard rail. A field that suppresses the fluctuations and one that
    exaggerates them are both penalised, and a smooth field can no longer buy a
    good score by predicting nothing.
    """

    x, y = _paired(candidate, reference, valid_mask)
    numerator = float(np.sum(x**2))
    denominator = float(np.sum(y**2))
    if not (numerator > 0.0 and denominator > 0.0):
        return float("inf") if denominator > 0.0 else float("nan")
    return abs(float(np.log(numerator / denominator)))


def energy_ratio(
    candidate: FloatArray,
    reference: FloatArray,
    *,
    valid_mask: NDArray[np.bool_] | None = None,
) -> float:
    """``R`` itself, reported beside its defect so the sign stays visible."""

    x, y = _paired(candidate, reference, valid_mask)
    denominator = float(np.sum(y**2))
    return float(np.sum(x**2)) / denominator if denominator > 0.0 else float("nan")


def evaluate(
    candidate_gradient: NDArray[np.generic],
    reference_gradient: NDArray[np.generic],
    *,
    label: str,
    scale_pixels: int = PRINCIPAL_SCALE_PIXELS,
    valid_mask: NDArray[np.bool_] | None = None,
) -> Defects:
    """Score one candidate gradient field against the DIC at one scale."""

    candidate = fluctuation_magnitude(candidate_gradient, scale_pixels=scale_pixels)
    reference = fluctuation_magnitude(reference_gradient, scale_pixels=scale_pixels)
    return Defects(
        label=label,
        scale_pixels=scale_pixels,
        shape=shape_defect(candidate, reference, valid_mask=valid_mask),
        amplitude=amplitude_defect(candidate, reference, valid_mask=valid_mask),
        localisation=localisation_defect(
            candidate, reference, scale_pixels=scale_pixels, valid_mask=valid_mask
        ),
        presence=presence_defect(candidate, reference, valid_mask=valid_mask),
    )


def normalise(
    defects: dict[str, float],
    *,
    self_defects: dict[str, float],
    null_defects: dict[str, float],
) -> dict[str, float]:
    """``Z = (D - D_self) / (D_null - D_self)``, per indicator.

    `D_self` is the measurement floor and `D_null` the best negative control,
    both declared per indicator. A degenerate span gives `nan` rather than a
    division by something near zero.
    """

    normalised: dict[str, float] = {}
    for name in DEFECT_NAMES:
        floor = self_defects.get(name, float("nan"))
        ceiling = null_defects.get(name, float("nan"))
        span = ceiling - floor
        value = defects.get(name, float("nan"))
        normalised[name] = (
            (value - floor) / span
            if np.isfinite(span) and abs(span) > 0.0 and np.isfinite(value)
            else float("nan")
        )
    return normalised


def minimax(normalised: dict[str, float]) -> float:
    """The worst normalised defect. No weighted sum is formed anywhere."""

    values = [normalised.get(name, float("nan")) for name in DEFECT_NAMES]
    if any(not np.isfinite(value) for value in values):
        return float("nan")
    return float(max(values))


def _mask(
    valid_mask: NDArray[np.bool_] | None,
    *fields: FloatArray,
) -> NDArray[np.bool_]:
    shape = fields[0].shape
    for field in fields:
        if field.shape != shape:
            raise ValueError("all fields must share the same support")
    mask = np.ones(shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError("valid_mask must match the field support")
    for field in fields:
        mask = mask & np.isfinite(field)
    if not mask.any():
        raise ValueError("no valid point remains")
    return mask


def _paired(
    candidate: FloatArray,
    reference: FloatArray,
    valid_mask: NDArray[np.bool_] | None,
) -> tuple[FloatArray, FloatArray]:
    a = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    mask = _mask(valid_mask, a, b)
    return a[mask], b[mask]
