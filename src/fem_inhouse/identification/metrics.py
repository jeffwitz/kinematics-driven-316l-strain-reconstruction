"""Multi-objective metrics for joint nonlocal parameter identification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isfinite, log
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import spearmanr

from fem_inhouse.postprocessing.metrics import (
    absolute_threshold_overlap_metrics,
    field_diffusivity_metrics,
    field_error_metrics,
    localization_overlap_metrics,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AmplitudeMetricConfig:
    """Configuration of the smooth quantile-amplitude objective."""

    quantiles: tuple[float, ...] = (0.50, 0.75, 0.90, 0.95, 0.99)
    quantile_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    standard_deviation_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.quantiles:
            raise ValueError("at least one amplitude quantile is required")
        if len(self.quantiles) != len(self.quantile_weights):
            raise ValueError("quantile_weights must match quantiles")
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in self.quantiles):
            raise ValueError("quantiles must be finite and lie in [0, 1]")
        if any(not isfinite(weight) or weight < 0.0 for weight in self.quantile_weights):
            raise ValueError("quantile weights must be finite and non-negative")
        if (
            not isfinite(self.standard_deviation_weight)
            or self.standard_deviation_weight < 0.0
        ):
            raise ValueError("standard_deviation_weight must be finite and non-negative")


DEFAULT_AMPLITUDE_CONFIG = AmplitudeMetricConfig()


def _valid_values(
    reference: ArrayLike,
    prediction: ArrayLike,
    mask: ArrayLike | None,
) -> tuple[FloatArray, FloatArray]:
    reference_array = np.asarray(reference, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    if reference_array.shape != prediction_array.shape:
        raise ValueError("reference and prediction must have the same shape")
    if reference_array.ndim != 2:
        raise ValueError("identification fields must be two-dimensional")
    valid = np.isfinite(reference_array) & np.isfinite(prediction_array)
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != reference_array.shape:
            raise ValueError("mask must have the same shape as the fields")
        valid &= mask_array
    if not valid.any():
        raise ValueError("no valid values remain for identification")
    return reference_array[valid], prediction_array[valid]


def amplitude_objective(
    reference: ArrayLike,
    prediction: ArrayLike,
    *,
    config: AmplitudeMetricConfig = DEFAULT_AMPLITUDE_CONFIG,
    mask: ArrayLike | None = None,
) -> dict[str, Any]:
    """Evaluate quantile log-ratios and the smooth amplitude objective."""

    reference_values, prediction_values = _valid_values(reference, prediction, mask)
    quantile_records: list[dict[str, float]] = []
    objective = 0.0
    regularization = (
        max(float(np.max(reference_values)), float(np.max(prediction_values)), 1.0)
        * np.finfo(np.float64).eps
    )
    for quantile, weight in zip(config.quantiles, config.quantile_weights, strict=True):
        reference_quantile = float(np.quantile(reference_values, quantile))
        prediction_quantile = float(np.quantile(prediction_values, quantile))
        if reference_quantile < 0.0 or prediction_quantile < 0.0:
            raise ValueError("amplitude log-ratios require non-negative quantiles")
        log_ratio = log(
            (prediction_quantile + regularization)
            / (reference_quantile + regularization)
        )
        objective += weight * log_ratio**2
        quantile_records.append(
            {
                "quantile": quantile,
                "weight": weight,
                "reference": reference_quantile,
                "prediction": prediction_quantile,
                "log_ratio": log_ratio,
            }
        )
    reference_std = float(np.std(reference_values))
    prediction_std = float(np.std(prediction_values))
    if reference_std <= 0.0 or prediction_std <= 0.0:
        raise ValueError("amplitude objective requires non-constant fields")
    std_log_ratio = log(prediction_std / reference_std)
    objective += config.standard_deviation_weight * std_log_ratio**2
    return {
        "objective": objective,
        "quantiles": quantile_records,
        "reference_standard_deviation": reference_std,
        "prediction_standard_deviation": prediction_std,
        "standard_deviation_log_ratio": std_log_ratio,
        "standard_deviation_weight": config.standard_deviation_weight,
        "zero_regularization": regularization,
    }


def radial_power_spectrum(
    field: ArrayLike,
    *,
    spacing_x_mm: float,
    spacing_y_mm: float,
    mask: ArrayLike | None = None,
) -> dict[str, list[float]]:
    """Return an annularly averaged two-dimensional power spectrum.

    A mask is applied as a zero-weight window after subtracting the valid
    mean. The normalization makes the spectrum useful for comparisons while
    preserving the frequency axis in cycles per millimetre.
    """

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("field must be two-dimensional")
    if not np.isfinite(spacing_x_mm) or spacing_x_mm <= 0.0:
        raise ValueError("spacing_x_mm must be finite and positive")
    if not np.isfinite(spacing_y_mm) or spacing_y_mm <= 0.0:
        raise ValueError("spacing_y_mm must be finite and positive")
    valid = np.isfinite(values)
    if mask is not None:
        mask_values = np.asarray(mask, dtype=bool)
        if mask_values.shape != values.shape:
            raise ValueError("mask must have the same shape as field")
        valid &= mask_values
    if not valid.any():
        raise ValueError("field contains no valid values")
    centred = np.zeros_like(values)
    centred[valid] = values[valid] - float(np.mean(values[valid]))
    spectrum_2d = np.abs(np.fft.rfft2(centred)) ** 2
    frequency_x = np.fft.fftfreq(values.shape[0], d=spacing_x_mm)
    frequency_y = np.fft.rfftfreq(values.shape[1], d=spacing_y_mm)
    radial_frequency = np.hypot(frequency_x[:, None], frequency_y[None, :])
    maximum_frequency = float(radial_frequency.max())
    bin_count = max(2, min(values.shape) // 2)
    bin_edges = np.linspace(0.0, maximum_frequency, bin_count + 1)
    bin_index = np.minimum(
        np.digitize(radial_frequency.ravel(), bin_edges) - 1,
        bin_count - 1,
    )
    power_sum = np.bincount(
        bin_index,
        weights=spectrum_2d.ravel(),
        minlength=bin_count,
    )
    counts = np.bincount(bin_index, minlength=bin_count)
    power = np.divide(
        power_sum,
        counts,
        out=np.zeros_like(power_sum, dtype=np.float64),
        where=counts > 0,
    )
    total_power = float(power.sum())
    if total_power > 0.0:
        power /= total_power
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return {
        "frequency_cycles_per_mm": centres.astype(float).tolist(),
        "normalized_power": power.astype(float).tolist(),
    }


def evaluate_identification_metrics(
    reference: ArrayLike,
    prediction: ArrayLike,
    *,
    spacing_x_mm: float,
    spacing_y_mm: float,
    mask: ArrayLike | None = None,
    amplitude_config: AmplitudeMetricConfig = DEFAULT_AMPLITUDE_CONFIG,
    top_fraction: float = 0.1,
    absolute_reference_quantile: float = 0.9,
) -> dict[str, Any]:
    """Evaluate independent global, amplitude, localization and spatial metrics."""

    reference_values, prediction_values = _valid_values(reference, prediction, mask)
    field_errors = field_error_metrics(reference, prediction, mask=mask)
    relative_overlap = localization_overlap_metrics(
        reference,
        prediction,
        top_fraction=top_fraction,
        mask=mask,
    )
    absolute_overlap = absolute_threshold_overlap_metrics(
        reference,
        prediction,
        reference_quantile=absolute_reference_quantile,
        mask=mask,
    )
    spearman = float(spearmanr(reference_values, prediction_values).statistic)
    prediction_diffusivity = field_diffusivity_metrics(
        prediction,
        raw_field=reference,
        spacing_x_mm=spacing_x_mm,
        spacing_y_mm=spacing_y_mm,
    )
    return {
        "global": {
            **asdict(field_errors),
            "spearman_correlation": spearman,
        },
        "amplitude": amplitude_objective(
            reference,
            prediction,
            config=amplitude_config,
            mask=mask,
        ),
        "localization_relative_top": asdict(relative_overlap),
        "localization_absolute_dic_quantile": asdict(absolute_overlap),
        "spatial": {
            "prediction_gradient_rms": prediction_diffusivity.gradient_rms,
            "prediction_total_variation": prediction_diffusivity.total_variation,
            "reference_spectrum": radial_power_spectrum(
                reference,
                spacing_x_mm=spacing_x_mm,
                spacing_y_mm=spacing_y_mm,
                mask=mask,
            ),
            "prediction_spectrum": radial_power_spectrum(
                prediction,
                spacing_x_mm=spacing_x_mm,
                spacing_y_mm=spacing_y_mm,
                mask=mask,
            ),
        },
    }


def peeq_diagnostic_metrics(
    peeq: ArrayLike,
    *,
    spacing_x_mm: float,
    spacing_y_mm: float,
    first_positive_plastic_strain: float,
    nonlocal_hardening_mpa: ArrayLike | None = None,
    quantiles: Sequence[float] = (0.90, 0.95, 0.99),
) -> dict[str, Any]:
    """Summarize PEEQ without presenting it as a DIC observable."""

    values = np.asarray(peeq, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("peeq must be a finite two-dimensional field")
    if first_positive_plastic_strain < 0.0 or not isfinite(
        first_positive_plastic_strain
    ):
        raise ValueError("first_positive_plastic_strain must be finite and non-negative")
    diffusivity = field_diffusivity_metrics(
        values,
        raw_field=values,
        spacing_x_mm=spacing_x_mm,
        spacing_y_mm=spacing_y_mm,
    )
    quantile_values = {
        f"q{round(quantile * 100):02d}": float(np.quantile(values, quantile))
        for quantile in quantiles
    }
    result: dict[str, Any] = {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "maximum": float(np.max(values)),
        "quantiles": quantile_values,
        "gradient_rms": diffusivity.gradient_rms,
        "total_variation": diffusivity.total_variation,
        "plastic_fraction": float(np.mean(values > first_positive_plastic_strain)),
    }
    if nonlocal_hardening_mpa is not None:
        hardening = np.asarray(nonlocal_hardening_mpa, dtype=np.float64)
        if hardening.shape != values.shape or not np.isfinite(hardening).all():
            raise ValueError("nonlocal_hardening_mpa must be finite and match PEEQ")
        result["nonlocal_hardening_l2_mpa"] = float(np.linalg.norm(hardening))
        result["nonlocal_hardening_energy_density_mpa2"] = float(
            np.mean(hardening**2)
        )
    return result
