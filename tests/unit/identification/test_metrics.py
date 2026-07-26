from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.identification.metrics import (
    AmplitudeMetricConfig,
    amplitude_objective,
    evaluate_identification_metrics,
    peeq_diagnostic_metrics,
    radial_power_spectrum,
)


def test_amplitude_objective_is_zero_for_identical_positive_fields() -> None:
    field = np.linspace(0.1, 1.0, 100).reshape(10, 10)
    metrics = amplitude_objective(field, field)
    assert metrics["objective"] == pytest.approx(0.0, abs=1e-30)
    assert all(item["log_ratio"] == pytest.approx(0.0) for item in metrics["quantiles"])


def test_amplitude_objective_detects_uniform_scaling() -> None:
    field = np.linspace(0.1, 1.0, 100).reshape(10, 10)
    config = AmplitudeMetricConfig(
        quantiles=(0.5, 0.9),
        quantile_weights=(1.0, 2.0),
        standard_deviation_weight=3.0,
    )
    metrics = amplitude_objective(field, 2.0 * field, config=config)
    assert metrics["objective"] == pytest.approx(6.0 * np.log(2.0) ** 2)


def test_relative_and_absolute_localization_are_reported_separately() -> None:
    reference = np.linspace(0.01, 1.0, 100).reshape(10, 10)
    prediction = reference * 0.5
    metrics = evaluate_identification_metrics(
        reference,
        prediction,
        spacing_x_mm=0.001,
        spacing_y_mm=0.001,
    )
    assert metrics["localization_relative_top"]["intersection_over_union"] == 1.0
    assert (
        metrics["localization_absolute_dic_quantile"]["prediction_active_fraction"]
        < metrics["localization_absolute_dic_quantile"]["reference_active_fraction"]
    )


def test_radial_spectrum_identifies_sinusoidal_frequency() -> None:
    nx = ny = 64
    spacing = 0.01
    cycles_per_mm = 4.6875
    x = np.arange(nx)[:, None] * spacing
    field = np.sin(2.0 * np.pi * cycles_per_mm * x) * np.ones((1, ny))
    spectrum = radial_power_spectrum(
        field,
        spacing_x_mm=spacing,
        spacing_y_mm=spacing,
    )
    frequencies = np.asarray(spectrum["frequency_cycles_per_mm"])
    power = np.asarray(spectrum["normalized_power"])
    assert frequencies[np.argmax(power)] == pytest.approx(cycles_per_mm, abs=1.0)


def test_peeq_diagnostics_measure_plastic_fraction_and_hardening_norm() -> None:
    peeq = np.array([[0.0, 0.2], [0.1, 0.3]])
    hardening = np.ones_like(peeq) * 5.0
    metrics = peeq_diagnostic_metrics(
        peeq,
        spacing_x_mm=0.01,
        spacing_y_mm=0.02,
        first_positive_plastic_strain=0.05,
        nonlocal_hardening_mpa=hardening,
    )
    assert metrics["plastic_fraction"] == 0.75
    assert metrics["nonlocal_hardening_l2_mpa"] == 10.0
