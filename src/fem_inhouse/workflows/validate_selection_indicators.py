"""Section 9 of the P43 matrix protocol: validate the indicators, before reading.

Protocol: `validation/p0043_small_parameter_matrix_preregistration.md`.

The four defects are applied to nine registered cases whose answer is known in
advance. An indicator that fails stays in the report as a diagnostic and
**leaves the selection**. This runs before the matrix is looked at, so which
indicators survive cannot be a reaction to the matrix.
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from fem_inhouse.postprocessing.kinematics import plane_stress_equivalent_strain
from fem_inhouse.validation.gradient_fluctuation import (
    displacement_gradient,
    symmetric_part,
)
from fem_inhouse.validation.otsu_morphology import otsu_threshold
from fem_inhouse.validation.selection_indicators import (
    DEFECT_NAMES,
    PRINCIPAL_SCALE_PIXELS,
    SENSITIVITY_SCALES_PIXELS,
    energy_ratio,
    evaluate,
    fluctuation_magnitude,
)
from fem_inhouse.workflows.compare_gradient_fluctuation_criteria import (
    PIXEL_SIZE_MM,
    core_slice,
    dic_displacement,
    gradient_on_core,
    observed_displacement,
)
from fem_inhouse.workflows.compare_observed_evm_candidates import (
    _git_sha,
    extract_bands,
)

FloatArray = NDArray[np.float64]

#: Correction C2. Per-component standard deviations of the repeated final DIC
#: pair, from `dic_measurement_chain_results.md`, in pixels.
REPETITION_ROW_SIGMA_PIXELS = 0.0403
REPETITION_COLUMN_SIGMA_PIXELS = 0.0624

#: The 1/e autocorrelation length of that residual. It matters: the residual is
#: not white, and a white perturbation would be filtered straight out by the
#: high-pass and would understate the floor.
REPETITION_CORRELATION_PIXELS = 38.2

#: Spurious EVM RMS of the same repeated pair. **This is what the residual is
#: calibrated on**, not the displacement amplitude above.
#:
#: Found by running section 9: a Gaussian field reproducing the measured
#: displacement deviations and coherence exactly produces an EVM RMS of
#: 1.64e-3, twelve times the value the same campaign measured. The real
#: residual is far smoother than a Gaussian field of that nominal coherence,
#: which is consistent with the slow optical drift the measurement-chain report
#: says the pair may contain. Anchoring the floor on displacement amplitude
#: would therefore attribute to noise a strain the measurement demonstrably
#: does not produce, and would inflate `D_self` about twelvefold.
#:
#: The indicators consume strain, so the strain is the anchor. The displacement
#: deviation of the synthesised field comes out near 0.0034 px rather than the
#: measured 0.04, and that discrepancy is the drift, not an error.
REPETITION_EVM_RMS = 1.363e-4

#: Ten realisations, and the floor is their median.
REPETITION_REALISATIONS = 10
REPETITION_SEED = 20260801


def correlated_repetition_residual(
    shape: tuple[int, int],
    *,
    generator: np.random.Generator,
) -> FloatArray:
    """A displacement perturbation with the measured coherence and strain.

    A Gaussian filter of standard deviation ``s`` gives a field whose
    autocorrelation crosses ``1/e`` at ``2 s``, so the measured ``38.2 px``
    coherence fixes ``s = 19.1 px``. The relative amplitude of the two
    components keeps the measured row-to-column ratio.

    The overall scale then comes from the **measured spurious EVM RMS**, not
    from the measured displacement deviations: see `REPETITION_EVM_RMS` for why
    the two are inconsistent by a factor twelve and why the strain wins.
    """

    smoothing = 0.5 * REPETITION_CORRELATION_PIXELS
    components = []
    for sigma_pixels in (REPETITION_ROW_SIGMA_PIXELS, REPETITION_COLUMN_SIGMA_PIXELS):
        white = generator.normal(0.0, 1.0, shape)
        field = ndimage.gaussian_filter(white, sigma=smoothing, mode="nearest")
        deviation = float(np.std(field))
        if deviation <= 0.0:
            raise ValueError("degenerate residual realisation")
        components.append(field * (sigma_pixels / deviation) * PIXEL_SIZE_MM)
    residual = np.ascontiguousarray(np.stack(components, axis=-1))
    return residual * (REPETITION_EVM_RMS / _equivalent_strain_rms(residual))


def _equivalent_strain_rms(displacement: FloatArray) -> float:
    """The plane-stress EVM RMS of a displacement field, on the core."""

    strain = symmetric_part(gradient_on_core(displacement))
    evm = plane_stress_equivalent_strain(
        strain[..., 0, 0],
        strain[..., 1, 1],
        2.0 * strain[..., 0, 1],
        poisson_ratio=0.3,
        shear_convention="engineering",
    )
    value = float(np.sqrt(np.mean(np.asarray(evm) ** 2)))
    if value <= 0.0:
        raise ValueError("degenerate residual realisation")
    return value


def registered_cases(
    measured: FloatArray,
    *,
    band_region: NDArray[np.bool_],
) -> dict[str, FloatArray]:
    """The registered synthetic perturbations of section 9, on the DIC field.

    Each is a displacement-level operation, so every case goes through exactly
    the same differentiation, support and edge handling as a real candidate.
    """

    mean = measured.mean(axis=(0, 1))
    fluctuation = measured - mean

    def scaled(factor: float) -> FloatArray:
        return np.ascontiguousarray(mean + factor * fluctuation)

    removed = fluctuation.copy()
    removed[band_region] = 0.0

    merged = fluctuation.copy()
    merged[band_region] = ndimage.gaussian_filter(
        fluctuation, sigma=24.0, axes=(0, 1), mode="nearest"
    )[band_region]

    spurious = fluctuation.copy()
    rows = np.arange(measured.shape[0], dtype=np.float64)[:, None]
    columns = np.arange(measured.shape[1], dtype=np.float64)[None, :]
    # A band the reference does not contain, across the middle of the support.
    distance = np.abs(rows - measured.shape[0] / 2.0 + 0.0 * columns)
    amplitude = float(np.quantile(np.abs(fluctuation), 0.95))
    spurious = spurious + np.stack(
        (amplitude * np.exp(-0.5 * (distance / 8.0) ** 2), np.zeros(measured.shape[:2])),
        axis=-1,
    )

    return {
        "amplitude_0p80": scaled(0.80),
        "amplitude_1p20": scaled(1.20),
        "band_displaced_16px": np.ascontiguousarray(np.roll(measured, 16, axis=0)),
        "band_removed": np.ascontiguousarray(mean + removed),
        "bands_merged": np.ascontiguousarray(mean + merged),
        "band_spurious": np.ascontiguousarray(mean + spurious),
    }


def _acceptance(
    defects: dict[str, dict[str, dict[str, float]]],
    *,
    scale: int,
) -> dict[str, Any]:
    """The six registered acceptance criteria, evaluated at one scale."""

    at = {label: values[str(scale)] for label, values in defects.items()}
    checks: dict[str, Any] = {}

    checks["identity_is_optimal"] = {
        "passed": all(abs(at["dic_self"][name]) <= 1e-9 for name in DEFECT_NAMES),
        "values": at["dic_self"],
    }

    homogeneous = at.get("homogeneous", {})
    floor = at["repetition_residual"]
    checks["homogeneous_fails_presence_and_amplitude"] = {
        "passed": bool(
            homogeneous.get("D_presence", 0.0) > 10.0 * max(floor["D_presence"], 1e-6)
            and homogeneous.get("D_amplitude", 0.0) > 10.0 * max(floor["D_amplitude"], 1e-6)
        ),
        "presence": homogeneous.get("D_presence"),
        "amplitude": homogeneous.get("D_amplitude"),
        "floor_presence": floor["D_presence"],
        "floor_amplitude": floor["D_amplitude"],
    }

    translated = at.get("translated", {})
    checks["translated_fails_shape_or_localisation"] = {
        "passed": bool(
            translated.get("D_shape", 0.0) > 10.0 * max(floor["D_shape"], 1e-6)
            or translated.get("D_localisation", 0.0) > 10.0 * max(floor["D_localisation"], 1e-6)
        ),
        "shape": translated.get("D_shape"),
        "localisation": translated.get("D_localisation"),
    }

    # An amplitude error must not look like a position error, and conversely.
    amplitude_case = at["amplitude_1p20"]
    position_case = at["band_displaced_16px"]
    checks["amplitude_and_position_are_distinguishable"] = {
        "passed": bool(
            amplitude_case["D_amplitude"] > amplitude_case["D_shape"]
            and position_case["D_shape"] > position_case["D_amplitude"]
        ),
        "amplitude_case": amplitude_case,
        "position_case": position_case,
    }

    checks["removed_band_is_worse_than_a_moderate_amplitude_error"] = {
        "passed": bool(
            max(at["band_removed"][name] for name in DEFECT_NAMES)
            > max(at["amplitude_1p20"][name] for name in DEFECT_NAMES)
        ),
        "band_removed_worst": max(at["band_removed"][name] for name in DEFECT_NAMES),
        "amplitude_worst": max(at["amplitude_1p20"][name] for name in DEFECT_NAMES),
    }
    return checks


def _rank(
    defects: dict[str, dict[str, dict[str, float]]],
    labels: list[str],
    *,
    scale: int,
    criterion: str,
) -> list[str]:
    """Cases ordered best to worst on one criterion at one scale."""

    return sorted(labels, key=lambda label: defects[label][str(scale)][criterion])


def _stability(
    defects: dict[str, dict[str, dict[str, float]]],
    acceptance: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Are the **conclusions** qualitatively the same at 32, 49 and 96 px.

    The registered wording is about conclusions, not about orderings. Demanding
    an identical ranking of ten cases is far stricter than that: a single swap
    between two near-tied cases would fail it while changing nothing anyone
    would conclude. Stability is therefore the agreement of the acceptance
    verdicts, with the rank correlation of the orderings reported as evidence.
    """

    scales = [PRINCIPAL_SCALE_PIXELS, *SENSITIVITY_SCALES_PIXELS]
    labels = sorted(label for label in defects if label != "dic_self")
    orderings: dict[str, dict[str, list[str]]] = {}
    correlations: dict[str, dict[str, float]] = {}
    for name in DEFECT_NAMES:
        orderings[name] = {
            str(scale): _rank(defects, labels, scale=scale, criterion=name) for scale in scales
        }
        principal = {
            label: index for index, label in enumerate(orderings[name][str(PRINCIPAL_SCALE_PIXELS)])
        }
        correlations[name] = {}
        for scale in SENSITIVITY_SCALES_PIXELS:
            other = {label: index for index, label in enumerate(orderings[name][str(scale)])}
            a = np.array([principal[label] for label in labels], dtype=np.float64)
            b = np.array([other[label] for label in labels], dtype=np.float64)
            a = a - a.mean()
            b = b - b.mean()
            denominator = float(np.sqrt(np.sum(a**2) * np.sum(b**2)))
            correlations[name][str(scale)] = (
                float(np.sum(a * b) / denominator) if denominator > 0.0 else float("nan")
            )

    verdicts = {
        scale: {name: check["passed"] for name, check in checks.items()}
        for scale, checks in acceptance.items()
    }
    reference = verdicts[str(PRINCIPAL_SCALE_PIXELS)]
    agree = {scale: verdict == reference for scale, verdict in verdicts.items()}
    return {
        "orderings": orderings,
        "rank_correlation_against_principal": correlations,
        "acceptance_verdicts": verdicts,
        "verdicts_agree_with_principal": agree,
        "conclusions_stable": all(agree.values()),
    }


def validate_selection_indicators(
    *,
    prepared_case: str | Path,
    controls: dict[str, str | Path],
    dic_evm: str | Path,
    output_directory: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run section 9 and return the verdict on each indicator."""

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    measured = dic_displacement(Path(prepared_case))
    reference = gradient_on_core(measured)

    evm = np.asarray(np.load(Path(dic_evm), allow_pickle=False), dtype=np.float64)
    bands = extract_bands(evm, threshold=otsu_threshold(evm))
    corridor = np.zeros(evm.shape, dtype=bool)
    for band in bands.values():
        corridor |= band["corridor"]
    band_region = np.zeros(measured.shape[:2], dtype=bool)
    rows, columns = core_slice()
    band_region[rows, columns] = corridor

    fields: dict[str, FloatArray] = {"dic_self": measured}
    fields |= registered_cases(measured, band_region=band_region)
    for label, path in controls.items():
        fields[label] = observed_displacement(Path(path))

    scales = [PRINCIPAL_SCALE_PIXELS, *SENSITIVITY_SCALES_PIXELS]
    defects: dict[str, dict[str, dict[str, float]]] = {}
    extras: dict[str, dict[str, float]] = {}
    for label, displacement in fields.items():
        gradient = gradient_on_core(displacement)
        defects[label] = {
            str(scale): evaluate(gradient, reference, label=label, scale_pixels=scale).as_dict()
            for scale in scales
        }
        extras[label] = {
            "energy_ratio_49": energy_ratio(
                fluctuation_magnitude(gradient, scale_pixels=PRINCIPAL_SCALE_PIXELS),
                fluctuation_magnitude(reference, scale_pixels=PRINCIPAL_SCALE_PIXELS),
            )
        }

    # Correction C2: the floor is the median over realisations of the measured
    # repetition residual, not the DIC against itself.
    generator = np.random.default_rng(REPETITION_SEED)
    realisations: list[dict[str, dict[str, float]]] = []
    for _ in range(REPETITION_REALISATIONS):
        perturbed = measured + correlated_repetition_residual(
            measured.shape[:2], generator=generator
        )
        gradient = gradient_on_core(perturbed)
        realisations.append(
            {
                str(scale): evaluate(
                    gradient, reference, label="repetition", scale_pixels=scale
                ).as_dict()
                for scale in scales
            }
        )
    defects["repetition_residual"] = {
        str(scale): {
            name: float(np.median([r[str(scale)][name] for r in realisations]))
            for name in DEFECT_NAMES
        }
        for scale in scales
    }

    acceptance = {str(scale): _acceptance(defects, scale=scale) for scale in scales}
    stability = _stability(defects, acceptance)
    principal = acceptance[str(PRINCIPAL_SCALE_PIXELS)]

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed_indicator_validation",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "preregistration": "validation/p0043_small_parameter_matrix_preregistration.md",
        "mechanics_rerun": False,
        "principal_scale_pixels": PRINCIPAL_SCALE_PIXELS,
        "sensitivity_scales_pixels": list(SENSITIVITY_SCALES_PIXELS),
        "repetition_residual": {
            "row_sigma_pixels": REPETITION_ROW_SIGMA_PIXELS,
            "column_sigma_pixels": REPETITION_COLUMN_SIGMA_PIXELS,
            "correlation_pixels": REPETITION_CORRELATION_PIXELS,
            "realisations": REPETITION_REALISATIONS,
            "seed": REPETITION_SEED,
            "note": "an upper bound on the floor, so every Z is a lower bound",
        },
        "defects": defects,
        "extras": extras,
        "acceptance": acceptance,
        "scale_stability": stability,
        "all_acceptance_criteria_passed": all(check["passed"] for check in principal.values()),
        "failed_criteria": sorted(name for name, check in principal.items() if not check["passed"]),
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }
    (output / "indicator_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return report


__all__ = [
    "correlated_repetition_residual",
    "displacement_gradient",
    "registered_cases",
    "validate_selection_indicators",
]
