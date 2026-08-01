from __future__ import annotations

import numpy as np

from fem_inhouse.validation.otsu_morphology import describe_morphology, otsu_threshold
from fem_inhouse.workflows.compare_observed_evm_morphology import (
    CRITERION_SENSES,
    REGISTERED_SEVERITY,
    _kendall_tau,
    _morphology_criteria,
    run_gate_g2,
)

SHAPE = (120, 160)


def _two_bands(amplitude=1.0, minor=8.0, background=0.1) -> np.ndarray:
    rows = np.arange(SHAPE[0], dtype=np.float64)[:, None]
    field = np.full(SHAPE, background)
    for offset in (-30.0, 30.0):
        distance = rows - SHAPE[0] / 2 - offset
        field = field + amplitude * np.exp(-0.5 * (distance / minor) ** 2)
    return np.ascontiguousarray(field)


def test_tau_is_one_when_the_criterion_follows_severity() -> None:
    assert _kendall_tau([4.0, 3.0, 2.0, 1.0], [4, 3, 2, 1]) == 1.0


def test_tau_is_minus_one_when_the_criterion_inverts_severity() -> None:
    assert _kendall_tau([1.0, 2.0, 3.0, 4.0], [4, 3, 2, 1]) == -1.0


def test_a_constant_criterion_is_insensitive_rather_than_wrong() -> None:
    # It must land on exactly zero: being blind to a defect is not the same as
    # ranking it backwards, and only the latter removes a criterion.
    assert _kendall_tau([2.0, 2.0, 2.0, 2.0], [4, 3, 2, 1]) == 0.0


def test_ties_in_severity_carry_no_information() -> None:
    # width_0p80 and width_1p20 share a severity rank; a criterion that
    # separates them must not be rewarded or punished for it.
    assert _kendall_tau([5.0, 1.0], [1, 1]) == 0.0


def test_non_finite_cases_drop_out_instead_of_poisoning_the_statistic() -> None:
    # band_removed leaves the minor-axis ratio undefined; that case must not
    # decide the verdict for the whole criterion.
    assert _kendall_tau([float("nan"), 2.0, 1.0], [4, 3, 2]) == 1.0


def test_the_registered_severity_only_places_the_documented_cases() -> None:
    # The preregistration places five levels. shift_1px, shift_4px and
    # amplitude_1p50 are reported but must stay out of the statistic.
    assert set(REGISTERED_SEVERITY) == {
        "band_removed",
        "band_spurious",
        "shift_16px",
        "width_0p80",
        "width_1p20",
        "amplitude_0p90",
    }
    assert REGISTERED_SEVERITY["band_removed"] > REGISTERED_SEVERITY["band_spurious"]
    assert REGISTERED_SEVERITY["shift_16px"] > REGISTERED_SEVERITY["amplitude_0p90"]


def test_gate_g2_passes_only_on_a_perfect_reference() -> None:
    perfect = {
        "criteria": {
            "object_count_error": 0.0,
            "log_minor_axis_ratio": 0.0,
            "eccentricity_error": 0.0,
            "worst_centreline_error": 0.0,
            "worst_mass_error": 0.0,
            "minimum_skilful_scale_q90": 1.0,
            "worst_detected_fraction": 1.0,
            "corridor_energy_fraction": float("nan"),
        }
    }

    assert run_gate_g2(perfect)["passed"]

    # This is the v1 defect amendment 2 exists for: an absolute centreline
    # offset is non-zero for the DIC against itself.
    biased = {"criteria": dict(perfect["criteria"]) | {"worst_centreline_error": 3.2}}
    outcome = run_gate_g2(biased)
    assert not outcome["passed"]
    assert outcome["failed"] == ["worst_centreline_error"]


def test_a_defined_corridor_fraction_fails_g2() -> None:
    # A field compared with itself has no residual, so a finite fraction means
    # the residual was not actually zero.
    scored = {
        "criteria": {
            "object_count_error": 0.0,
            "log_minor_axis_ratio": 0.0,
            "eccentricity_error": 0.0,
            "worst_centreline_error": 0.0,
            "worst_mass_error": 0.0,
            "minimum_skilful_scale_q90": 1.0,
            "worst_detected_fraction": 1.0,
            "corridor_energy_fraction": 0.4,
        }
    }

    assert run_gate_g2(scored)["failed"] == ["corridor_energy_fraction"]


def test_the_object_count_is_satisfiable_by_a_speck() -> None:
    """Lock the defect the blind profile found on partition 43.

    The translated control matched the DIC's object count with a 413 px
    fragment against an 8340 px band. A raw count of connected components is
    not a morphology descriptor, and any v3 repair must break this test.
    """

    reference = _two_bands()
    threshold = otsu_threshold(reference)
    reference_morphology = describe_morphology(
        reference, threshold=threshold, label_name="ref"
    )
    assert reference_morphology.object_count == 2

    # One real band plus a speck barely over the registered 256 px floor.
    speck = np.full(SHAPE, 0.1)
    rows = np.arange(SHAPE[0], dtype=np.float64)[:, None]
    speck = speck + 1.0 * np.exp(-0.5 * ((rows - SHAPE[0] / 2 - 30.0) / 8.0) ** 2)
    speck[4:22, 4:22] = 1.0
    candidate = describe_morphology(
        np.ascontiguousarray(speck), threshold=threshold, label_name="speck"
    )

    criteria = _morphology_criteria(reference_morphology, candidate)

    assert candidate.object_count == 2
    assert criteria["object_count_error"] == 0.0
    # A perfect count while one "object" is an order of magnitude too small.
    areas = sorted(o.area_pixels for o in candidate.objects)
    assert areas[0] < 0.2 * sorted(o.area_pixels for o in reference_morphology.objects)[0]


def test_every_registered_criterion_has_a_sense() -> None:
    assert len(CRITERION_SENSES) == 7
    assert "object_count_error" in CRITERION_SENSES
    assert "eccentricity_error" in CRITERION_SENSES
