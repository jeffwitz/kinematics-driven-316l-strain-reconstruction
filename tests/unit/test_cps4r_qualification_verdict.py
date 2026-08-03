"""The registered criteria of the CPS4R qualification, as code.

The verdict function decides whether a scientific criterion passed. It once
reported a perfect score for a crystal case because crystal laws leave PEEQ at
zero and it compared an empty field against an empty field. That is the failure
mode worth locking: a criterion that cannot fail certifies nothing.
"""

from __future__ import annotations

import pytest

from scripts.qualify_reduced_integration import (
    DISPLACEMENT_RELATIVE_BOUND,
    HOURGLASS_RATIO_BOUND,
    PEEQ_RELATIVE_BOUND,
    verdict,
)


def _metrics(**overrides) -> dict:
    base = {
        "peeq_relative_l2": 1e-4,
        "displacement_relative_l2": 1e-5,
        "hourglass_energy_ratio": 1e-4,
        "cutbacks": 0,
        "reference_cutbacks": 0,
        "reference_peeq_max": 0.03,
        "stress_relative_l2": 1e-4,
        "reference_stress_norm_mpa": 1000.0,
    }
    base.update(overrides)
    return base


def test_a_clean_result_is_recommendable() -> None:
    assert verdict(_metrics())["recommendable"] is True


def test_a_crystal_result_is_judged_on_stress_not_on_empty_plastic_strain() -> None:
    """The regression. PEEQ is identically zero for a crystal law."""

    result = verdict(_metrics(reference_peeq_max=0.0, stress_relative_l2=0.05))

    assert result["A1_field"] == "stress"
    assert result["A1_value"] == 0.05
    assert result["A1_constitutive"] is False
    assert result["recommendable"] is False


def test_plastic_strain_is_preferred_when_the_reference_has_any() -> None:
    result = verdict(_metrics(peeq_relative_l2=0.2, stress_relative_l2=1e-9))

    assert result["A1_field"] == "peeq"
    assert result["A1_constitutive"] is False


def test_a_reference_with_neither_field_is_refused_rather_than_passed() -> None:
    with pytest.raises(ValueError, match="neither plastic strain nor stress"):
        verdict(_metrics(reference_peeq_max=0.0, reference_stress_norm_mpa=0.0))


def test_the_ratio_gate_passing_while_accuracy_fails_refutes_the_diagnostic() -> None:
    """Falsifier F3, as the campaign actually observed it."""

    result = verdict(
        _metrics(peeq_relative_l2=0.10, hourglass_energy_ratio=1.03e-3)
    )

    assert result["A4_hourglass_ratio"] is True
    assert result["A1_constitutive"] is False
    assert result["F3_diagnostic_conservative"] is False


def test_a_ratio_that_fails_its_own_gate_cannot_refute_the_diagnostic() -> None:
    """F3 is about the gate being permissive, not about it being triggered."""

    result = verdict(_metrics(peeq_relative_l2=0.10, hourglass_energy_ratio=0.5))

    assert result["A4_hourglass_ratio"] is False
    assert result["F3_diagnostic_conservative"] is True


def test_a_new_cutback_disqualifies() -> None:
    assert verdict(_metrics(cutbacks=2, reference_cutbacks=0))["A5_no_new_cutback"] is False
    assert verdict(_metrics(cutbacks=2, reference_cutbacks=3))["A5_no_new_cutback"] is True


@pytest.mark.parametrize(
    ("key", "bound"),
    [
        ("peeq_relative_l2", PEEQ_RELATIVE_BOUND),
        ("displacement_relative_l2", DISPLACEMENT_RELATIVE_BOUND),
    ],
)
def test_the_accuracy_bounds_are_inclusive(key: str, bound: float) -> None:
    """Exactly on the bound passes; a hair over does not."""

    assert verdict(_metrics(**{key: bound}))["recommendable"] is True
    assert verdict(_metrics(**{key: bound * 1.001}))["recommendable"] is False


def test_the_hourglass_gate_is_strict() -> None:
    assert verdict(_metrics(hourglass_energy_ratio=HOURGLASS_RATIO_BOUND))[
        "A4_hourglass_ratio"
    ] is False
