from __future__ import annotations

import pytest

from fem_inhouse.validation.pareto_decision import (
    NO_CANDIDATE_PASSES,
    SEVERAL_NON_DOMINATED,
    SINGLE_NON_DOMINATED,
    EliminationRule,
    Sense,
    apply_elimination,
    decide,
    dominates,
    pareto_front,
    worst_band_vector,
)

LOWER = Sense.LOWER_IS_BETTER
HIGHER = Sense.HIGHER_IS_BETTER
SENSES = {"error": LOWER, "skill": HIGHER}


def test_a_strictly_better_candidate_dominates() -> None:
    assert dominates({"error": 0.1, "skill": 0.9}, {"error": 0.2, "skill": 0.8}, senses=SENSES)


def test_equal_on_all_criteria_is_not_domination() -> None:
    same = {"error": 0.1, "skill": 0.9}

    assert not dominates(same, dict(same), senses=SENSES)


def test_a_trade_off_is_not_domination_either_way() -> None:
    a = {"error": 0.1, "skill": 0.5}  # better error
    b = {"error": 0.3, "skill": 0.9}  # better skill

    assert not dominates(a, b, senses=SENSES)
    assert not dominates(b, a, senses=SENSES)


def test_a_missing_criterion_prevents_domination() -> None:
    assert not dominates({"error": 0.1}, {"error": 0.2, "skill": 0.8}, senses=SENSES)


def test_the_front_keeps_every_non_dominated_candidate() -> None:
    candidates = {
        "a": {"error": 0.1, "skill": 0.5},
        "b": {"error": 0.3, "skill": 0.9},
        "c": {"error": 0.4, "skill": 0.4},  # worse than a and b on both
    }

    front, dominated = pareto_front(candidates, senses=SENSES)

    assert front == ["a", "b"]
    assert set(dominated["c"]) == {"a", "b"}


def test_elimination_reports_a_reason_per_failure() -> None:
    rules = (
        EliminationRule("error", LOWER, 0.25, "beyond DIC reproducibility"),
        EliminationRule("skill", HIGHER, 0.60, "band not reproduced"),
    )
    candidates = {
        "good": {"error": 0.1, "skill": 0.9},
        "bad": {"error": 0.4, "skill": 0.2},
    }

    survivors, eliminated = apply_elimination(candidates, rules)

    assert survivors == ["good"]
    assert len(eliminated["bad"]) == 2
    assert "beyond DIC reproducibility" in eliminated["bad"][0]


def test_an_unmeasured_mandatory_criterion_eliminates() -> None:
    rules = (EliminationRule("error", LOWER, 0.5, "mandatory"),)

    survivors, eliminated = apply_elimination({"x": {"skill": 0.9}}, rules)

    # Not measuring a mandatory criterion is not a pass.
    assert survivors == []
    assert "not measured" in eliminated["x"][0]


def test_a_non_finite_value_eliminates_too() -> None:
    rules = (EliminationRule("error", LOWER, 0.5, "mandatory"),)

    survivors, _ = apply_elimination({"x": {"error": float("nan")}}, rules)

    assert survivors == []


def test_the_worst_band_is_a_vector_not_a_sum() -> None:
    per_band = {
        "band1": {"error": 0.10, "skill": 0.95},
        "band2": {"error": 0.40, "skill": 0.30},
    }

    worst = worst_band_vector(per_band, senses=SENSES)

    # The worst value per criterion, kept separate. A sum would let band1's
    # excellent skill offset band2's loss.
    assert worst["error"] == pytest.approx(0.40)
    assert worst["skill"] == pytest.approx(0.30)


def test_the_worst_band_can_differ_between_criteria() -> None:
    per_band = {
        "band1": {"error": 0.50, "skill": 0.95},  # worst error
        "band2": {"error": 0.10, "skill": 0.20},  # worst skill
    }

    worst = worst_band_vector(per_band, senses=SENSES)

    assert worst["error"] == pytest.approx(0.50)
    assert worst["skill"] == pytest.approx(0.20)


def test_a_single_survivor_on_the_front_is_reported_as_such() -> None:
    candidates = {
        "a": {"error": 0.1, "skill": 0.9},
        "b": {"error": 0.2, "skill": 0.8},
    }

    report = decide(candidates, senses=SENSES)

    assert report.conclusion == SINGLE_NON_DOMINATED
    assert report.non_dominated == ["a"]
    assert report.dominated["b"] == ["a"]


def test_a_trade_off_refuses_to_name_a_winner() -> None:
    candidates = {
        "a": {"error": 0.1, "skill": 0.5},
        "b": {"error": 0.3, "skill": 0.9},
    }

    report = decide(candidates, senses=SENSES)

    assert report.conclusion == SEVERAL_NON_DOMINATED
    assert report.non_dominated == ["a", "b"]


def test_no_candidate_passing_is_a_permitted_outcome() -> None:
    rules = (EliminationRule("error", LOWER, 0.05, "tight bound"),)
    candidates = {"a": {"error": 0.1, "skill": 0.9}, "b": {"error": 0.2, "skill": 0.8}}

    report = decide(candidates, senses=SENSES, rules=rules)

    assert report.conclusion == NO_CANDIDATE_PASSES
    assert report.non_dominated == []
    assert set(report.eliminated) == {"a", "b"}


def test_elimination_runs_before_domination() -> None:
    # "a" would dominate, but it fails a mandatory criterion and must not
    # appear on the front at all.
    rules = (EliminationRule("skill", HIGHER, 0.8, "band must be reproduced"),)
    candidates = {
        "a": {"error": 0.1, "skill": 0.5},
        "b": {"error": 0.3, "skill": 0.9},
    }

    report = decide(candidates, senses=SENSES, rules=rules)

    assert "a" in report.eliminated
    assert report.non_dominated == ["b"]


def test_the_report_records_the_criteria_it_used() -> None:
    report = decide({"a": {"error": 0.1, "skill": 0.9}}, senses=SENSES)

    assert report.criteria == {"error": "lower_is_better", "skill": "higher_is_better"}


def test_deciding_without_candidates_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        decide({}, senses=SENSES)
