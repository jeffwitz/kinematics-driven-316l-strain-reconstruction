"""Multi-criterion decision without an arbitrary weighted sum.

Collapsing disagreeing criteria into one score hides the disagreement and
manufactures a winner. Here a candidate is eliminated, dominated, non-dominated
or statistically indistinguishable, and "no candidate passes" is a permitted
outcome.

The worst band is carried as its own criterion: an excellent reproduction of one
band must never compensate the loss of another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class Sense(StrEnum):
    """Whether a criterion is better low or better high."""

    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"


@dataclass(frozen=True, slots=True)
class EliminationRule:
    """A mandatory criterion a candidate must satisfy to stay in the running."""

    criterion: str
    sense: Sense
    bound: float
    reason: str


@dataclass(frozen=True, slots=True)
class DecisionReport:
    """The full outcome, including the permitted "no winner" cases."""

    eliminated: dict[str, list[str]]
    survivors: list[str]
    dominated: dict[str, list[str]]
    non_dominated: list[str]
    conclusion: str
    criteria: dict[str, str] = field(default_factory=dict)


#: Registered conclusion vocabulary.
NO_CANDIDATE_PASSES = "no_candidate_passes_all_mandatory_criteria"
SINGLE_NON_DOMINATED = "one_non_dominated_candidate"
SEVERAL_NON_DOMINATED = "several_non_dominated_candidates"


def apply_elimination(
    candidates: dict[str, dict[str, float]],
    rules: tuple[EliminationRule, ...],
) -> tuple[list[str], dict[str, list[str]]]:
    """Return the survivors and, for each eliminated candidate, why.

    A missing or non-finite criterion eliminates: an unmeasured mandatory
    criterion is not a pass.
    """

    survivors: list[str] = []
    eliminated: dict[str, list[str]] = {}
    for label, values in candidates.items():
        failures: list[str] = []
        for rule in rules:
            value = values.get(rule.criterion)
            if value is None or not np.isfinite(value):
                failures.append(f"{rule.criterion}: not measured")
                continue
            fails = (
                value > rule.bound
                if rule.sense is Sense.LOWER_IS_BETTER
                else value < rule.bound
            )
            if fails:
                failures.append(
                    f"{rule.criterion}={value:.6g} fails {rule.sense.value} "
                    f"bound {rule.bound:.6g} ({rule.reason})"
                )
        if failures:
            eliminated[label] = failures
        else:
            survivors.append(label)
    return sorted(survivors), eliminated


def dominates(
    first: dict[str, float],
    second: dict[str, float],
    *,
    senses: dict[str, Sense],
) -> bool:
    """True when ``first`` is at least as good everywhere and better somewhere."""

    if not senses:
        raise ValueError("at least one criterion is required")
    strictly_better = False
    for criterion, sense in senses.items():
        a, b = first.get(criterion), second.get(criterion)
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            return False
        if sense is Sense.LOWER_IS_BETTER:
            if a > b:
                return False
            if a < b:
                strictly_better = True
        else:
            if a < b:
                return False
            if a > b:
                strictly_better = True
    return strictly_better


def pareto_front(
    candidates: dict[str, dict[str, float]],
    *,
    senses: dict[str, Sense],
) -> tuple[list[str], dict[str, list[str]]]:
    """Return the non-dominated set and, for each dominated one, its dominators."""

    labels = sorted(candidates)
    dominated: dict[str, list[str]] = {}
    for label in labels:
        by = [
            other
            for other in labels
            if other != label
            and dominates(candidates[other], candidates[label], senses=senses)
        ]
        if by:
            dominated[label] = by
    return [label for label in labels if label not in dominated], dominated


def worst_band_vector(
    per_band: dict[str, dict[str, float]],
    *,
    senses: dict[str, Sense],
) -> dict[str, float]:
    """Return, per criterion, the value of the band that performs worst.

    Deliberately a vector rather than a sum. Summing would let a band scored
    very well offset a band that was lost, which is exactly the compensation
    this criterion exists to prevent — and the worst band can differ from one
    criterion to the next.
    """

    if not per_band:
        raise ValueError("at least one band is required")
    worst: dict[str, float] = {}
    for criterion, sense in senses.items():
        values = [
            band[criterion]
            for band in per_band.values()
            if criterion in band and np.isfinite(band[criterion])
        ]
        if not values:
            worst[criterion] = float("nan")
        elif sense is Sense.LOWER_IS_BETTER:
            worst[criterion] = float(max(values))
        else:
            worst[criterion] = float(min(values))
    return worst


def decide(
    candidates: dict[str, dict[str, float]],
    *,
    senses: dict[str, Sense],
    rules: tuple[EliminationRule, ...] = (),
) -> DecisionReport:
    """Eliminate, then dominate, and refuse to force a total ranking."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    survivors, eliminated = apply_elimination(candidates, rules)
    if not survivors:
        return DecisionReport(
            eliminated=eliminated,
            survivors=[],
            dominated={},
            non_dominated=[],
            conclusion=NO_CANDIDATE_PASSES,
            criteria={k: v.value for k, v in senses.items()},
        )
    remaining = {label: candidates[label] for label in survivors}
    non_dominated, dominated = pareto_front(remaining, senses=senses)
    conclusion = (
        SINGLE_NON_DOMINATED if len(non_dominated) == 1 else SEVERAL_NON_DOMINATED
    )
    return DecisionReport(
        eliminated=eliminated,
        survivors=survivors,
        dominated=dominated,
        non_dominated=non_dominated,
        conclusion=conclusion,
        criteria={k: v.value for k, v in senses.items()},
    )
