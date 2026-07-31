"""Paired block bootstrap over band sections.

Treating hundreds of thousands of pixels as independent degrees of freedom
would produce confidence intervals narrow enough to declare any difference
significant. Neighbouring pixels of a strain band are strongly correlated, so
the resampling unit here is a **block of consecutive sections**, never a pixel.

The resampling is **paired**: one draw selects a set of sections, and every
candidate is scored on that same set. Drawing independently per candidate would
compare separate noise realisations and inflate every difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]

#: Registered decision vocabulary, keyed by probability of superiority.
ROBUSTLY_BETTER = "robustly_better"
PROBABLY_BETTER = "probably_better"
INDISTINGUISHABLE = "indistinguishable"
PROBABLY_WORSE = "probably_worse"
ROBUSTLY_WORSE = "robustly_worse"

#: Registered thresholds. Fixed before any candidate comparison is read.
DECISION_THRESHOLDS: tuple[float, float, float, float] = (0.05, 0.20, 0.80, 0.95)

#: Nominal draw count from the specification.
DEFAULT_DRAWS = 10_000


@dataclass(frozen=True, slots=True)
class BootstrapDesign:
    """Everything needed to reproduce a resampling, recorded with the result."""

    block_length: int
    draws: int
    seed: int

    def as_dict(self) -> dict[str, int]:
        return {"block_length": self.block_length, "draws": self.draws, "seed": self.seed}


@dataclass(frozen=True, slots=True)
class PairwiseResult:
    """One metric, one ordered pair of candidates."""

    metric: str
    first: str
    second: str
    median_difference: float
    lower_95: float
    upper_95: float
    probability_first_better: float
    sign_change_fraction: float
    decision: str


def block_indices(
    section_count: int,
    *,
    block_length: int,
    generator: np.random.Generator,
) -> IntArray:
    """Draw one bootstrap replicate of section indices, in blocks.

    Blocks are circular, so every section has the same chance of appearing and
    the ends are not under-sampled.
    """

    if section_count < 1:
        raise ValueError("section_count must be positive")
    if block_length < 1 or block_length > section_count:
        raise ValueError("block_length must lie between one and the section count")
    blocks = int(np.ceil(section_count / block_length))
    starts = generator.integers(0, section_count, size=blocks)
    offsets = np.arange(block_length)
    drawn = (starts[:, None] + offsets[None, :]).ravel() % section_count
    return np.asarray(drawn[:section_count], dtype=np.int_)


def paired_band_bootstrap(
    per_band_sections: dict[str, dict[str, NDArray[np.generic]]],
    *,
    design: BootstrapDesign,
) -> dict[str, FloatArray]:
    """Bootstrap a per-candidate statistic, paired across candidates.

    ``per_band_sections`` maps band identifier to candidate label to the
    per-section metric values of that candidate on that band.

    Bands are resampled **separately** and then averaged with **equal weight**,
    so a long band does not dominate a short one merely by carrying more
    sections.
    """

    if not per_band_sections:
        raise ValueError("at least one band is required")
    labels = sorted({label for band in per_band_sections.values() for label in band})
    if not labels:
        raise ValueError("at least one candidate is required")

    counts: dict[str, int] = {}
    for band, candidates in per_band_sections.items():
        if set(candidates) != set(labels):
            raise ValueError(f"band {band!r} does not score every candidate")
        sizes = {np.asarray(v).size for v in candidates.values()}
        if len(sizes) != 1:
            raise ValueError(f"band {band!r} has unequal section counts across candidates")
        counts[band] = sizes.pop()
        if counts[band] < 1:
            raise ValueError(f"band {band!r} has no section")

    generator = np.random.default_rng(design.seed)
    samples = {label: np.empty(design.draws, dtype=np.float64) for label in labels}
    for draw in range(design.draws):
        # One index set per band per draw, shared by every candidate.
        selections = {
            band: block_indices(
                counts[band], block_length=min(design.block_length, counts[band]),
                generator=generator,
            )
            for band in per_band_sections
        }
        for label in labels:
            per_band = []
            for band, candidates in per_band_sections.items():
                values = np.asarray(candidates[label], dtype=np.float64)[selections[band]]
                finite = values[np.isfinite(values)]
                per_band.append(float(np.mean(finite)) if finite.size else np.nan)
            samples[label][draw] = float(np.nanmean(per_band)) if per_band else np.nan
    return samples


def classify_probability(
    probability: float,
    *,
    thresholds: tuple[float, float, float, float] = DECISION_THRESHOLDS,
) -> str:
    """Map a probability of superiority onto the registered vocabulary."""

    low, low_mid, high_mid, high = thresholds
    if not 0.0 < low < low_mid < high_mid < high < 1.0:
        raise ValueError("thresholds must be strictly increasing inside (0, 1)")
    if not np.isfinite(probability):
        return INDISTINGUISHABLE
    if probability > high:
        return ROBUSTLY_BETTER
    if probability > high_mid:
        return PROBABLY_BETTER
    if probability >= low_mid:
        return INDISTINGUISHABLE
    if probability >= low:
        return PROBABLY_WORSE
    return ROBUSTLY_WORSE


def compare_pair(
    first_samples: NDArray[np.generic],
    second_samples: NDArray[np.generic],
    *,
    metric: str,
    first: str,
    second: str,
    lower_is_better: bool,
    thresholds: tuple[float, float, float, float] = DECISION_THRESHOLDS,
) -> PairwiseResult:
    """Compare two candidates on paired bootstrap samples of one metric."""

    a = np.asarray(first_samples, dtype=np.float64)
    b = np.asarray(second_samples, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same number of draws")
    difference = a - b
    finite = difference[np.isfinite(difference)]
    if finite.size == 0:
        raise ValueError("no finite draw to compare")
    positive = float(np.count_nonzero(finite > 0.0)) / finite.size
    negative = float(np.count_nonzero(finite < 0.0)) / finite.size
    ties = float(np.count_nonzero(finite == 0.0)) / finite.size
    # Ties count as half a win each. Strict inequality would score two identical
    # candidates as a certain loss, which biases every comparison downwards and
    # is worst for metrics whose values are discrete.
    wins = negative if lower_is_better else positive
    probability = wins + 0.5 * ties
    return PairwiseResult(
        metric=metric,
        first=first,
        second=second,
        median_difference=float(np.median(finite)),
        lower_95=float(np.percentile(finite, 2.5)),
        upper_95=float(np.percentile(finite, 97.5)),
        probability_first_better=probability,
        sign_change_fraction=float(min(positive, negative)),
        # tie fraction is folded into the probability, see above
        decision=classify_probability(probability, thresholds=thresholds),
    )


def block_length_sensitivity(
    per_band_sections: dict[str, dict[str, NDArray[np.generic]]],
    *,
    block_lengths: tuple[int, ...],
    draws: int,
    seed: int,
    metric: str,
    first: str,
    second: str,
    lower_is_better: bool,
) -> list[PairwiseResult]:
    """Repeat one pairwise comparison across several block lengths.

    A conclusion that survives only one block length is a conclusion about the
    resampling scheme, not about the candidates.
    """

    if not block_lengths:
        raise ValueError("at least one block length is required")
    results = []
    for length in block_lengths:
        samples = paired_band_bootstrap(
            per_band_sections,
            design=BootstrapDesign(block_length=length, draws=draws, seed=seed),
        )
        results.append(
            compare_pair(
                samples[first],
                samples[second],
                metric=f"{metric}@block{length}",
                first=first,
                second=second,
                lower_is_better=lower_is_better,
            )
        )
    return results
