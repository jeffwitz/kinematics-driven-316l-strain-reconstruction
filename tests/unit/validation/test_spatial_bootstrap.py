from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.validation.spatial_bootstrap import (
    INDISTINGUISHABLE,
    PROBABLY_BETTER,
    ROBUSTLY_BETTER,
    ROBUSTLY_WORSE,
    BootstrapDesign,
    block_indices,
    block_length_sensitivity,
    classify_probability,
    compare_pair,
    paired_band_bootstrap,
)


def _sections(band_count=2, sections=40, offset=0.0, seed=1):
    generator = np.random.default_rng(seed)
    return {
        f"band{b}": {
            "reference": generator.normal(1.0, 0.1, sections),
            "candidate": generator.normal(1.0 + offset, 0.1, sections),
        }
        for b in range(band_count)
    }


def test_block_indices_return_the_requested_count() -> None:
    generator = np.random.default_rng(0)

    drawn = block_indices(37, block_length=8, generator=generator)

    assert drawn.size == 37
    assert drawn.min() >= 0 and drawn.max() < 37


def test_blocks_keep_consecutive_sections_together() -> None:
    generator = np.random.default_rng(0)

    drawn = block_indices(50, block_length=10, generator=generator)

    # Within a block, indices advance by one modulo the section count.
    steps = (drawn[1:10] - drawn[0:9]) % 50
    assert set(steps.tolist()) == {1}


def test_a_block_longer_than_the_series_is_rejected() -> None:
    with pytest.raises(ValueError, match="between one and the section count"):
        block_indices(10, block_length=11, generator=np.random.default_rng(0))


def test_the_bootstrap_is_reproducible_from_its_seed() -> None:
    data = _sections()
    design = BootstrapDesign(block_length=5, draws=200, seed=42)

    first = paired_band_bootstrap(data, design=design)
    second = paired_band_bootstrap(data, design=design)

    np.testing.assert_array_equal(first["reference"], second["reference"])
    np.testing.assert_array_equal(first["candidate"], second["candidate"])


def test_a_different_seed_gives_a_different_draw() -> None:
    data = _sections()

    a = paired_band_bootstrap(data, design=BootstrapDesign(5, 200, 1))
    b = paired_band_bootstrap(data, design=BootstrapDesign(5, 200, 2))

    assert not np.array_equal(a["reference"], b["reference"])


def test_the_draw_is_paired_across_candidates() -> None:
    # Two candidates carrying identical values must give identical samples:
    # only a shared index set can produce that.
    values = np.random.default_rng(3).normal(0.0, 1.0, 30)
    data = {"band0": {"a": values, "b": values.copy()}}

    samples = paired_band_bootstrap(data, design=BootstrapDesign(4, 300, 7))

    np.testing.assert_allclose(samples["a"], samples["b"])


def test_bands_carry_equal_weight_regardless_of_length() -> None:
    # A long band at 0 and a short band at 10 must average to 5, not to 0.3.
    data = {
        "long": {"c": np.zeros(200)},
        "short": {"c": np.full(10, 10.0)},
    }

    samples = paired_band_bootstrap(data, design=BootstrapDesign(5, 200, 11))

    assert float(np.mean(samples["c"])) == pytest.approx(5.0, abs=1e-9)


def test_unequal_section_counts_between_candidates_are_rejected() -> None:
    data = {"band0": {"a": np.zeros(10), "b": np.zeros(11)}}

    with pytest.raises(ValueError, match="unequal section counts"):
        paired_band_bootstrap(data, design=BootstrapDesign(3, 10, 0))


def test_a_band_missing_a_candidate_is_rejected() -> None:
    data = {"band0": {"a": np.zeros(10)}, "band1": {"a": np.zeros(10), "b": np.zeros(10)}}

    with pytest.raises(ValueError, match="does not score every candidate"):
        paired_band_bootstrap(data, design=BootstrapDesign(3, 10, 0))


def test_a_clear_improvement_reads_as_robustly_better() -> None:
    data = _sections(offset=-0.5)  # candidate lower, and lower is better
    samples = paired_band_bootstrap(data, design=BootstrapDesign(5, 2000, 4))

    result = compare_pair(
        samples["candidate"], samples["reference"],
        metric="error", first="candidate", second="reference", lower_is_better=True,
    )

    assert result.decision == ROBUSTLY_BETTER
    assert result.probability_first_better > 0.95
    assert result.upper_95 < 0.0


def test_the_mirror_pair_reads_as_robustly_worse() -> None:
    data = _sections(offset=-0.5)
    samples = paired_band_bootstrap(data, design=BootstrapDesign(5, 2000, 4))

    result = compare_pair(
        samples["reference"], samples["candidate"],
        metric="error", first="reference", second="candidate", lower_is_better=True,
    )

    assert result.decision == ROBUSTLY_WORSE


def test_identical_candidates_read_as_indistinguishable() -> None:
    values = np.random.default_rng(9).normal(1.0, 0.1, 40)
    data = {"band0": {"a": values, "b": values.copy()}}
    samples = paired_band_bootstrap(data, design=BootstrapDesign(5, 1000, 6))

    result = compare_pair(
        samples["a"], samples["b"],
        metric="error", first="a", second="b", lower_is_better=True,
    )

    assert result.decision == INDISTINGUISHABLE
    assert result.median_difference == pytest.approx(0.0, abs=1e-12)


def test_the_sense_of_the_metric_is_honoured() -> None:
    data = _sections(offset=0.5)  # candidate higher
    samples = paired_band_bootstrap(data, design=BootstrapDesign(5, 2000, 4))

    as_error = compare_pair(
        samples["candidate"], samples["reference"],
        metric="m", first="candidate", second="reference", lower_is_better=True,
    )
    as_skill = compare_pair(
        samples["candidate"], samples["reference"],
        metric="m", first="candidate", second="reference", lower_is_better=False,
    )

    assert as_error.probability_first_better == pytest.approx(
        1.0 - as_skill.probability_first_better, abs=1e-12
    )


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.99, ROBUSTLY_BETTER),
        (0.90, PROBABLY_BETTER),
        (0.50, INDISTINGUISHABLE),
        (0.01, ROBUSTLY_WORSE),
    ],
)
def test_the_registered_thresholds_map_onto_the_vocabulary(probability, expected) -> None:
    assert classify_probability(probability) == expected


def test_non_monotone_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        classify_probability(0.5, thresholds=(0.2, 0.1, 0.8, 0.95))


def test_a_conclusion_can_be_checked_against_several_block_lengths() -> None:
    data = _sections(offset=-0.5)

    results = block_length_sensitivity(
        data, block_lengths=(2, 5, 10), draws=500, seed=3,
        metric="error", first="candidate", second="reference", lower_is_better=True,
    )

    assert len(results) == 3
    # A conclusion surviving only one block length would be about the scheme.
    assert {r.decision for r in results} == {ROBUSTLY_BETTER}


def test_ties_count_as_half_a_win_not_as_a_loss() -> None:
    # Half the draws tie, half favour the first candidate.
    first = np.array([0.0, 0.0, -1.0, -1.0])
    second = np.zeros(4)

    result = compare_pair(
        first, second, metric="m", first="a", second="b", lower_is_better=True
    )

    # Strict inequality would give 0.5 here only by luck; with all draws tied it
    # would give 0.0 and call two identical candidates robustly worse.
    assert result.probability_first_better == pytest.approx(0.75)


def test_all_ties_give_an_even_probability() -> None:
    zeros = np.zeros(100)

    result = compare_pair(
        zeros, zeros, metric="m", first="a", second="b", lower_is_better=True
    )

    assert result.probability_first_better == pytest.approx(0.5)
    assert result.decision == INDISTINGUISHABLE
