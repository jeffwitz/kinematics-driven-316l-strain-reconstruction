from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.validation.selection_indicators import DEFECT_NAMES
from fem_inhouse.workflows.select_p0043_parameters import (
    CONTROL_LABELS,
    PRINCIPAL_SCALE_PIXELS,
    MatrixPoint,
    _conclusion,
    _iso_achi,
    _null_defects,
    _zone,
)


def _point(alpha: float, ell: float, *, converged: bool = True) -> MatrixPoint:
    from pathlib import Path

    return MatrixPoint(
        label=f"a{alpha:g}-ell{ell:g}".replace(".", "p"),
        alpha=alpha,
        ell_um=ell,
        flow_path=Path("/nowhere"),
        converged=converged,
    )


def test_achi_is_the_degeneracy_direction() -> None:
    assert _point(2.0, 20.0).achi == pytest.approx(800.0)
    assert _point(0.5, 40.0).achi == pytest.approx(800.0)


def test_the_iso_achi_pairs_of_the_registered_grid_are_found() -> None:
    points = [
        _point(alpha, ell) for ell in (20.0, 40.0, 58.88, 90.0) for alpha in (0.5, 1.0, 2.0, 4.0)
    ]
    scores = {point.label: 1.0 for point in points}

    pairs = _iso_achi(points, scores)

    assert [pair["achi"] for pair in pairs] == [800.0, 1600.0]
    assert set(pairs[0]["members"]) == {"a2-ell20", "a0p5-ell40"}
    assert set(pairs[1]["members"]) == {"a4-ell20", "a1-ell40"}


def test_a_pair_with_a_non_converged_member_is_not_comparable() -> None:
    points = [_point(2.0, 20.0), _point(0.5, 40.0, converged=False)]
    scores = {"a2-ell20": 0.5}

    pair = _iso_achi(points, scores)[0]

    assert pair["comparable"] is False
    assert pair["separation"] is None


def test_the_null_defect_is_the_best_control_and_says_which() -> None:
    scale = str(PRINCIPAL_SCALE_PIXELS)
    defects = {
        "homogeneous": {
            scale: {"D_shape": 0.7, "D_amplitude": 2.0, "D_localisation": 1.0, "D_presence": 4.0}
        },
        "translated": {
            scale: {"D_shape": 0.9, "D_amplitude": 0.3, "D_localisation": 0.4, "D_presence": 0.8}
        },
    }

    values, source = _null_defects(defects, scale=PRINCIPAL_SCALE_PIXELS)

    # The bar is the easier control per indicator, and the two differ.
    assert values["D_shape"] == pytest.approx(0.7)
    assert source["D_shape"] == "homogeneous"
    assert values["D_presence"] == pytest.approx(0.8)
    assert source["D_presence"] == "translated"
    assert set(CONTROL_LABELS) == {"homogeneous", "translated"}


def test_the_zone_uses_paired_differences_not_overlapping_bands() -> None:
    """Amendment A4, and the reason for it.

    Two candidates offset by a constant have widely overlapping marginal bands
    yet a paired difference that never reaches zero. The marginal reading would
    call them indistinguishable; the paired one must not.
    """

    draws = 4000
    generator = np.random.default_rng(0)
    common = generator.normal(1.0, 0.30, draws)
    scores = np.vstack([common, common + 0.10])
    usable = np.ones(draws, dtype=bool)

    zone = _zone(scores, ["best", "offset"], usable=usable)

    # Marginal bands overlap heavily: the spread dwarfs the offset.
    assert np.quantile(scores[1], 0.05) < np.quantile(scores[0], 0.95)
    # The paired difference is a constant 0.10 and excludes zero.
    assert zone["members"] == ["best"]
    assert zone["reference"] == "best"
    assert zone["differences"]["offset"]["q05"] == pytest.approx(0.10, abs=1e-9)


def test_a_genuinely_tied_candidate_joins_the_zone() -> None:
    draws = 4000
    generator = np.random.default_rng(1)
    common = generator.normal(1.0, 0.30, draws)
    scores = np.vstack([common, common + generator.normal(0.0, 0.02, draws)])

    zone = _zone(scores, ["a", "b"], usable=np.ones(draws, dtype=bool))

    assert set(zone["members"]) == {"a", "b"}


def _report(*, share: float, verdict: str, winner: str, zone: list[str]) -> dict:
    scale = str(PRINCIPAL_SCALE_PIXELS)
    good = dict.fromkeys(DEFECT_NAMES, 0.2)
    control = dict.fromkeys(DEFECT_NAMES, 0.8)
    return {
        "bootstrap": {
            "most_frequent": winner,
            "most_frequent_share": share,
            "verdict": verdict,
        },
        "zone": zone,
        "minimax": {winner: 0.3},
        "raw_table": {winner: good},
        "defects": {label: {scale: control} for label in CONTROL_LABELS},
    }


def test_case_a_needs_both_robustness_and_beating_the_controls() -> None:
    outcome = _conclusion(
        _report(share=0.97, verdict="robustly_preferred", winner="a1-ell40", zone=["a1-ell40"])
    )

    assert outcome["case"] == "A_robust_optimum"
    assert outcome["beats_both_controls"] is True


def test_a_robust_winner_no_better_than_a_control_is_not_case_a() -> None:
    """A candidate that cannot beat a negative control is not a parameterisation."""

    report = _report(share=0.99, verdict="robustly_preferred", winner="a1-ell40", zone=["a1-ell40"])
    # Worse than the control on every indicator.
    report["raw_table"]["a1-ell40"] = dict.fromkeys(DEFECT_NAMES, 0.9)

    outcome = _conclusion(report)

    assert outcome["case"] != "A_robust_optimum"
    assert outcome["beats_both_controls"] is False


def test_a_multi_point_zone_is_case_b() -> None:
    outcome = _conclusion(
        _report(
            share=0.4,
            verdict="indistinguishable_zone",
            winner="a1-ell40",
            zone=["a1-ell40", "a2-ell20"],
        )
    )

    assert outcome["case"] == "B_robust_zone"


def test_a_single_unstable_point_is_case_c() -> None:
    outcome = _conclusion(
        _report(share=0.3, verdict="indistinguishable_zone", winner="a1-ell40", zone=["a1-ell40"])
    )

    assert outcome["case"] == "C_indicators_not_selective"
