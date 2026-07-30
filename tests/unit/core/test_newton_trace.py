from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from fem_inhouse.core.nonlinear import _tangent_diagonal_statistics
from fem_inhouse.examples import reduced_biaxial_case
from fem_inhouse.solver import run_case_study
from fem_inhouse.workflows.dic_multistep import NEWTON_TRACE_FIELDS, _write_newton_trace

TRACED_FIELDS = (
    "displacement_mm",
    "stress_mpa",
    "total_strain",
    "equivalent_plastic_strain",
)


def _run(trace: list[dict[str, object]] | None):
    case = reduced_biaxial_case(nx=3, ny=3, constitutive_backend="python")
    return run_case_study(
        case.config,
        displacement_x_mm=case.displacement_x_mm,
        displacement_y_mm=case.displacement_y_mm,
        yield_stress_mpa=case.yield_stress_mpa,
        hardening_coefficient_mpa=case.hardening_coefficient_mpa,
        newton_trace=trace,
    )


def test_newton_trace_does_not_change_the_numerical_path() -> None:
    """Preregistered constraint: the trace is observational only."""

    untraced = _run(None)
    trace: list[dict[str, object]] = []
    traced = _run(trace)

    for field in TRACED_FIELDS:
        np.testing.assert_array_equal(getattr(untraced, field), getattr(traced, field))
    assert trace


def test_newton_trace_records_one_entry_per_iteration() -> None:
    trace: list[dict[str, object]] = []
    _run(trace)

    assert all(record["outcome"] in {"converged", "corrected"} for record in trace)
    assert [record["increment"] for record in trace] == sorted(
        record["increment"] for record in trace
    )
    for record in trace:
        assert set(record).issubset(set(NEWTON_TRACE_FIELDS))
        assert record["boundary_increment_norm"] > 0.0
        assert np.isfinite(float(record["total_strain_maximum"]))


def test_newton_trace_reports_an_elastic_tangent_ratio_of_one() -> None:
    trace: list[dict[str, object]] = []
    _run(trace)

    ratios = [
        float(record["constitutive_to_elastic_tangent_ratio"])
        for record in trace
        if "constitutive_to_elastic_tangent_ratio" in record
    ]

    assert ratios
    # A J2 consistent tangent never exceeds the elastic operator in magnitude.
    assert max(ratios) == pytest.approx(1.0, rel=1.0e-9)


def test_tangent_diagonal_statistics_flags_a_degenerate_row() -> None:
    from scipy.sparse import csr_matrix

    healthy = csr_matrix(np.diag([2.0, 3.0, 4.0]))
    degenerate = csr_matrix(np.diag([2.0, 0.0, 4.0]))

    assert _tangent_diagonal_statistics(healthy)["tangent_diagonal_nonpositive_count"] == 0
    flagged = _tangent_diagonal_statistics(degenerate)
    assert flagged["tangent_diagonal_nonpositive_count"] == 1
    assert flagged["tangent_diagonal_minimum"] == 0.0


def test_write_newton_trace_pads_missing_keys(tmp_path: Path) -> None:
    _write_newton_trace(
        tmp_path,
        [
            {"increment": 1, "newton_iteration": 1, "outcome": "corrected"},
            {"increment": 1, "newton_iteration": 2, "outcome": "constitutive_rejection"},
        ],
    )

    with (tmp_path / "newton_trace.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert [row["outcome"] for row in rows] == ["corrected", "constitutive_rejection"]
    assert rows[0]["correction_norm"] == ""
    assert list(rows[0]) == list(NEWTON_TRACE_FIELDS)


def test_write_newton_trace_skips_an_empty_trace(tmp_path: Path) -> None:
    _write_newton_trace(tmp_path, None)
    _write_newton_trace(tmp_path, [])

    assert not (tmp_path / "newton_trace.csv").exists()
