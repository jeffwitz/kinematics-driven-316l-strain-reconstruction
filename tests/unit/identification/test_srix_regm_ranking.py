"""Small deterministic tests for the preregistered REGM/FEMU ranking driver."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.compare_srix_regm_femu import _femu_rms, _population, _statistics
from scripts.compare_srix_regm_femu_observed import _observed_rms


class _Identity:
    def apply(self, values: np.ndarray) -> np.ndarray:
        return values


def test_preregistered_population_is_fixed_and_off_truth() -> None:
    population = _population()
    assert len(population) == 20
    assert population[0][0] == "initial"
    assert len({identifier for identifier, _, _ in population}) == 20
    assert all(np.linalg.norm(offset) > 0.0 for _, _, offset in population)


def test_femu_rms_uses_only_interior_macro_endpoints() -> None:
    target = np.zeros((3, 4, 4, 2))
    candidate = target.copy()
    candidate[1:, 1:-1, 1:-1, :] = 2.0
    candidate[1:, (0, -1), :, :] = 1.0e6
    candidate[1:, :, (0, -1), :] = 1.0e6
    assert _femu_rms(candidate, (1, 2), target, (1, 2)) == pytest.approx(2.0)


def test_ranking_statistics_apply_all_preregistered_gates() -> None:
    candidates = [
        {
            "id": f"c{index}",
            "status": "complete",
            "regm_rms_mm": float(index),
            "femu_rms_mm": float(index**2),
            "regm_seconds": 1.0,
            "femu_seconds": 10.0,
        }
        for index in range(1, 21)
    ]
    result = _statistics(candidates)
    assert result["gate_passed"] is True
    assert result["best_five_overlap_count"] == 5
    assert result["median_speedup"] == pytest.approx(10.0)


def test_observed_rms_excludes_boundary_reactions() -> None:
    target = np.zeros((2, 5, 5, 2))
    candidate = target.copy()
    candidate[:, 1:-1, 1:-1, :] = 3.0
    candidate[:, (0, -1), :, :] = 1.0e6
    candidate[:, :, (0, -1), :] = 1.0e6
    assert _observed_rms(candidate, target, _Identity()) == pytest.approx(3.0)
