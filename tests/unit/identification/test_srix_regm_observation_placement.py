"""Small tests for the SRIX-REGM observation-placement ablation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.ablate_srix_regm_observation_placement import _pseudo_trajectory_rms


class _State:
    scored = True

    def __init__(self, norm: float) -> None:
        self.pseudo_displacement_norm = norm


class _Evaluation:
    states = (_State(2.0), _State(4.0))


def test_pseudo_trajectory_rms_normalises_by_nodal_dofs() -> None:
    result = _pseudo_trajectory_rms(_Evaluation(), (2, 2))
    assert result == pytest.approx(np.sqrt((2.0**2 + 4.0**2) / 16.0))
