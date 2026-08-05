from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.compare_srix_meric_slip_maps_p43 import _load_fields, _resolve_field_file

ROOT = Path(__file__).parents[3]
MERIC_REPORT = ROOT / "validation/_generated/performance/crystal_tet2_meric_p43_m100_slip_maps.json"
SRIX_REPORT = (
    ROOT / "validation/_generated/performance/crystal_tet2_srix_p43_m100_16_slip_maps.json"
)


def _report(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_registered_p43_slip_archives_have_expected_contract() -> None:
    meric = _report(MERIC_REPORT)
    srix = _report(SRIX_REPORT)

    assert meric["mesh"] == [100, 100]
    assert srix["mesh"] == [100, 100]
    assert meric["increments"] == 16
    assert srix["increments"] == 16
    assert (
        meric["crystal_material"]["backbone"]["sha256"]
        == srix["crystal_material"]["backbone"]["sha256"]
    )
    assert meric["orientation"]["sha256"] == srix["orientation"]["sha256"]

    meric_equivalent, meric_signed = _load_fields(
        _resolve_field_file(MERIC_REPORT, meric)
    )
    srix_equivalent, srix_signed = _load_fields(
        _resolve_field_file(SRIX_REPORT, srix)
    )
    assert meric_equivalent.shape == (12, 100, 100)
    assert srix_equivalent.shape == (12, 100, 100)
    assert meric_signed.shape == meric_equivalent.shape
    assert srix_signed.shape == srix_equivalent.shape
    assert np.isclose(float(np.max(meric_equivalent)), 0.019392914417017293)
    assert np.isclose(float(np.max(srix_equivalent)), 0.008584967144019375)
