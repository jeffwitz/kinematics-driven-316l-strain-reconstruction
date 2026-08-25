from __future__ import annotations

import json

import pytest

from fem_inhouse.identification.dic_noise_reference import load_dic_noise_reference


def _report(tmp_path, *, robust_px: float = 0.05, robust_mm: float = 9.2e-5):
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "temporal_noise": {
                    "robust_px": robust_px,
                    "robust_mm": robust_mm,
                    "rms_px": 0.04,
                    "rms_mm": 7.36e-5,
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_dic_noise_reference_checks_pixel_mm_conversion(tmp_path):
    path = _report(tmp_path, robust_mm=0.05 * 0.00184)
    result = load_dic_noise_reference(path, pixel_size_mm=0.00184)
    assert result["robust_px"] == pytest.approx(0.05)
    assert result["robust_mm"] == pytest.approx(9.2e-5)


def test_dic_noise_reference_rejects_inconsistent_units(tmp_path):
    path = _report(tmp_path, robust_mm=9.4e-5)
    with pytest.raises(ValueError, match="inconsistent pixel/mm"):
        load_dic_noise_reference(path, pixel_size_mm=0.00184)
