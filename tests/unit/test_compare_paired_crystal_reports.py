"""Tests for fail-fast paired-report comparison."""

from scripts.compare_paired_crystal_reports import compare


def _report(increments: int) -> dict[str, object]:
    return {
        "behaviour": "fcc_meric_cailletaud",
        "increments": increments,
        "crop_nodes": [1570, 1670, 1035, 1135],
        "mesh": [100, 100],
        "boundary_sha256": "boundary",
        "units": "mm, MPa",
        "orientation": {"sha256": "orientation"},
        "crystal_material": {
            "backbone": {
                "sha256": "backbone",
                "slip_systems": {
                    "crystal_structure": "FCC",
                    "family": "<0,1,-1>{1,1,1}",
                    "count": 12,
                },
                "interaction_matrix": [1.0, 1.0, 0.6, 1.8, 1.6, 12.3, 1.6],
                "interaction_convention": "mfront_fcc_7",
            },
            "mfront_structure": {"structure_contract_sha256": "structure"},
        },
    }


def test_different_temporal_discretisations_are_not_comparable() -> None:
    result = compare(_report(16), _report(8))
    assert result["comparison_authorized"] is False
    assert result["field_comparison_authorized"] is False
    assert result["performance_comparison_authorized"] is False
    assert "different temporal discretizations" in result["reasons"][0]


def test_identical_manifests_are_authorized() -> None:
    result = compare(_report(8), _report(8))
    assert result["comparison_authorized"] is True
    assert result["field_comparison_authorized"] is True
    assert result["performance_comparison_authorized"] is True
