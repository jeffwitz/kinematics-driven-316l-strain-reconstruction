"""Pure tests for the explicit Meric--SRIX 316L parameter pair."""

from dataclasses import replace

import pytest

from fem_inhouse.core.crystal_parameter_pairs import (
    PAIRED_PARAMETER_SET,
    get_paired_crystal_parameter_set,
    resolve_paired_crystal_parameters,
)
from fem_inhouse.core.single_crystal_presets import CubicElasticity


def test_historical_backbone_and_interaction_matrix_are_locked() -> None:
    pair = get_paired_crystal_parameter_set(PAIRED_PARAMETER_SET)
    backbone = pair.backbone
    assert backbone.elasticity == CubicElasticity(197000.0, 125000.0, 122000.0)
    assert backbone.tau0_mpa == 40.0
    assert backbone.q_mpa == 10.0
    assert backbone.b == 3.0
    assert backbone.c_mpa == 40000.0
    assert backbone.d == 1500.0
    assert backbone.interaction_matrix == (1.0, 1.0, 0.6, 1.8, 1.6, 12.3, 1.6)


def test_digest_is_canonical_and_stable() -> None:
    pair = get_paired_crystal_parameter_set(PAIRED_PARAMETER_SET)
    assert pair.backbone.sha256() == replace(pair.backbone).sha256()
    altered = replace(pair.backbone, q_mpa=10.0000001)
    assert altered.sha256() != pair.backbone.sha256()


def test_r_is_the_single_allowed_analytical_transposition() -> None:
    pair = get_paired_crystal_parameter_set(PAIRED_PARAMETER_SET)
    expected = resolve_paired_crystal_parameters(
        paired_parameter_set=PAIRED_PARAMETER_SET,
        law="forest_rubin_srix",
    )[0]["SrixOverstressModulus"]
    assert expected == pytest.approx(pair.srix.overstress_modulus_mpa, rel=1.0e-13)


def test_resolution_changes_only_the_flow_rule_specific_names() -> None:
    meric, _ = resolve_paired_crystal_parameters(
        paired_parameter_set=PAIRED_PARAMETER_SET,
        law="meric_cailletaud",
    )
    srix, _ = resolve_paired_crystal_parameters(
        paired_parameter_set=PAIRED_PARAMETER_SET,
        law="forest_rubin_srix",
    )
    common = {"tau0", "Q", "b", "C", "d"} | {
        *[f"YoungModulus{i}" for i in (1, 2, 3)],
        *[f"PoissonRatio{p}" for p in ("12", "23", "13")],
        *[f"ShearModulus{p}" for p in ("12", "23", "13")],
    }
    assert {name: meric[name] for name in common} == {name: srix[name] for name in common}
    assert meric.keys() - common == {"K", "n"}
    assert srix.keys() - common == {
        "SrixOverstressModulus",
        "MinimumEquivalentStrainIncrement",
    }


def test_a_backbone_drift_is_refused() -> None:
    pair = get_paired_crystal_parameter_set(PAIRED_PARAMETER_SET)
    altered = replace(pair.backbone, elasticity=CubicElasticity(197000.0, 125000.0, 122001.0))
    with pytest.raises(ValueError, match="cubic elasticity"):
        replace(pair, backbone=altered).validate()
