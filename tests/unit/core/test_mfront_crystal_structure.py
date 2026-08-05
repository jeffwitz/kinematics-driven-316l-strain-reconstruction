"""Tests for compile-time FCC structure fingerprints."""

from pathlib import Path

from fem_inhouse.core.mfront_crystal_structure import read_crystal_structure_fingerprint


def test_meric_and_srix_have_the_same_fcc_structure_contract() -> None:
    meric = read_crystal_structure_fingerprint(Path("mfront/Fcc316LMericCailletaud.mfront"))
    srix = read_crystal_structure_fingerprint(Path("mfront/Fcc316LForestRubinSrix.mfront"))
    assert meric.crystal_structure == srix.crystal_structure == "FCC"
    assert meric.sliding_system == srix.sliding_system == "<0,1,-1>{1,1,1}"
    assert meric.interaction_matrix == srix.interaction_matrix == (
        1.0,
        1.0,
        0.6,
        1.8,
        1.6,
        12.3,
        1.6,
    )
