from __future__ import annotations

import pytest

from fem_inhouse.core.single_crystal_presets import (
    FCC_COLINEAR_SLOT,
    FCC_INTERACTION_COEFFICIENTS,
    SINGLE_CRYSTAL_PRESETS,
    CubicElasticity,
    get_preset,
)


def test_the_registry_is_not_silently_overwritten() -> None:
    """A parameter set is a citation; replacing one would make results
    computed with it unattributable."""

    from fem_inhouse.core import single_crystal_presets as module

    existing = next(iter(SINGLE_CRYSTAL_PRESETS.values()))
    with pytest.raises(ValueError, match="already registered"):
        module._register(existing)


def test_an_unknown_preset_names_the_registered_ones() -> None:
    with pytest.raises(KeyError, match="316l_guilhem2013_nasri2018"):
        get_preset("nope")


def test_engineering_constants_are_derived_not_stored() -> None:
    """Austenitic single crystals are soft along <100> and strongly
    anisotropic; 205 GPa is the polycrystal average, not a crystal constant."""

    elasticity = CubicElasticity(c11_mpa=197_000.0, c12_mpa=125_000.0, c44_mpa=122_000.0)

    assert elasticity.young_modulus_100_mpa == pytest.approx(99_950, rel=1e-3)
    assert elasticity.poisson_ratio_100 == pytest.approx(0.3882, abs=1e-4)
    assert elasticity.shear_modulus_mpa == 122_000.0
    assert elasticity.zener_anisotropy == pytest.approx(3.389, rel=1e-3)


def test_an_unstable_cubic_stiffness_is_rejected() -> None:
    with pytest.raises(ValueError, match="c11 > c12"):
        CubicElasticity(c11_mpa=100_000.0, c12_mpa=120_000.0, c44_mpa=80_000.0)


def test_the_complete_preset_drives_the_law() -> None:
    preset = get_preset("316l_guilhem2013_nasri2018")

    assert preset.complete
    assert preset.missing() == ()
    parameters = preset.mfront_parameters()
    assert parameters == {
        "n": 11.0,
        "K": 12.0,
        "tau0": 40.0,
        "Q": 10.0,
        "b": 3.0,
        "C": 40_000.0,
        "d": 1_500.0,
    }
    assert len(preset.interaction_matrix) == FCC_INTERACTION_COEFFICIENTS
    # Slot six is the colinear interaction, the strongest one.
    assert preset.colinear_coefficient == 12.3
    assert preset.colinear_coefficient == max(preset.interaction_matrix)
    assert preset.interaction_matrix[FCC_COLINEAR_SLOT - 1] == 12.3


def test_the_incomplete_preset_refuses_rather_than_inventing_values() -> None:
    """Six coefficients cannot be mapped onto MFront's seven from the numbers
    alone, and the source gives no tau0, Q or b."""

    preset = get_preset("316ln_guery2016_d50")

    assert not preset.complete
    missing = preset.missing()
    assert "initial_crss_mpa" in missing
    assert "isotropic_saturation_mpa" in missing
    assert "isotropic_rate" in missing
    assert any("6 coefficients" in gap for gap in missing)
    assert preset.colinear_coefficient is None

    with pytest.raises(ValueError, match="incomplete"):
        preset.mfront_parameters()


def test_the_two_conventions_order_the_colinear_term_differently() -> None:
    """The reason the six-coefficient set cannot be mapped: 12.3 sits in a
    different slot in each convention, so the orderings are not the same."""

    seven = get_preset("316l_guilhem2013_nasri2018").interaction_matrix
    six = get_preset("316ln_guery2016_d50").interaction_matrix

    assert seven.index(12.3) == FCC_COLINEAR_SLOT - 1
    assert six.index(12.3) != FCC_COLINEAR_SLOT - 1


def test_the_elastic_brick_options_are_cubic() -> None:
    options = get_preset("316l_guilhem2013_nasri2018").elastic_brick_options()

    assert options["young_modulus1"] == options["young_modulus2"] == options["young_modulus3"]
    assert options["shear_modulus12"] == 122_000.0
    # Cubic, not isotropic: the shear modulus is not E / (2 (1 + nu)).
    isotropic_shear = options["young_modulus1"] / (2.0 * (1.0 + options["poisson_ratio12"]))
    assert options["shear_modulus12"] > 3.0 * isotropic_shear


def test_presets_are_frozen() -> None:
    preset = get_preset("316l_guilhem2013_nasri2018")

    with pytest.raises((AttributeError, TypeError)):
        preset.initial_crss_mpa = 99.0  # type: ignore[misc]
