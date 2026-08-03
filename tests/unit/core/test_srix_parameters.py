"""Registered SRIX parameter sets, their provenance, and their configuration.

Sections 4, 5 and 6 of the specification. The tests that need MGIS are in
`test_forest_rubin_srix.py`; everything here is pure and always runs.

The recurring subject is the same one: a parameter set mixes values of very
different standing -- an elasticity adopted from a paper, a threshold taken as a
prior, a modulus transposed from a different flow rule entirely -- and the
software must never let that mixture be reported as an identification of 316L.
"""

from __future__ import annotations

import math

import pytest

from fem_inhouse.core.single_crystal_presets import CubicElasticity
from fem_inhouse.core.srix_parameters import (
    DEFAULT_PARAMETER_SET,
    ELASTIC_PARAMETER_NAMES,
    EXPLICIT_PARAMETER_NAMES,
    EXPLORATORY_OVERSTRESS_MODULI_MPA,
    SRIX_PARAMETER_SETS,
    UNIAXIAL_FACTOR,
    ParameterOrigin,
    SrixParameterSet,
    get_parameter_set,
    resolve_srix_parameters,
)

HISTORICAL = "316l_srix_transposed_from_nasri2018_rate_1e-3"
UPDATED = "316l_srix_updated_elasticity_prior"


class TestRegistry:
    def test_the_historical_set_keeps_the_published_numbers(self) -> None:
        """Section 6.1. These are the values every archived result was run with."""

        historical = get_parameter_set(HISTORICAL)

        assert historical.elasticity.c11_mpa == 197_000.0
        assert historical.elasticity.c12_mpa == 125_000.0
        assert historical.elasticity.c44_mpa == 122_000.0
        assert historical.tau0_mpa == 40.0
        assert historical.q_mpa == 10.0
        assert historical.b == 3.0
        assert historical.c_mpa == 40_000.0
        assert historical.d == 1_500.0
        assert historical.overstress_modulus_mpa == pytest.approx(18.7819100705, abs=1e-9)

    def test_the_identifier_says_transposed_rather_than_just_srix(self) -> None:
        """Section 2. The name is the first thing a reader sees in a manifest."""

        assert "transposed" in HISTORICAL
        assert "nasri2018" in HISTORICAL
        assert "rate_1e-3" in HISTORICAL
        assert HISTORICAL in SRIX_PARAMETER_SETS

    def test_the_default_is_the_historical_set(self) -> None:
        """So an unconfigured run reproduces what is already archived."""

        assert DEFAULT_PARAMETER_SET == HISTORICAL

    def test_the_updated_elastic_set_changes_elasticity_and_threshold_only(self) -> None:
        """Section 6.2. Everything else is explicitly inherited and provisional."""

        historical = get_parameter_set(HISTORICAL)
        updated = get_parameter_set(UPDATED)

        assert updated.elasticity.c11_mpa == 218_300.0
        assert updated.elasticity.c12_mpa == 144_800.0
        assert updated.elasticity.c44_mpa == 125_400.0
        assert updated.tau0_mpa == pytest.approx(38.33)
        assert updated.q_mpa == historical.q_mpa
        assert updated.b == historical.b
        assert updated.c_mpa == historical.c_mpa
        assert updated.d == historical.d
        assert updated.interaction_matrix == historical.interaction_matrix

    def test_the_updated_elasticity_keeps_essentially_the_same_anisotropy(self) -> None:
        """Worth stating: the change is in stiffness level, not in anisotropy."""

        historical = get_parameter_set(HISTORICAL).elasticity.zener_anisotropy
        updated = get_parameter_set(UPDATED).elasticity.zener_anisotropy

        assert updated == pytest.approx(historical, rel=0.01)
        assert historical > 3.0

    @pytest.mark.parametrize("value", EXPLORATORY_OVERSTRESS_MODULI_MPA)
    def test_each_exploratory_set_changes_r_and_nothing_else(self, value: float) -> None:
        """Section 6.3, so a difference between two of them is attributable."""

        historical = get_parameter_set(HISTORICAL)
        candidates = [
            candidate
            for candidate in SRIX_PARAMETER_SETS.values()
            if candidate.overstress_origin.status == "exploratory"
            and candidate.overstress_modulus_mpa == value
        ]

        assert len(candidates) == 1
        exploratory = candidates[0]
        assert exploratory.elasticity == historical.elasticity
        assert exploratory.tau0_mpa == historical.tau0_mpa
        assert exploratory.q_mpa == historical.q_mpa
        assert exploratory.c_mpa == historical.c_mpa
        assert exploratory.interaction_matrix == historical.interaction_matrix

    def test_the_sweep_brackets_the_historical_value_and_contains_it(self) -> None:
        assert min(EXPLORATORY_OVERSTRESS_MODULI_MPA) == 1.0
        assert max(EXPLORATORY_OVERSTRESS_MODULI_MPA) == pytest.approx(18.7819100705)

    def test_an_unknown_identifier_lists_the_registered_ones(self) -> None:
        with pytest.raises(KeyError, match="registered:"):
            get_parameter_set("no_such_set")


class TestOverstressRatio:
    def test_it_is_the_registered_formula(self) -> None:
        """Section 6.3: `O_R = (sqrt(6) / 8) R / tau0`."""

        historical = get_parameter_set(HISTORICAL)
        expected = (
            math.sqrt(6.0) / 8.0 * historical.overstress_modulus_mpa / historical.tau0_mpa
        )

        assert pytest.approx(math.sqrt(6.0) / 8.0) == UNIAXIAL_FACTOR
        assert historical.overstress_ratio == pytest.approx(expected)
        assert historical.overstress_ratio == pytest.approx(0.143769, abs=1e-6)

    def test_it_orders_the_sweep(self) -> None:
        """A larger R rounds the transition more, at fixed threshold."""

        ratios = [
            get_parameter_set(name).overstress_ratio
            for name in sorted(
                SRIX_PARAMETER_SETS,
                key=lambda k: SRIX_PARAMETER_SETS[k].overstress_modulus_mpa,
            )
            if SRIX_PARAMETER_SETS[name].overstress_origin.status == "exploratory"
        ]

        assert ratios == sorted(ratios)


class TestProvenance:
    def test_no_registered_set_claims_to_identify_316l(self) -> None:
        """Section 2 and criterion 16. The load-bearing assertion of this file."""

        for name, candidate in SRIX_PARAMETER_SETS.items():
            assert not candidate.claims_material_identification, name
            assert candidate.weakest_statuses(), name

    def test_the_historical_overstress_modulus_is_marked_transposed(self) -> None:
        historical = get_parameter_set(HISTORICAL)

        assert historical.overstress_origin.status == "analytical_transposition"
        assert historical.reference_strain_rate == pytest.approx(1e-3)
        assert "equation (16)" in historical.overstress_origin.reference

    def test_the_exploratory_modulus_is_marked_exploratory(self) -> None:
        exploratory = get_parameter_set("316l_srix_exploratory_r1")

        assert exploratory.overstress_origin.status == "exploratory"
        assert "no claim" in exploratory.overstress_origin.note

    def test_the_updated_elasticity_does_not_invent_a_citation(self) -> None:
        """Section 16 forbids inventing missing coefficients; a citation is worse.

        The specification supplied the three stiffnesses with no source, and the
        record has to say so rather than attach a plausible-looking paper.
        """

        origin = get_parameter_set(UPDATED).elastic_origin

        assert origin.status == "literature_prior"
        assert "not supplied" in origin.reference

    def test_a_record_carries_every_group_with_a_status_and_a_reference(self) -> None:
        record = get_parameter_set(HISTORICAL).provenance_record()

        assert set(record["origins"]) == {
            "elasticity",
            "tau0",
            "isotropic_hardening",
            "kinematic_hardening",
            "overstress_modulus",
            "interaction_matrix",
        }
        for group in record["origins"].values():
            assert group["status"]
            assert group["reference"]

    def test_a_record_states_units_for_every_value(self) -> None:
        """Section 5. A number without a unit is not traceable."""

        record = get_parameter_set(HISTORICAL).provenance_record()

        assert set(record["values"]) <= set(record["units"])
        assert record["units"]["R_mpa"] == "MPa"
        assert record["units"]["b"] == "1"
        assert record["units"]["reference_strain_rate"] == "1/s"

    def test_a_record_carries_the_interaction_matrix_and_its_convention(self) -> None:
        record = get_parameter_set(HISTORICAL).provenance_record()

        assert len(record["interaction_matrix"]["coefficients"]) == 7
        assert "colinear" in record["interaction_matrix"]["convention"]
        assert "fcc_interaction_matrix_mapping" in record["interaction_matrix"]["convention"]


class TestConstruction:
    @staticmethod
    def _origin(status: str = "literature_prior") -> ParameterOrigin:
        return ParameterOrigin(status=status, reference="test")  # type: ignore[arg-type]

    def _set(self, **overrides: object) -> SrixParameterSet:
        base: dict[str, object] = {
            "identifier": "probe",
            "elasticity": CubicElasticity(c11_mpa=2.0e5, c12_mpa=1.2e5, c44_mpa=1.2e5),
            "interaction_matrix": (1.0,) * 7,
            "overstress_modulus_mpa": 10.0,
            "tau0_mpa": 40.0,
            "q_mpa": 10.0,
            "b": 3.0,
            "c_mpa": 4.0e4,
            "d": 1.5e3,
            "elastic_origin": self._origin(),
            "threshold_origin": self._origin(),
            "isotropic_origin": self._origin(),
            "kinematic_origin": self._origin(),
            "overstress_origin": self._origin(),
            "interaction_origin": self._origin(),
            "overstress_method": "probe",
        }
        base.update(overrides)
        return SrixParameterSet(**base)  # type: ignore[arg-type]

    def test_a_valid_set_builds(self) -> None:
        assert self._set().identifier == "probe"

    def test_a_transposed_modulus_without_a_rate_is_refused(self) -> None:
        """The transposition is only defined at a rate; hiding it is not allowed."""

        with pytest.raises(ValueError, match="states no reference strain rate"):
            self._set(overstress_origin=self._origin("analytical_transposition"))

    def test_a_transposed_modulus_with_a_rate_is_accepted(self) -> None:
        candidate = self._set(
            overstress_origin=self._origin("analytical_transposition"),
            reference_strain_rate=1e-3,
        )

        assert candidate.reference_strain_rate == pytest.approx(1e-3)

    @pytest.mark.parametrize("name", ["overstress_modulus_mpa", "tau0_mpa", "q_mpa", "c_mpa"])
    def test_a_non_positive_modulus_is_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match=f"{name} must be positive"):
            self._set(**{name: 0.0})

    def test_an_interaction_matrix_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="seven coefficients"):
            self._set(interaction_matrix=(1.0,) * 6)

    def test_an_origin_must_cite_something(self) -> None:
        with pytest.raises(ValueError, match="must cite something"):
            ParameterOrigin(status="exploratory", reference="")

    def test_a_fully_identified_set_would_claim_material_knowledge(self) -> None:
        """The positive case, so the guard is known to be reachable."""

        identified = self._set(
            elastic_origin=self._origin("literature_measurement"),
            threshold_origin=self._origin("identified"),
            isotropic_origin=self._origin("identified"),
            kinematic_origin=self._origin("identified"),
            overstress_origin=self._origin("identified"),
            interaction_origin=self._origin("literature_measurement"),
        )

        assert identified.claims_material_identification
        assert identified.weakest_statuses() == ()


class TestMFrontOverrides:
    def test_r_is_exported_under_its_entry_name(self) -> None:
        """`R` in the papers, `SrixOverstressModulus` in the compiled behaviour."""

        overrides = get_parameter_set(HISTORICAL).mfront_overrides()

        assert "R" not in overrides
        assert overrides["SrixOverstressModulus"] == pytest.approx(18.7819100705)

    def test_the_cubic_stiffnesses_arrive_as_engineering_constants(self) -> None:
        """The brick is orthotropic, so C11/C12/C44 never reach it directly."""

        historical = get_parameter_set(HISTORICAL)
        overrides = historical.mfront_overrides()

        assert "C11_mpa" not in overrides
        for axis in (1, 2, 3):
            assert overrides[f"YoungModulus{axis}"] == pytest.approx(
                historical.elasticity.young_modulus_100_mpa
            )
        for pair in ("12", "23", "13"):
            assert overrides[f"PoissonRatio{pair}"] == pytest.approx(
                historical.elasticity.poisson_ratio_100
            )
            assert overrides[f"ShearModulus{pair}"] == historical.elasticity.c44_mpa

    def test_every_value_is_a_plain_float(self) -> None:
        """MGIS setParameter takes a C double; a numpy scalar has bitten before."""

        for value in get_parameter_set(HISTORICAL).mfront_overrides().values():
            assert type(value) is float


class TestResolve:
    def test_no_configuration_selects_the_default_and_says_it_was_not_chosen(self) -> None:
        overrides, record = resolve_srix_parameters()

        assert record["identifier"] == DEFAULT_PARAMETER_SET
        assert record["selected_explicitly"] is False
        assert record["explicit_overrides"] == {}
        assert overrides == get_parameter_set(DEFAULT_PARAMETER_SET).mfront_overrides()

    def test_selecting_the_default_by_name_is_recorded_as_a_choice(self) -> None:
        _, record = resolve_srix_parameters(parameter_set=DEFAULT_PARAMETER_SET)

        assert record["selected_explicitly"] is True

    def test_an_unknown_set_is_refused_before_anything_is_loaded(self) -> None:
        with pytest.raises(KeyError, match="registered:"):
            resolve_srix_parameters(parameter_set="no_such_set")

    def test_a_non_string_set_is_refused(self) -> None:
        with pytest.raises(TypeError, match="registered identifier string"):
            resolve_srix_parameters(parameter_set=18.78)

    def test_an_inline_value_overrides_the_preset(self) -> None:
        overrides, record = resolve_srix_parameters(explicit={"R_mpa": 2.0})

        assert overrides["SrixOverstressModulus"] == 2.0
        assert record["explicit_overrides"] == {"R_mpa": 2.0}
        assert record["base_parameter_set"] == DEFAULT_PARAMETER_SET
        assert record["identifier"].endswith("+inline")

    def test_an_inline_value_demotes_its_group_to_exploratory(self) -> None:
        """Nothing knows where an inline number came from, so it claims nothing."""

        _, record = resolve_srix_parameters(explicit={"R_mpa": 2.0})

        assert record["origins"]["overstress_modulus"]["status"] == "exploratory"
        assert record["claims_material_identification"] is False
        assert "overstress_modulus" in record["weakest_statuses"]

    def test_an_inline_hardening_value_demotes_the_right_group(self) -> None:
        _, record = resolve_srix_parameters(explicit={"C_mpa": 1.0, "d": 2.0})

        assert record["origins"]["kinematic_hardening"]["status"] == "exploratory"
        assert record["origins"]["isotropic_hardening"]["status"] != "exploratory"

    def test_the_cubic_stiffnesses_are_overridden_as_a_group(self) -> None:
        overrides, record = resolve_srix_parameters(
            explicit={"C11_mpa": 2.0e5, "C12_mpa": 1.2e5, "C44_mpa": 1.2e5}
        )
        expected = CubicElasticity(c11_mpa=2.0e5, c12_mpa=1.2e5, c44_mpa=1.2e5)

        assert overrides["YoungModulus1"] == pytest.approx(expected.young_modulus_100_mpa)
        assert overrides["ShearModulus12"] == expected.shear_modulus_mpa
        assert record["origins"]["elasticity"]["status"] == "exploratory"
        assert record["values"]["zener_anisotropy"] == pytest.approx(
            expected.zener_anisotropy
        )

    @pytest.mark.parametrize("supplied", [("C11_mpa",), ("C11_mpa", "C12_mpa")])
    def test_a_partial_elasticity_is_refused(self, supplied: tuple[str, ...]) -> None:
        """Two of three stiffnesses describe no material."""

        with pytest.raises(ValueError, match="overridden as a group"):
            resolve_srix_parameters(explicit=dict.fromkeys(supplied, 1.0e5))

    def test_an_unknown_parameter_name_lists_the_accepted_ones(self) -> None:
        with pytest.raises(ValueError, match="unknown SRIX parameter"):
            resolve_srix_parameters(explicit={"R": 2.0})

    def test_a_non_finite_value_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            resolve_srix_parameters(explicit={"R_mpa": float("nan")})

    def test_a_non_mapping_parameters_block_is_refused(self) -> None:
        with pytest.raises(TypeError, match="must be a mapping"):
            resolve_srix_parameters(explicit=[("R_mpa", 2.0)])

    def test_the_accepted_names_cover_the_specified_parameters(self) -> None:
        """Section 4 names R, tau0, Q, b, C and d explicitly."""

        assert set(EXPLICIT_PARAMETER_NAMES) == {
            "R_mpa",
            "tau0_mpa",
            "Q_mpa",
            "b",
            "C_mpa",
            "d",
        }
        assert set(ELASTIC_PARAMETER_NAMES) == {"C11_mpa", "C12_mpa", "C44_mpa"}


class TestRunProvenance:
    """Section 5's other half: the machine and the moment, not the parameters."""

    def test_the_record_carries_the_mfront_source_digest(self) -> None:
        from fem_inhouse.core.srix_parameters import srix_provenance

        record = srix_provenance(mfront_source="mfront/Fcc316LForestRubinSrix.mfront")
        source = record["run"]["mfront_source"]

        assert source["path"].endswith("Fcc316LForestRubinSrix.mfront")
        assert len(source["sha256"]) == 64

    def test_a_missing_source_is_reported_rather_than_faked(self) -> None:
        from fem_inhouse.core.srix_parameters import srix_provenance

        source = srix_provenance(mfront_source="no/such/file.mfront")["run"][
            "mfront_source"
        ]

        assert source["sha256"] is None
        assert "not found" in source["note"]

    def test_an_absent_source_path_is_recorded_as_absent(self) -> None:
        from fem_inhouse.core.srix_parameters import srix_provenance

        source = srix_provenance()["run"]["mfront_source"]

        assert source["path"] is None
        assert source["sha256"] is None

    def test_the_toolchain_and_commit_keys_always_exist(self) -> None:
        """They may be `None` off a checkout, but the reader must see the key.

        A fabricated version string is worse than an absent one, so nothing here
        guesses; the contract is that the field is present and honest.
        """

        from fem_inhouse.core.srix_parameters import srix_provenance

        run = srix_provenance()["run"]

        assert set(run) == {"mfront_source", "toolchain", "git_commit"}
        assert set(run["toolchain"]) == {"tfel", "mgis"}

    def test_the_parameter_half_is_unchanged_by_the_run_half(self) -> None:
        from fem_inhouse.core.srix_parameters import srix_provenance

        resolved = resolve_srix_parameters(parameter_set=HISTORICAL)[1]
        full = srix_provenance(parameter_set=HISTORICAL)

        assert {k: v for k, v in full.items() if k != "run"} == resolved
