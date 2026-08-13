"""Canonical Forest-Rubin qualification, independent of 316L.

Sections 8 and 9. These tests describe the *model*, not the material, so they
can falsify the implementation while every parameter is still unsettled. The
material-dependent work is in the sensitivity campaign; nothing here identifies
anything.

The closed-form results are derived in `fem_inhouse.core.srix_canonical`. Tests
that touch MGIS skip without `MFRONT_BEHAVIOUR_LIBRARY`, following the
convention of the other MFront tests.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import pytest

from fem_inhouse.core.srix_canonical import (
    ACTIVE_SYSTEMS_001,
    SLIP_PER_EQUIVALENT_RATE_001,
    EnergyBalance,
    kinematic_stored_energy,
    overstress_diagnostic,
    slip_dissipation_increments,
    uniaxial_001_plateau_stress,
    uniaxial_001_relative_overstress,
)
from fem_inhouse.core.srix_parameters import get_parameter_set

SRIX = "Fcc316LForestRubinSrix"
HISTORICAL = "316l_srix_transposed_from_nasri2018_rate_1e-3"

ELASTIC_STRAIN = slice(0, 6)
PLASTIC_SLIP = slice(6, 18)
EQUIVALENT_SLIP = slice(18, 30)
BACK_STRAIN = slice(30, 42)

_BEHAVIOURS: dict[tuple[str, str], Any] = {}


def _mgis() -> tuple[Any, str]:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if not library:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    mgis = pytest.importorskip("mgis.behaviour")
    return mgis, library


#: Compiled defaults of every parameter these tests touch.
#:
#: `mgis.load` returns handles onto a SHARED behaviour: two loads of the same
#: library, name and hypothesis are the same object, and a `setParameter`
#: through one is visible through the other for the rest of the process. A test
#: that set `tau0` therefore changed every later test that relied on the
#: default, and the failures looked like physics rather than like leakage.
#:
#: The fix is to state every value on every call instead of inheriting any.
_DEFAULTS: dict[str, float] = {
    "SrixOverstressModulus": 18.7819100705,
    "tau0": 40.0,
    "Q": 10.0,
    "b": 3.0,
    "C": 40000.0,
    "d": 1500.0,
}


def _manager(*, parameters: dict[str, float] | None = None, points: int = 1) -> Any:
    """A manager whose behaviour carries exactly the requested parameters.

    Every parameter is written on every call, including the ones left at their
    default, because the behaviour is shared process-wide; see `_DEFAULTS`.
    """

    mgis, library = _mgis()
    behaviour = _BEHAVIOURS.get((library, SRIX))
    if behaviour is None:
        behaviour = mgis.load(library, SRIX, mgis.Hypothesis.Tridimensional)
        _BEHAVIOURS[(library, SRIX)] = behaviour
    for name, value in {**_DEFAULTS, **(parameters or {})}.items():
        mgis.setParameter(behaviour, name, float(value))
    data = mgis.MaterialDataManager(behaviour, points)
    for state in (data.s0, data.s1):
        mgis.setExternalStateVariable(state, "Temperature", 293.15)
        _set_local_coupling(mgis, behaviour, state)
    return mgis, data


def _set_local_coupling(mgis: Any, behaviour: Any, state: Any) -> None:
    """Neutralise the scalar micromorphic extension when the law declares it.

    MGIS gives neither `@MaterialProperty` nor `@ExternalStateVariable` a
    default, so a declared-but-unsupplied one fails inside `buildEvaluators` on
    the first integration rather than at construction. At `Hchi = 0` the law
    reduces exactly to the historical local SRIX response -- the only response
    this file is about.
    """

    for variable in behaviour.mps:
        if variable.name == "MicromorphicCouplingModulus":
            mgis.setMaterialProperty(state, variable.name, 0.0)
    for variable in behaviour.esvs:
        if variable.name == "NonlocalEquivalentPlasticStrain":
            mgis.setExternalStateVariable(state, variable.name, 0.0)


def _no_hardening(overstress: float, tau0: float) -> dict[str, float]:
    """`b = 0` removes isotropic hardening, `C = 0` removes the back stress."""

    return {
        "SrixOverstressModulus": overstress,
        "tau0": tau0,
        "b": 0.0,
        "C": 0.0,
    }


def _uniaxial_001(axial: float) -> np.ndarray:
    """Axial strain along `[001]` with plastic-incompressible contraction."""

    strain = np.zeros((1, 6))
    strain[0, 2] = axial
    strain[0, 0] = strain[0, 1] = -0.5 * axial
    return strain


def _drive(
    mgis: Any,
    data: Any,
    path: np.ndarray,
    *,
    time_increments: np.ndarray | None = None,
) -> None:
    steps = path.shape[0]
    increments = np.ones(steps) if time_increments is None else time_increments
    for index in range(steps):
        data.s1.gradients[:, :] = path[index]
        mgis.integrate(
            data,
            mgis.IntegrationType.IntegrationWithConsistentTangentOperator,
            float(increments[index]),
            0,
            1,
        )
        mgis.update(data)


def _axial_deviatoric_stress(data: Any) -> float:
    stress = data.s1.thermodynamic_forces[0]
    return float(stress[2] - 0.5 * (stress[0] + stress[1]))


# ---------------------------------------------------------------------------
# Section 8.1 -- the analytical [001] solution.
# ---------------------------------------------------------------------------


class TestClosedForm:
    """The derivation itself, with no MGIS involved."""

    def test_the_plateau_is_the_derived_expression(self) -> None:
        assert uniaxial_001_plateau_stress(
            tau0_mpa=40.0, overstress_modulus_mpa=18.7819100705
        ) == pytest.approx(math.sqrt(6.0) * 40.0 + 0.75 * 18.7819100705)

    def test_the_relative_overstress_is_the_registered_ratio(self) -> None:
        """`O_R` is not a separate convention: it is this number."""

        preset = get_parameter_set(HISTORICAL)

        assert uniaxial_001_relative_overstress(
            tau0_mpa=preset.tau0_mpa,
            overstress_modulus_mpa=preset.overstress_modulus_mpa,
        ) == pytest.approx(preset.overstress_ratio, rel=1e-15)

    def test_a_vanishing_overstress_modulus_recovers_the_schmid_limit(self) -> None:
        """As `R -> 0` the transition sharpens to the rate-independent corner."""

        assert uniaxial_001_plateau_stress(
            tau0_mpa=40.0, overstress_modulus_mpa=1e-12
        ) == pytest.approx(math.sqrt(6.0) * 40.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_non_positive_inputs_are_refused(self, bad: float) -> None:
        with pytest.raises(ValueError):
            uniaxial_001_plateau_stress(tau0_mpa=bad, overstress_modulus_mpa=1.0)
        with pytest.raises(ValueError):
            uniaxial_001_plateau_stress(tau0_mpa=1.0, overstress_modulus_mpa=bad)


@pytest.mark.mfront
class TestUniaxial001:
    """Section 8.1 against the compiled behaviour."""

    @pytest.mark.parametrize(
        ("overstress", "tau0"),
        [(18.7819100705, 40.0), (2.0, 40.0), (8.0, 60.0), (1.0, 25.0)],
    )
    def test_the_plateau_matches_the_closed_form(
        self, overstress: float, tau0: float
    ) -> None:
        """The sharpest statement this project can make about the flow rule."""

        mgis, data = _manager(parameters=_no_hardening(overstress, tau0))
        steps = 200
        path = np.concatenate(
            [_uniaxial_001(0.02 * (index + 1) / steps) for index in range(steps)]
        ).reshape(steps, 1, 6)
        _drive(mgis, data, path)

        expected = uniaxial_001_plateau_stress(
            tau0_mpa=tau0, overstress_modulus_mpa=overstress
        )
        assert _axial_deviatoric_stress(data) == pytest.approx(expected, rel=1e-12)

    def test_exactly_eight_systems_are_active(self) -> None:
        """The other four have a zero Schmid factor and must stay exactly zero."""

        mgis, data = _manager(parameters=_no_hardening(18.7819100705, 40.0))
        path = np.concatenate(
            [_uniaxial_001(0.02 * (index + 1) / 50) for index in range(50)]
        ).reshape(50, 1, 6)
        _drive(mgis, data, path)

        slips = data.s1.internal_state_variables[0, PLASTIC_SLIP]
        active = np.abs(slips) > 1e-14

        assert int(active.sum()) == ACTIVE_SYSTEMS_001
        assert np.all(np.abs(slips[~active]) == 0.0)

    def test_the_eight_active_systems_slip_identically(self) -> None:
        """Cubic symmetry of the `[001]` axis; anything else is a defect."""

        mgis, data = _manager(parameters=_no_hardening(18.7819100705, 40.0))
        path = np.concatenate(
            [_uniaxial_001(0.02 * (index + 1) / 50) for index in range(50)]
        ).reshape(50, 1, 6)
        _drive(mgis, data, path)

        slips = np.abs(data.s1.internal_state_variables[0, PLASTIC_SLIP])
        active = slips[slips > 1e-14]

        assert active.std() / active.mean() < 1e-12

    def test_the_slip_per_system_is_the_predicted_fraction(self) -> None:
        """`gamma = (sqrt(6)/8) * axial plastic strain`, the kinematic identity."""

        mgis, data = _manager(parameters=_no_hardening(18.7819100705, 40.0))
        steps, axial = 200, 0.02
        path = np.concatenate(
            [_uniaxial_001(axial * (index + 1) / steps) for index in range(steps)]
        ).reshape(steps, 1, 6)
        _drive(mgis, data, path)

        elastic = data.s1.internal_state_variables[0, ELASTIC_STRAIN]
        axial_plastic = float(
            (axial - elastic[2]) - 0.5 * ((-0.5 * axial) - elastic[0])
            - 0.5 * ((-0.5 * axial) - elastic[1])
        ) / 1.5
        slips = np.abs(data.s1.internal_state_variables[0, PLASTIC_SLIP])
        measured = slips[slips > 1e-14].mean()

        assert measured == pytest.approx(
            SLIP_PER_EQUIVALENT_RATE_001 * axial_plastic, rel=1e-6
        )

    def test_the_plateau_converges_monotonically_under_refinement(self) -> None:
        """Section 8.1's last requirement, on the analytical case."""

        expected = uniaxial_001_plateau_stress(
            tau0_mpa=40.0, overstress_modulus_mpa=18.7819100705
        )
        errors = []
        for steps in (10, 20, 40, 80):
            mgis, data = _manager(parameters=_no_hardening(18.7819100705, 40.0))
            path = np.concatenate(
                [_uniaxial_001(0.02 * (index + 1) / steps) for index in range(steps)]
            ).reshape(steps, 1, 6)
            _drive(mgis, data, path)
            errors.append(abs(_axial_deviatoric_stress(data) - expected))

        assert errors == sorted(errors, reverse=True) or max(errors) < 1e-9


# ---------------------------------------------------------------------------
# Section 9.1 -- dissipation.
# ---------------------------------------------------------------------------


class TestDissipationAlgebra:
    def test_the_product_is_non_negative_by_construction(self) -> None:
        increments = slip_dissipation_increments(
            resolved_stress_mpa=[10.0, -10.0, 0.0],
            back_stress_mpa=[0.0, 0.0, 0.0],
            slip_increment=[1e-3, -1e-3, 0.0],
        )

        assert np.all(increments >= 0.0)

    def test_a_sign_error_would_show_as_negative_dissipation(self) -> None:
        """The failure mode the check exists for, exhibited deliberately."""

        increments = slip_dissipation_increments(
            resolved_stress_mpa=[10.0],
            back_stress_mpa=[0.0],
            slip_increment=[-1e-3],
        )

        assert increments[0] < 0.0

    def test_mismatched_shapes_are_refused(self) -> None:
        with pytest.raises(ValueError, match="share a shape"):
            slip_dissipation_increments(
                resolved_stress_mpa=[1.0, 2.0],
                back_stress_mpa=[0.0],
                slip_increment=[0.0],
            )

    def test_the_kinematic_stored_energy_is_the_recoverable_part(self) -> None:
        """Not the integral of `X dgamma`: dynamic recovery dissipates the rest."""

        assert kinematic_stored_energy(back_strain=[1e-3, -2e-3], c_mpa=40_000.0) == (
            pytest.approx(0.5 * 40_000.0 * (1e-6 + 4e-6))
        )

    def test_a_negative_kinematic_modulus_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nonnegative"):
            kinematic_stored_energy(back_strain=[0.0], c_mpa=-1.0)

    def test_the_balance_residual_is_what_is_unaccounted_for(self) -> None:
        balance = EnergyBalance(
            total_work=10.0,
            elastic_energy=4.0,
            stored_isotropic=1.0,
            stored_kinematic=2.0,
            plastic_dissipation=3.0,
        )

        assert balance.residual == pytest.approx(0.0)
        assert set(balance.as_dict()) >= {"total_work", "residual"}


@pytest.mark.mfront
class TestDissipationOnPaths:
    @pytest.mark.parametrize("overstress", [1.0, 8.0, 18.7819100705])
    def test_no_system_dissipates_negatively_on_a_monotonic_path(
        self, overstress: float
    ) -> None:
        """Section 9.1, on the full hardening law rather than the reduced one."""

        preset = get_parameter_set(HISTORICAL)
        mgis, data = _manager(parameters={"SrixOverstressModulus": overstress})
        steps = 60
        for index in range(steps):
            previous = data.s1.internal_state_variables[0, PLASTIC_SLIP].copy()
            data.s1.gradients[:, :] = _uniaxial_001(0.03 * (index + 1) / steps)
            mgis.integrate(
                data,
                mgis.IntegrationType.IntegrationWithConsistentTangentOperator,
                1.0,
                0,
                1,
            )
            mgis.update(data)
            increment = data.s1.internal_state_variables[0, PLASTIC_SLIP] - previous
            back = preset.c_mpa * data.s1.internal_state_variables[0, BACK_STRAIN]
            resolved = _resolved_stresses(data)
            dissipation = slip_dissipation_increments(
                resolved_stress_mpa=resolved,
                back_stress_mpa=back,
                slip_increment=increment,
            )
            assert dissipation.min() > -1e-12 * max(
                float(np.abs(dissipation).max()), 1.0
            )

    def test_no_system_dissipates_negatively_through_a_reversal(self) -> None:
        """The case that actually exercises the back stress."""

        preset = get_parameter_set(HISTORICAL)
        mgis, data = _manager()
        forward = [0.02 * (index + 1) / 40 for index in range(40)]
        backward = [0.02 - 0.04 * (index + 1) / 40 for index in range(40)]
        for axial in forward + backward:
            previous = data.s1.internal_state_variables[0, PLASTIC_SLIP].copy()
            data.s1.gradients[:, :] = _uniaxial_001(axial)
            mgis.integrate(
                data,
                mgis.IntegrationType.IntegrationWithConsistentTangentOperator,
                1.0,
                0,
                1,
            )
            mgis.update(data)
            increment = data.s1.internal_state_variables[0, PLASTIC_SLIP] - previous
            back = preset.c_mpa * data.s1.internal_state_variables[0, BACK_STRAIN]
            dissipation = slip_dissipation_increments(
                resolved_stress_mpa=_resolved_stresses(data),
                back_stress_mpa=back,
                slip_increment=increment,
            )
            assert dissipation.min() > -1e-9 * max(float(np.abs(dissipation).max()), 1.0)


def _resolved_stresses(data: Any) -> np.ndarray:
    """`tau_s = sigma : m_s`, rebuilt from the Kelvin stress and the Schmid tensors."""

    from fem_inhouse.core.fcc_interaction_matrix import slip_systems

    stress = np.asarray(data.s1.thermodynamic_forces[0], dtype=float)
    root_two = math.sqrt(2.0)
    tensor = np.array(
        [
            [stress[0], stress[3] / root_two, stress[4] / root_two],
            [stress[3] / root_two, stress[1], stress[5] / root_two],
            [stress[4] / root_two, stress[5] / root_two, stress[2]],
        ]
    )
    resolved = []
    for system in slip_systems():
        burgers = system.burgers / np.linalg.norm(system.burgers)
        normal = system.normal / np.linalg.norm(system.normal)
        schmid = 0.5 * (np.outer(burgers, normal) + np.outer(normal, burgers))
        resolved.append(float(np.sum(tensor * schmid)))
    return np.array(resolved)


# ---------------------------------------------------------------------------
# Section 9.2 -- the overstress diagnostic.
# ---------------------------------------------------------------------------


class TestOverstressDiagnostic:
    def test_an_elastic_state_reports_nothing_active(self) -> None:
        diagnostic = overstress_diagnostic(
            resolved_stress_mpa=np.full(12, 10.0),
            back_stress_mpa=np.zeros(12),
            critical_resistance_mpa=np.full(12, 40.0),
        )

        assert diagnostic.active_count == 0
        assert diagnostic.maximum == 0.0
        assert diagnostic.mean_active == 0.0
        assert diagnostic.fraction_above_1pc == 0.0

    def test_the_ratio_is_relative_to_the_resistance(self) -> None:
        diagnostic = overstress_diagnostic(
            resolved_stress_mpa=np.full(12, 44.0),
            back_stress_mpa=np.zeros(12),
            critical_resistance_mpa=np.full(12, 40.0),
        )

        assert diagnostic.maximum == pytest.approx(0.1)
        assert diagnostic.active_count == 12
        assert diagnostic.fraction_above_5pc == 1.0
        assert diagnostic.fraction_above_10pc == 0.0

    def test_the_back_stress_shifts_the_centre(self) -> None:
        diagnostic = overstress_diagnostic(
            resolved_stress_mpa=np.full(4, 44.0),
            back_stress_mpa=np.full(4, 44.0),
            critical_resistance_mpa=np.full(4, 40.0),
        )

        assert diagnostic.active_count == 0

    def test_the_quantiles_order_correctly(self) -> None:
        eta = np.linspace(0.0, 0.2, 12)
        diagnostic = overstress_diagnostic(
            resolved_stress_mpa=40.0 * (1.0 + eta),
            back_stress_mpa=np.zeros(12),
            critical_resistance_mpa=np.full(12, 40.0),
        )

        assert diagnostic.q95 <= diagnostic.maximum
        assert diagnostic.q95 <= diagnostic.q99 <= diagnostic.maximum

    def test_mismatched_shapes_are_refused(self) -> None:
        with pytest.raises(ValueError, match="share a shape"):
            overstress_diagnostic(
                resolved_stress_mpa=np.zeros(3),
                back_stress_mpa=np.zeros(3),
                critical_resistance_mpa=np.zeros(4),
            )


@pytest.mark.mfront
def test_the_measured_overstress_matches_the_closed_form_on_the_plateau() -> None:
    """Section 9.2 tied back to section 8.1: the diagnostic reads `O_R`."""

    mgis, data = _manager(parameters=_no_hardening(18.7819100705, 40.0))
    steps = 200
    path = np.concatenate(
        [_uniaxial_001(0.02 * (index + 1) / steps) for index in range(steps)]
    ).reshape(steps, 1, 6)
    _drive(mgis, data, path)

    diagnostic = overstress_diagnostic(
        resolved_stress_mpa=_resolved_stresses(data),
        back_stress_mpa=np.zeros(12),
        critical_resistance_mpa=np.full(12, 40.0),
    )
    expected = uniaxial_001_relative_overstress(
        tau0_mpa=40.0, overstress_modulus_mpa=18.7819100705
    )

    assert diagnostic.active_count == ACTIVE_SYSTEMS_001
    assert diagnostic.maximum == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Sections 9.3 and 9.4 -- time.
# ---------------------------------------------------------------------------


@pytest.mark.mfront
class TestTimeIndependence:
    @staticmethod
    def _final_state(time_increments: np.ndarray) -> tuple[float, np.ndarray]:
        mgis, data = _manager()
        steps = time_increments.size
        path = np.concatenate(
            [_uniaxial_001(0.02 * (index + 1) / steps) for index in range(steps)]
        ).reshape(steps, 1, 6)
        _drive(mgis, data, path, time_increments=time_increments)
        return (
            _axial_deviatoric_stress(data),
            data.s1.internal_state_variables[0, PLASTIC_SLIP].copy(),
        )

    def test_uniform_and_non_uniform_pseudo_time_agree_bit_for_bit(self) -> None:
        """Section 9.3. The strain discretisation is identical; only `dt` differs."""

        steps = 40
        uniform = np.ones(steps)
        ramped = np.linspace(0.1, 5.0, steps)
        generator = np.random.default_rng(7)
        shuffled = generator.uniform(0.05, 3.0, steps)

        reference = self._final_state(uniform)
        for variant in (ramped, shuffled):
            stress, slips = self._final_state(variant)
            assert stress == reference[0]
            np.testing.assert_array_equal(slips, reference[1])

    def test_a_thousandfold_change_of_rate_changes_nothing(self) -> None:
        steps = 30
        slow = self._final_state(np.full(steps, 1000.0))
        fast = self._final_state(np.full(steps, 0.001))

        assert slow[0] == fast[0]
        np.testing.assert_array_equal(slow[1], fast[1])


@pytest.mark.mfront
class TestTimeConvergence:
    """Section 9.4. Refining the strain discretisation, monotonic and reversed.

    The specification asks for convergence *on the last refinements*, and that
    wording turns out to matter: the reversal has a step size below which it is
    qualitatively wrong, not merely inaccurate. See
    `test_a_coarse_reversal_misses_the_reverse_yield_point`.
    """

    PEAK = 0.01
    REVERSAL = 0.006

    @staticmethod
    def _monotonic(steps: int) -> dict[str, Any]:
        mgis, data = _manager()
        for index in range(steps):
            data.s1.gradients[:, :] = _uniaxial_001(
                TestTimeConvergence.PEAK * (index + 1) / steps
            )
            mgis.integrate(
                data,
                mgis.IntegrationType.IntegrationWithConsistentTangentOperator,
                1.0,
                0,
                1,
            )
            mgis.update(data)
        return _state(data)

    @staticmethod
    def _reversed(steps: int) -> dict[str, Any]:
        preset = get_parameter_set(HISTORICAL)
        mgis, data = _manager()
        half = steps // 2
        peak, back = TestTimeConvergence.PEAK, TestTimeConvergence.REVERSAL
        path = [peak * (i + 1) / half for i in range(half)]
        path += [peak + (back - peak) * (i + 1) / half for i in range(half)]
        dissipation = 0.0
        for axial in path:
            previous = data.s1.internal_state_variables[0, PLASTIC_SLIP].copy()
            data.s1.gradients[:, :] = _uniaxial_001(axial)
            mgis.integrate(
                data,
                mgis.IntegrationType.IntegrationWithConsistentTangentOperator,
                1.0,
                0,
                1,
            )
            mgis.update(data)
            dissipation += float(
                slip_dissipation_increments(
                    resolved_stress_mpa=_resolved_stresses(data),
                    back_stress_mpa=preset.c_mpa
                    * data.s1.internal_state_variables[0, BACK_STRAIN],
                    slip_increment=data.s1.internal_state_variables[0, PLASTIC_SLIP]
                    - previous,
                ).sum()
            )
        state = _state(data)
        state["dissipation"] = dissipation
        return state

    @staticmethod
    def _errors(states: dict[int, dict[str, Any]], counts: tuple[int, ...], key: str):
        finest = np.atleast_1d(np.asarray(states[counts[-1]][key], dtype=float))
        scale = max(float(np.abs(finest).max()), 1e-30)
        return [
            float(
                np.abs(
                    np.atleast_1d(np.asarray(states[count][key], dtype=float)) - finest
                ).max()
                / scale
            )
            for count in counts[:-1]
        ]

    def test_the_monotonic_branch_converges_at_first_order(self) -> None:
        """Clean, from the coarsest step: nothing qualitative happens here."""

        counts = (10, 20, 40, 80, 160)
        states = {count: self._monotonic(count) for count in counts}

        for key in ("stress", "slip", "equivalent"):
            errors = self._errors(states, counts, key)
            assert errors == sorted(errors, reverse=True), (key, errors)
            assert errors[-1] < 1e-3, (key, errors)

    def test_a_coarse_reversal_misses_the_reverse_yield_point(self) -> None:
        """A finding, recorded rather than tuned away -- and its threshold moved.

        Below a certain number of increments over the whole path, the reversal
        produces essentially **no reverse slip**: the total slip is what the
        forward branch left, and the back strain stays at its forward value
        instead of relaxing. That is a qualitative failure, not a large error,
        and it is why the specification asks for monotonicity on the last
        refinements rather than on all of them.

        The threshold used to be twenty increments, with exact equality against
        the forward branch. Making the symmetric Clarke element canonical at
        zero slip increment (`d|dg|/ddg = 0`) **halved it**. Measured total
        slip against `0.0204501740` for forward-only loading:

        ==============  ==============  =====================
        increments      total slip      gap to forward-only
        ==============  ==============  =====================
        10              0.0204799594    2.98e-05
        20              0.0171293670    3.32e-03
        40              0.0171415951    3.31e-03
        80              0.0171555376    3.30e-03
        ==============  ==============  =====================

        Twenty increments now resolve the reversal, and agree with eighty to
        `1e-4`. Ten still miss it, but no longer exactly: the gap is `0.15 %`
        of the forward slip rather than zero, so this asserts a tight relative
        bound instead of equality.

        A campaign that reverses its loading must still check its increment
        count against this, not assume that fewer increments merely cost
        accuracy. The margin is simply wider than it was.
        """

        coarse = self._reversed(10)
        resolved = self._reversed(80)
        forward_only = self._monotonic(10)

        assert np.abs(coarse["slip"]).sum() == pytest.approx(
            np.abs(forward_only["slip"]).sum(), rel=2e-3
        )
        assert np.abs(resolved["slip"]).sum() < 0.95 * np.abs(coarse["slip"]).sum()
        assert np.abs(resolved["back"]).max() < 0.2 * np.abs(coarse["back"]).max()

    def test_the_reversal_converges_once_the_yield_point_is_resolved(self) -> None:
        """Section 9.4 proper, over the range where the solution is the same one."""

        counts = (40, 80, 160)
        states = {count: self._reversed(count) for count in counts}

        for key, tolerance in (
            ("stress", 1e-2),
            ("slip", 1e-2),
            ("equivalent", 1e-2),
            ("back", 1e-1),
            ("dissipation", 1e-2),
        ):
            errors = self._errors(states, counts, key)
            assert errors == sorted(errors, reverse=True), (key, errors)
            assert errors[-1] < tolerance, (key, errors)


def _state(data: Any) -> dict[str, Any]:
    return {
        "stress": _axial_deviatoric_stress(data),
        "slip": data.s1.internal_state_variables[0, PLASTIC_SLIP].copy(),
        "equivalent": data.s1.internal_state_variables[0, EQUIVALENT_SLIP].copy(),
        "back": data.s1.internal_state_variables[0, BACK_STRAIN].copy(),
    }


@pytest.mark.mfront
class TestSharedBehaviourHandles:
    """MGIS hands out shared behaviours, and the bridge has to cope.

    Discovered while writing this file: two tests that each loaded "their own"
    behaviour were changing each other's parameters, and the symptom looked like
    physics. The same trap applies to the solver, where two batches with
    different parameter sets would otherwise share whichever was applied last.
    """

    def test_two_loads_return_the_same_underlying_behaviour(self) -> None:
        """The fact everything below follows from, asserted rather than assumed."""

        mgis, library = _mgis()
        first = mgis.load(library, SRIX, mgis.Hypothesis.Tridimensional)
        second = mgis.load(library, SRIX, mgis.Hypothesis.Tridimensional)

        mgis.setParameter(first, "tau0", 40.0)
        mgis.setParameter(first, "b", 0.0)
        mgis.setParameter(first, "C", 0.0)
        mgis.setParameter(first, "SrixOverstressModulus", 18.7819100705)
        reference = _plateau_through(mgis, first)

        mgis.setParameter(second, "tau0", 80.0)
        after = _plateau_through(mgis, first)

        assert after != pytest.approx(reference, rel=1e-9)
        assert after == pytest.approx(
            uniaxial_001_plateau_stress(
                tau0_mpa=80.0, overstress_modulus_mpa=18.7819100705
            ),
            rel=1e-9,
        )

    def test_the_bridge_reasserts_its_parameters_before_integrating(self) -> None:
        """Two batches, two parameter sets, same process: both stay correct.

        Without the re-assertion the second batch's set would win for both, and
        the first batch would quietly report the wrong material.
        """

        library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
        if not library:
            pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
        pytest.importorskip("mgis.behaviour")
        from fem_inhouse.core.plane_stress_material import (
            create_plane_stress_material_batch,
        )

        def batch(parameter_set: str) -> Any:
            return create_plane_stress_material_batch(
                "mfront-3d-condensed-plane-stress",
                np.full((1, 1), 250.0),
                np.full((1, 1), 500.0),
                0.245,
                young_modulus_mpa=205000.0,
                poisson_ratio=0.3,
                hardening_mode="ludwik",
                plastic_strain_max=0.2,
                plastic_table_points=1000,
                first_positive_plastic_strain=1e-6,
                mfront_library=library,
                mfront_threads=1,
                mfront_behaviour_id="fcc_forest_rubin_srix",
                constitutive_options={"parameter_set": parameter_set},
            )

        def drive(target: Any) -> float:
            for index in range(30):
                trial = target.evaluate(
                    np.array([[0.01 * (index + 1) / 30, -0.004 * (index + 1) / 30, 0.0]]),
                    time_increment=1.0,
                )
                target.commit()
            return float(trial.stress_in_plane_mpa[0, 0])

        soft = batch("316l_srix_exploratory_r1")
        stiff = batch("316l_srix_transposed_from_nasri2018_rate_1e-3")

        # Interleaved on purpose: each evaluate must restore its own set.
        soft_alone = drive(soft)
        stiff_alone = drive(stiff)
        soft_again = drive(batch("316l_srix_exploratory_r1"))

        assert soft_alone != pytest.approx(stiff_alone, rel=1e-6)
        assert soft_again == pytest.approx(soft_alone, rel=1e-12)


def _plateau_through(mgis: Any, behaviour: Any) -> float:
    data = mgis.MaterialDataManager(behaviour, 1)
    for state in (data.s0, data.s1):
        mgis.setExternalStateVariable(state, "Temperature", 293.15)
        _set_local_coupling(mgis, behaviour, state)
    for index in range(60):
        data.s1.gradients[:, :] = _uniaxial_001(0.03 * (index + 1) / 60)
        mgis.integrate(
            data, mgis.IntegrationType.IntegrationWithConsistentTangentOperator, 1.0, 0, 1
        )
        mgis.update(data)
    return _axial_deviatoric_stress(data)
