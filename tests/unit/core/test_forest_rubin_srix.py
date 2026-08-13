"""Acceptance tests for the Forest-Rubin SRIX rate-independent crystal law.

The tests that touch MGIS skip without ``MFRONT_BEHAVIOUR_LIBRARY``, following
the convention of ``test_mfront.py``. The conversion utility is pure and always
runs.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from fem_inhouse.core import single_crystal_presets
from fem_inhouse.core.mfront_behaviours import MFRONT_BEHAVIOURS
from fem_inhouse.core.single_crystal_presets import (
    get_preset,
    get_srix_preset,
    srix_overstress_modulus_from_meric,
)

SRIX = "Fcc316LForestRubinSrix"
MERIC_CAILLETAUD = "Fcc316LMericCailletaud"

#: Offsets of the internal state variables, verified by
#: test_the_internal_state_layout_is_what_the_catalogue_declares.
ELASTIC_STRAIN = slice(0, 6)
PLASTIC_SLIP = slice(6, 18)
EQUIVALENT_SLIP = slice(18, 30)
BACK_STRAIN = slice(30, 42)

#: sqrt(2) appears wherever a Kelvin shear component is built from a tensor one.
ROOT_TWO = np.sqrt(2.0)


def _mgis() -> Any:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if not library:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    try:
        import mgis.behaviour as mgis
    except ImportError:  # pragma: no cover - exercised only without MGIS
        pytest.skip("mgis.behaviour is not importable")
    return mgis, library


#: Loaded behaviours, kept alive for the whole session.
#:
#: A MaterialDataManager holds a C++ pointer to its behaviour but no Python
#: reference to it. Letting the behaviour returned by mgis.load fall out of
#: scope frees it under a manager that is still using it, and the process
#: segfaults later, in an unrelated test. Caching also stops the library being
#: reopened once per material point.
_BEHAVIOURS: dict[tuple[str, str], Any] = {}


def _manager(behaviour_name: str, points: int = 1) -> Any:
    mgis, library = _mgis()
    key = (library, behaviour_name)
    behaviour = _BEHAVIOURS.get(key)
    if behaviour is None:
        behaviour = mgis.load(library, behaviour_name, mgis.Hypothesis.Tridimensional)
        _BEHAVIOURS[key] = behaviour
    data = mgis.MaterialDataManager(behaviour, points)
    for state in (data.s0, data.s1):
        mgis.setExternalStateVariable(state, "Temperature", 293.15)
        _set_local_coupling(mgis, behaviour, state)
    return mgis, data


def _set_local_coupling(mgis: Any, behaviour: Any, state: Any) -> None:
    """Neutralise the scalar micromorphic extension when the law declares it.

    MGIS gives neither `@MaterialProperty` nor `@ExternalStateVariable` a
    default: once the non-local extension added `Hchi` and the external field,
    every direct-MGIS test here died inside `buildEvaluators` on the first
    integration rather than at construction, which is what kept the omission
    invisible. Both zeros are documented values, not arbitrary filler -- at
    `Hchi = 0` the law reduces exactly to the historical local SRIX response,
    the only response these tests are about.
    """

    for variable in behaviour.mps:
        if variable.name == "MicromorphicCouplingModulus":
            mgis.setMaterialProperty(state, variable.name, 0.0)
    for variable in behaviour.esvs:
        if variable.name == "NonlocalEquivalentPlasticStrain":
            mgis.setExternalStateVariable(state, variable.name, 0.0)


def _integrate(mgis: Any, data: Any, strain: np.ndarray, time_increment: float) -> bool:
    """Integrate one increment, reporting non-convergence as False.

    Only RuntimeError is caught: that is how MGIS reports a behaviour that
    declined the step, and it is a legitimate outcome several tests assert on.
    Anything else is a defect in the test and must surface.
    """

    points = strain.shape[0]
    data.s1.gradients[:, :] = strain
    integration = mgis.IntegrationType.IntegrationWithConsistentTangentOperator
    try:
        return bool(mgis.integrate(data, integration, time_increment, 0, points) == 1)
    except RuntimeError:
        return False


def _isochoric_axial(direction: tuple[float, float, float], magnitude: float) -> np.ndarray:
    """Kelvin strain of an isochoric extension along ``direction``.

    Its von Mises equivalent equals ``magnitude`` exactly, which makes the
    imposed equivalent strain increment readable straight off the argument.
    """

    axis = np.asarray(direction, dtype=float)
    axis /= np.linalg.norm(axis)
    projector = np.outer(axis, axis)
    tensor = magnitude * (projector - 0.5 * (np.eye(3) - projector))
    return np.array(
        [
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            ROOT_TWO * tensor[0, 1],
            ROOT_TWO * tensor[0, 2],
            ROOT_TWO * tensor[1, 2],
        ]
    )


def _axis(direction: tuple[float, float, float]) -> np.ndarray:
    axis = np.asarray(direction, dtype=float)
    return axis / np.linalg.norm(axis)


def _axial_stress(stress_kelvin: np.ndarray, axis: np.ndarray) -> float:
    """Stress resolved along the loading axis, not the x component.

    For anything but [001] these differ, and comparing raw Kelvin components
    across orientations compares different physical quantities.
    """

    shear = stress_kelvin[3:] / ROOT_TWO
    tensor = np.array(
        [
            [stress_kelvin[0], shear[0], shear[1]],
            [shear[0], stress_kelvin[1], shear[2]],
            [shear[1], shear[2], stress_kelvin[2]],
        ]
    )
    return float(axis @ tensor @ axis)


def _ramp(
    behaviour_name: str,
    final_strain: np.ndarray,
    *,
    steps: int = 20,
    total_time: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk a linear strain path and return stress, internal state and tangent."""

    mgis, data = _manager(behaviour_name)
    for step in range(1, steps + 1):
        strain = final_strain * step / steps
        assert _integrate(mgis, data, strain[None, :], total_time / steps), (
            f"{behaviour_name} failed to integrate step {step} of {steps}"
        )
        mgis.update(data)
    return (
        data.s1.thermodynamic_forces[0].copy(),
        data.s1.internal_state_variables[0].copy(),
        data.K[0].copy(),
    )


# --------------------------------------------------------------------------
# Section 9.1 - the conversion utility
# --------------------------------------------------------------------------


def test_equation_sixteen_reproduces_the_published_value() -> None:
    """Forest and Rubin (2016), equation (16), at the specified operating point."""

    value = srix_overstress_modulus_from_meric(
        norton_strength_mpa=12.0,
        norton_exponent=11.0,
        reference_strain_rate=1.0e-3,
    )

    assert value == pytest.approx(18.7819100705, abs=1e-9)


def test_the_old_name_still_works_and_says_it_is_deprecated() -> None:
    """Section 3.2. The rename must not silently break an existing script.

    `srix_reference_stress` named what the number was used for, not where it
    came from, so reading it in a manifest gave no hint that the value was
    transposed from a rate-dependent law rather than measured.
    """

    arguments = {
        "norton_strength_mpa": 12.0,
        "norton_exponent": 11.0,
        "reference_strain_rate": 1.0e-3,
    }

    with pytest.warns(DeprecationWarning, match="srix_overstress_modulus_from_meric"):
        legacy = single_crystal_presets.srix_reference_stress(**arguments)

    assert legacy == srix_overstress_modulus_from_meric(**arguments)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"norton_strength_mpa": 0.0}, "norton_strength_mpa"),
        ({"norton_strength_mpa": -12.0}, "norton_strength_mpa"),
        ({"norton_exponent": 0.0}, "norton_exponent"),
        ({"norton_exponent": -11.0}, "norton_exponent"),
        ({"reference_strain_rate": 0.0}, "reference_strain_rate"),
        ({"reference_strain_rate": -1.0e-3}, "reference_strain_rate"),
    ],
)
def test_the_conversion_rejects_unphysical_arguments(
    kwargs: dict[str, float], message: str
) -> None:
    arguments = {
        "norton_strength_mpa": 12.0,
        "norton_exponent": 11.0,
        "reference_strain_rate": 1.0e-3,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        srix_overstress_modulus_from_meric(**arguments)  # type: ignore[arg-type]


def test_a_faster_reference_rate_needs_a_larger_modulus() -> None:
    """R absorbs the rate the viscous law was frozen at; it is not a constant."""

    slow = srix_overstress_modulus_from_meric(
        norton_strength_mpa=12.0, norton_exponent=11.0, reference_strain_rate=1e-4
    )
    fast = srix_overstress_modulus_from_meric(
        norton_strength_mpa=12.0, norton_exponent=11.0, reference_strain_rate=1e-2
    )

    assert slow < fast


# --------------------------------------------------------------------------
# Section 5 - the preset
# --------------------------------------------------------------------------


def test_the_srix_preset_drops_the_rate_dependent_parameters() -> None:
    preset = get_srix_preset("316l_forest_rubin_srix_from_nasri2018")

    parameters = preset.mfront_parameters()

    assert "K" not in parameters
    assert "n" not in parameters
    assert parameters["R"] == pytest.approx(18.7819100705, abs=1e-9)
    # The hardening is inherited untouched, not restated.
    source = get_preset("316l_guilhem2013_nasri2018").mfront_parameters()
    for shared in ("tau0", "Q", "b", "C", "d"):
        assert parameters[shared] == source[shared]


def test_the_preset_inherits_rather_than_copies_the_parent() -> None:
    """A copied interaction matrix could drift; a delegated one cannot."""

    preset = get_srix_preset("316l_forest_rubin_srix_from_nasri2018")
    source = get_preset("316l_guilhem2013_nasri2018")

    assert preset.interaction_matrix is source.interaction_matrix
    assert preset.elasticity is source.elasticity


def test_the_provenance_names_the_transposition_as_such() -> None:
    record = get_srix_preset("316l_forest_rubin_srix_from_nasri2018").provenance_record()

    assert "NOT an identification" in record["status"]
    assert record["reference_strain_rate"] == 1.0e-3
    assert record["norton_strength_mpa"] == 12.0
    assert record["norton_exponent"] == 11.0
    assert "euromechsol" in record["srix_reference"]
    assert "crme" in record["hardening_reference"]


def test_an_unknown_srix_preset_names_the_registered_ones() -> None:
    with pytest.raises(KeyError, match="316l_forest_rubin_srix_from_nasri2018"):
        get_srix_preset("nope")


# --------------------------------------------------------------------------
# Section 6 - catalogue registration
# --------------------------------------------------------------------------


def test_the_catalogue_declares_both_crystal_laws() -> None:
    for identifier, expected in (
        ("fcc_forest_rubin_srix", "Fcc316LForestRubinSrix"),
        ("fcc_meric_cailletaud", "Fcc316LMericCailletaud"),
    ):
        specification = MFRONT_BEHAVIOURS.get(identifier)
        assert specification.tridimensional_behaviour == expected
        # A crystal has no native plane-stress hypothesis: the 3D law is
        # condensed instead, which is what section 8 requires.
        assert specification.native_plane_stress_behaviour is None
        assert specification.linear_system_matrix_type == "nonsymmetric"
        assert specification.requires_rotation_matrix
        declared = {item.canonical_name for item in specification.internal_state_variables}
        assert "equivalent_plastic_strain" not in declared
        assert "yield_surface_radius_mpa" not in declared


def test_the_internal_state_layout_is_what_the_catalogue_declares() -> None:
    mgis, library = _mgis()
    behaviour = mgis.load(library, SRIX, mgis.Hypothesis.Tridimensional)

    names = [variable.name for variable in behaviour.isvs]

    assert names[0] == "ElasticStrain"
    assert names[6].startswith("PlasticSlip")
    assert names[18].startswith("EquivalentPlasticSlip")
    assert names[30].startswith("BackStrain")
    assert len(names) == 1 + 3 * 12


# --------------------------------------------------------------------------
# Section 9.3 - the fundamental test: no dependence on time
# --------------------------------------------------------------------------


def test_srix_is_time_independent() -> None:
    """The same strain path traversed over times differing by 1e6.

    Equality is required to the bit, not to a tolerance. There is no dt in the
    law, so there is no mechanism by which a difference could appear; anything
    non-zero means dt has crept back in.
    """

    path = _isochoric_axial((1.0, 0.0, 0.0), 6.0e-3)
    stress, state, tangent = _ramp(SRIX, path, total_time=1.0e-2)

    for total_time in (1.0, 1.0e2, 1.0e4):
        other_stress, other_state, other_tangent = _ramp(SRIX, path, total_time=total_time)
        assert np.array_equal(other_stress, stress)
        assert np.array_equal(other_state, state)
        assert np.array_equal(other_tangent, tangent)


def test_meric_cailletaud_is_not_time_independent() -> None:
    """The control. Without it, the test above could pass on a broken harness."""

    path = _isochoric_axial((1.0, 0.0, 0.0), 6.0e-3)
    fast, _, _ = _ramp(MERIC_CAILLETAUD, path, total_time=1.0e-2)
    slow, _, _ = _ramp(MERIC_CAILLETAUD, path, total_time=1.0e2)

    assert abs(fast[0] - slow[0]) > 1.0


# --------------------------------------------------------------------------
# Section 4.3 - degenerate increments
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "strain"),
    [
        ("null", np.zeros(6)),
        ("hydrostatic", np.array([1e-3, 1e-3, 1e-3, 0.0, 0.0, 0.0])),
    ],
)
def test_a_degenerate_increment_produces_no_slip(label: str, strain: np.ndarray) -> None:
    """Neither a null nor a purely hydrostatic increment is deviatoric.

    The equivalent strain increment is zero in both cases, so no slip may
    appear, no internal variable may move and nothing may divide by zero.
    """

    mgis, data = _manager(SRIX)

    assert _integrate(mgis, data, strain[None, :], 1.0)

    state = data.s1.internal_state_variables[0]
    assert np.isfinite(state).all()
    assert np.isfinite(data.s1.thermodynamic_forces[0]).all()
    assert np.abs(state[PLASTIC_SLIP]).max() == 0.0
    assert np.abs(state[EQUIVALENT_SLIP]).max() == 0.0
    assert np.abs(state[BACK_STRAIN]).max() == 0.0


def test_the_hydrostatic_increment_is_purely_elastic() -> None:
    """No slip, but a real pressure: the branch must not swallow the stress."""

    mgis, data = _manager(SRIX)
    strain = np.array([1e-3, 1e-3, 1e-3, 0.0, 0.0, 0.0])

    assert _integrate(mgis, data, strain[None, :], 1.0)

    stress = data.s1.thermodynamic_forces[0]
    assert stress[0] > 100.0
    assert stress[0] == pytest.approx(stress[1])
    assert stress[0] == pytest.approx(stress[2])


# --------------------------------------------------------------------------
# Section 9.2 - material-point behaviour
# --------------------------------------------------------------------------


def test_a_small_increment_stays_elastic_and_reversible() -> None:
    mgis, data = _manager(SRIX)
    strain = _isochoric_axial((1.0, 0.0, 0.0), 1.0e-5)

    assert _integrate(mgis, data, strain[None, :], 1.0)
    state = data.s1.internal_state_variables[0]

    assert np.abs(state[PLASTIC_SLIP]).max() == 0.0
    # Purely elastic: the elastic strain absorbs the whole increment.
    assert state[ELASTIC_STRAIN] == pytest.approx(strain, abs=1e-18)


@pytest.mark.parametrize(
    ("label", "direction", "expected_active"),
    [("[001]", (1.0, 0.0, 0.0), 8), ("[111]", (1.0, 1.0, 1.0), 6), ("[123]", (1.0, 2.0, 3.0), 8)],
)
def test_the_active_slip_systems_follow_the_orientation(
    label: str, direction: tuple[float, float, float], expected_active: int
) -> None:
    """Symmetry of the orientation dictates how many systems carry the load."""

    _, state, _ = _ramp(SRIX, _isochoric_axial(direction, 6.0e-3))

    active = int((np.abs(state[PLASTIC_SLIP]) > 1e-12).sum())
    assert active == expected_active


def test_the_slips_and_the_plastic_strain_agree() -> None:
    """Section 9.2, item 11, plus the positivity and finiteness requirements."""

    path = _isochoric_axial((1.0, 2.0, 3.0), 6.0e-3)
    _, state, _ = _ramp(SRIX, path)

    slips = state[PLASTIC_SLIP]
    equivalent = state[EQUIVALENT_SLIP]
    plastic_strain = path - state[ELASTIC_STRAIN]

    assert np.isfinite(state).all()
    # p accumulates |Dg| and can only grow.
    assert (equivalent >= 0.0).all()
    assert equivalent.sum() == pytest.approx(np.abs(slips).sum(), rel=1e-9)
    # Slip is isochoric, so the plastic strain it produces must be too.
    assert plastic_strain[:3].sum() == pytest.approx(0.0, abs=1e-15)
    assert np.abs(plastic_strain).max() > 1e-4


def test_unloading_is_smooth_rather_than_abrupt() -> None:
    """SRIX has no sharp elastic-plastic threshold, and that is the point.

    The flow rule is an overstress law with no loading-unloading switch: as long
    as f stays positive the systems keep sliding, so a small reversal still
    produces a little slip. It stops once the overstress has collapsed. This is
    the smooth transition named in the title of Forest and Rubin (2016), and it
    is the behaviour that removes the slip indeterminacy of the classical
    rate-independent formulation.

    Meric-Cailletaud does the same thing for the same reason; the difference
    between the two laws is the rate dependence, not the transition.
    """

    peak = _isochoric_axial((1.0, 0.0, 0.0), 6.0e-3)
    residual_slip = {}
    for fraction in (0.999, 0.99, 0.95):
        mgis, data = _manager(SRIX)
        for step in range(1, 21):
            assert _integrate(mgis, data, (peak * step / 20)[None, :], 1.0)
            mgis.update(data)
        at_peak = data.s1.internal_state_variables[0][PLASTIC_SLIP].copy()
        accumulated = np.abs(at_peak).max()

        assert _integrate(mgis, data, (peak * fraction)[None, :], 1.0)

        unloaded = data.s1.internal_state_variables[0][PLASTIC_SLIP]
        residual_slip[fraction] = np.abs(unloaded - at_peak).max() / accumulated

    # A whisker of reversal still slips, but by a negligible fraction.
    assert 0.0 < residual_slip[0.999] < 0.01
    # Unload far enough and the overstress is gone, so slip stops exactly.
    assert residual_slip[0.95] == 0.0


def test_reversing_the_load_reverses_the_slips() -> None:
    mgis, data = _manager(SRIX)
    peak = _isochoric_axial((1.0, 0.0, 0.0), 6.0e-3)
    for step in range(1, 21):
        assert _integrate(mgis, data, (peak * step / 20)[None, :], 1.0)
        mgis.update(data)
    forward = data.s1.internal_state_variables[0][PLASTIC_SLIP].copy()
    forward_equivalent = data.s1.internal_state_variables[0][EQUIVALENT_SLIP].copy()

    for step in range(1, 41):
        assert _integrate(mgis, data, (peak * (1.0 - step / 20.0))[None, :], 1.0)
        mgis.update(data)

    state = data.s1.internal_state_variables[0]
    # The slips travel back, but the accumulated equivalent slip only ever grows.
    assert np.abs(state[PLASTIC_SLIP]).max() < np.abs(forward).max()
    assert (state[EQUIVALENT_SLIP] >= forward_equivalent - 1e-15).all()


# --------------------------------------------------------------------------
# Section 9.4 - convergence under subdivision
# --------------------------------------------------------------------------


def test_the_answer_converges_as_the_step_shrinks() -> None:
    """Not equality across discretisations, but a shrinking difference.

    A rate-independent law is still path-discretisation dependent: the flow
    amplitude is proportional to the equivalent increment, so halving the step
    changes the answer until the sequence converges.
    """

    path = _isochoric_axial((1.0, 2.0, 3.0), 6.0e-3)
    stresses = {steps: _ramp(SRIX, path, steps=steps)[0][0] for steps in (10, 20, 40, 80)}

    differences = [
        abs(stresses[20] - stresses[10]),
        abs(stresses[40] - stresses[20]),
        abs(stresses[80] - stresses[40]),
    ]
    assert differences[1] < differences[0]
    assert differences[2] < differences[1]
    # The finest pair must already agree to well under a percent.
    assert differences[2] / abs(stresses[80]) < 1e-2


# --------------------------------------------------------------------------
# Section 9.5 - the consistent tangent
# --------------------------------------------------------------------------


def _tangent_against_finite_differences(
    behaviour_name: str,
    preload: np.ndarray,
    increment: np.ndarray,
    *,
    perturbation: float = 1e-9,
) -> float:
    mgis, data = _manager(behaviour_name)
    for step in range(1, 11):
        assert _integrate(mgis, data, (preload * step / 10)[None, :], 1.0)
        mgis.update(data)

    strain = preload + increment
    assert _integrate(mgis, data, strain[None, :], 1.0)
    algorithmic = data.K[0].copy()
    reference_stress = data.s1.thermodynamic_forces[0].copy()

    numerical = np.zeros((6, 6))
    for column in range(6):
        perturbed = strain.copy()
        perturbed[column] += perturbation
        assert _integrate(mgis, data, perturbed[None, :], 1.0)
        numerical[:, column] = (
            data.s1.thermodynamic_forces[0] - reference_stress
        ) / perturbation

    return float(np.abs(algorithmic - numerical).max() / np.abs(numerical).max())


@pytest.mark.parametrize(
    ("label", "preload", "increment"),
    [
        ("elastic", np.zeros(6), _isochoric_axial((1.0, 0.0, 0.0), 1e-5)),
        (
            "near the transition",
            _isochoric_axial((1.0, 0.0, 0.0), 3e-4),
            _isochoric_axial((1.0, 0.0, 0.0), 3e-5),
        ),
        (
            "established plasticity",
            _isochoric_axial((1.0, 0.0, 0.0), 5e-3),
            _isochoric_axial((1.0, 0.0, 0.0), 3e-4),
        ),
        (
            "transverse perturbation",
            _isochoric_axial((1.0, 0.0, 0.0), 5e-3),
            np.array([0.0, 0.0, 3e-4, 0.0, 0.0, 0.0]),
        ),
        ("shear", np.array([0.0, 0.0, 0.0, 4e-3, 0.0, 0.0]), np.array([0, 0, 0, 3e-4, 0, 0.0])),
        (
            "orientation [111]",
            _isochoric_axial((1.0, 1.0, 1.0), 5e-3),
            _isochoric_axial((1.0, 1.0, 1.0), 3e-4),
        ),
    ],
)
def test_the_algorithmic_tangent_matches_finite_differences(
    label: str, preload: np.ndarray, increment: np.ndarray
) -> None:
    """The tolerance is set by the finite differences, not by the law.

    A one-sided difference with a 1e-9 perturbation on stresses of order 1e5
    carries a relative truncation error near 1e-7, so anything at that level is
    the measurement and not the model. Deriving f alone while treating the
    equivalent strain increment as a constant scores 2e-2 to 4e-1 here, three
    to six orders of magnitude worse, which is what this test exists to catch.
    """

    deviation = _tangent_against_finite_differences(SRIX, preload, increment)

    assert deviation < 1e-5


# --------------------------------------------------------------------------
# Section 10 - comparison against the law it was transposed from
# --------------------------------------------------------------------------


def test_the_transposition_matches_meric_cailletaud_in_tension_along_001() -> None:
    """Equation (16) equates the two overstresses for [001] tension only.

    The reference rate is 1e-3 per second, so the Meric-Cailletaud run has to
    traverse 6e-3 of equivalent strain in 6 seconds for the comparison to mean
    anything.
    """

    axis = _axis((1.0, 0.0, 0.0))
    path = _isochoric_axial((1.0, 0.0, 0.0), 6.0e-3)
    viscous, _, _ = _ramp(MERIC_CAILLETAUD, path, total_time=6.0)
    rate_independent, _, _ = _ramp(SRIX, path, total_time=6.0)

    assert _axial_stress(rate_independent, axis) == pytest.approx(
        _axial_stress(viscous, axis), rel=0.01
    )


@pytest.mark.parametrize(
    ("label", "direction"), [("[111]", (1.0, 1.0, 1.0)), ("[123]", (1.0, 2.0, 3.0))]
)
def test_the_transposition_is_not_exact_away_from_001(
    label: str, direction: tuple[float, float, float]
) -> None:
    """Documented as a limit, and asserted so it cannot be quietly overstated.

    The two laws stay the same order of magnitude, but they are different
    models; only [001] is pinned by equation (16).
    """

    axis = _axis(direction)
    path = _isochoric_axial(direction, 6.0e-3)
    viscous = _axial_stress(_ramp(MERIC_CAILLETAUD, path, total_time=6.0)[0], axis)
    rate_independent = _axial_stress(_ramp(SRIX, path, total_time=6.0)[0], axis)

    relative = abs(rate_independent - viscous) / abs(viscous)
    assert relative > 0.01
    # Measured at 7.1 percent for [111] and 14.2 percent for [123].
    assert relative < 0.20


def test_the_rate_independent_law_takes_larger_steps() -> None:
    """The practical reason to prefer SRIX for our increment-hungry campaigns.

    Meric-Cailletaud rejects a step whose overstress exceeds 1.1 K, a guard its
    Norton power needs and the linear SRIX flow does not.
    """

    mgis, data = _manager(SRIX)
    assert _integrate(mgis, data, _isochoric_axial((1.0, 0.0, 0.0), 5.0e-2)[None, :], 1.0)

    mgis, viscous = _manager(MERIC_CAILLETAUD)
    assert not _integrate(
        mgis, viscous, _isochoric_axial((1.0, 0.0, 0.0), 5.0e-2)[None, :], 1.0
    )
