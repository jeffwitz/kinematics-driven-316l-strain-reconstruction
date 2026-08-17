"""Gate 1: the split reproduces the behaviour it decomposes, to rounding."""

from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.core.element import plane_stress_elasticity
from fem_inhouse.core.plane_stress_material import PythonJ2PlaneStressBatch
from fem_inhouse.hyperreduction import ConstitutiveSplit, reference_stiffness_of


def _material(points: int) -> PythonJ2PlaneStressBatch:
    return PythonJ2PlaneStressBatch(
        np.full(points, 260.0), np.full(points, 900.0), 0.32,
        young_modulus_mpa=205_000.0, poisson_ratio=0.30,
    )


def test_reference_stiffness_is_the_engineering_elasticity() -> None:
    """Taken from the behaviour, not rebuilt -- the conventions must not drift.

    These batches are engineering, not Kelvin: the shear entry differs by mu.
    """

    stiffness = reference_stiffness_of(_material(8))
    expected = plane_stress_elasticity(205_000.0, 0.30)
    assert np.abs(stiffness - expected).max() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("scale", [0.0, 5.0e-4, 1.5e-3, 4.0e-3])
def test_split_reassembles_the_exact_stress(scale: float) -> None:
    """Elastic and plastic alike, reference plus correction is the stress."""

    points = 512
    generator = np.random.default_rng(20260817)
    material = _material(points)
    split = ConstitutiveSplit(_material(points))
    strain = np.zeros((points, 3))
    strain[:, 1] = scale
    strain[:, 0] = -0.3 * scale
    strain += 1e-5 * generator.standard_normal((points, 3))

    exact = np.asarray(material.evaluate(strain, time_increment=1.0).stress_in_plane_mpa)
    trial = split.evaluate(strain)
    scale_of = max(float(np.abs(exact).max()), 1.0)
    assert np.abs(trial.stress_mpa - exact).max() / scale_of < 1e-14
    assert np.abs(trial.tangent_correction_mpa
                  - (trial.tangent_mpa - split.reference_stiffness)).max() == 0.0


def test_correction_vanishes_at_the_committed_state() -> None:
    """h(u_n) = 0, which is why the split is taken here and not at the origin."""

    points = 256
    split = ConstitutiveSplit(_material(points))
    strain = np.zeros((points, 3))
    strain[:, 1] = 2.5e-3
    strain[:, 0] = -0.3 * 2.5e-3
    split.evaluate(strain)
    split.commit()

    trial = split.evaluate(strain)
    reference = max(float(np.abs(trial.stress_mpa).max()), 1.0)
    # Not machine epsilon: the residue is the return mapping's own local
    # tolerance, 1e-10 MPa absolute, re-entered at the committed strain.
    assert np.abs(trial.correction_mpa).max() / reference < 1e-11


def test_committed_state_survives_a_trial() -> None:
    """A trial must never move the point the split is taken around."""

    points = 128
    split = ConstitutiveSplit(_material(points))
    before = split.committed_stress_mpa.copy()
    strain = np.zeros((points, 3))
    strain[:, 1] = 3.0e-3
    split.evaluate(strain)
    split.evaluate(0.5 * strain)
    assert np.array_equal(split.committed_stress_mpa, before)
