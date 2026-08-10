"""The subgradient at the cusp must not reach the SRIX residual.

``SrixSlipZeroDerivative`` picks which element of the Clarke subdifferential of
``|x|`` is used at exactly ``dg = 0``. That is a property of the generalised
Jacobian, not of the law: the value must never influence what the converged
state is, only how easily the local Newton reaches it.

The qualification behind this is in
``validation/srix_semismooth_subgradient_preregistration.md`` and its results
document. These tests keep the two facts that qualification rests on from
regressing: the subgradient does not move the root, and the compact ``delta``
regularisation does.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("mgis")

from fem_inhouse.core.crystal_parameter_pairs import (
    resolve_paired_crystal_parameters,
)
from fem_inhouse.core.mfront_behaviours import MFRONT_BEHAVIOURS
from fem_inhouse.core.mfront_gps.adapter import (
    MFrontNativeGeneralisedPlaneStressBatch,
)

pytestmark = pytest.mark.mfront


def _material(
    zero_derivative: float, delta: float = 0.0
) -> MFrontNativeGeneralisedPlaneStressBatch:
    spec = MFRONT_BEHAVIOURS.get("fcc_forest_rubin_srix_gps")
    parameters, _ = resolve_paired_crystal_parameters(
        paired_parameter_set="316l_guilhem2013_nasri2018_meric_srix_rate_1e-3",
        law="forest_rubin_srix",
    )
    parameters["SrixSlipSmoothingDelta"] = delta
    parameters["SrixSlipZeroDerivative"] = zero_derivative
    library = os.environ.get(
        "MFRONT_BEHAVIOUR_LIBRARY", "build/mfront/src/libBehaviour.so"
    )
    return MFrontNativeGeneralisedPlaneStressBatch(
        library,
        behaviour_spec=spec,
        point_count=1,
        rotation_global_to_material=np.eye(3)[None, :, :],
        thread_count=1,
        behaviour_name=spec.behaviour_name("condensed_3d"),
        behaviour_parameters=parameters,
        backend_label="subgradient-regression",
    )


#: Amplitude and step count of the probe path. Both matter. `delta` acts on the
#: per-step slip increment, not on the accumulated slip, so a path made of a few
#: large steps never enters the regularised zone and the control below passes
#: for the wrong reason -- measured at `2.7e-14` for both variants with eight
#: steps of `0.02`, which proves nothing. Sixty-four steps of `0.004` put the
#: increments below `1e-5`, where `delta` is live.
PROBE_AMPLITUDE = 0.004
PROBE_STEPS = 64


def _walk(material: MFrontNativeGeneralisedPlaneStressBatch) -> np.ndarray:
    """Load past yield in steps small enough to exercise the cusp."""

    for step in range(1, PROBE_STEPS + 1):
        strain = (step / PROBE_STEPS) * PROBE_AMPLITUDE * np.array([1.0, -0.4, 0.15])
        material.evaluate(np.atleast_2d(strain), time_increment=1.0 / PROBE_STEPS)
        material.commit()
    return np.asarray(
        material._manager.s1.internal_state_variables[0, :], dtype=float
    ).copy()


def _relative(candidate: np.ndarray, reference: np.ndarray) -> float:
    scale = float(np.linalg.norm(reference))
    return float(np.linalg.norm(candidate - reference)) / scale


def test_the_cusp_subgradient_does_not_move_the_converged_state() -> None:
    """`sign(0) = -1` and `sign(0) = 0` must agree to the Newton tolerance.

    Measured at `4.3e-11` on this path and `1.5e-11` on the 380 archived hard
    states, where it was also shown to fall proportionally with `epsilon`. The
    bound is loose on purpose: it must catch the subgradient leaking into the
    residual, not tolerance noise.
    """

    historical = _walk(_material(-1.0))
    symmetric = _walk(_material(0.0))
    assert _relative(symmetric, historical) < 1.0e-8


def test_the_compact_regularisation_does_move_the_converged_state() -> None:
    """The control. Without it the test above could pass on an inert switch.

    `delta` enters `p_`, `da` and `x`, so it changes the residual and therefore
    the root; measured at `8.6e-4` here, seven decades above the subgradient.
    If this ever agrees to the tolerance too, the first test proves nothing
    about the subgradient -- it would just mean neither knob is live on this
    path, which is exactly how the first draft of this file passed.
    """

    historical = _walk(_material(-1.0))
    regularised = _walk(_material(-1.0, delta=1.0e-5))
    assert _relative(regularised, historical) > 1.0e-5
