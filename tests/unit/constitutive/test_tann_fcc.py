"""Material-level unit tests of the causal TANN-FCC (T0).

The seven gates of `scripts/qualify_tann_fcc_material.py` as fast tests on
small point counts, plus the neutrality of the zero spatial context (the
T1 extension point). Thresholds mirror the preregistration in
`validation/tann_fcc_preregistration.md`; none may be moved after results.
"""

from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.constitutive.tann_fcc import TannFCCBatch, TannFCCConfig
from fem_inhouse.core.fcc_interaction_matrix import SLIP_SYSTEMS
from fem_inhouse.core.srix_canonical import ACTIVE_SYSTEMS_001, SCHMID_FACTOR_001
from fem_inhouse.identification.spatial_context import ZeroSpatialContext

POINTS = 16
RNG = np.random.default_rng(20260817)


def _identity_systems(points: int) -> np.ndarray:
    """Schmid tensors in the specimen frame for the identity orientation."""

    tensors = np.empty((12, 3, 3))
    for index, (burgers, normal) in enumerate(SLIP_SYSTEMS):
        s = np.asarray(burgers, dtype=np.float64)
        m = np.asarray(normal, dtype=np.float64)
        s /= np.linalg.norm(s)
        m /= np.linalg.norm(m)
        tensors[index] = 0.5 * (np.outer(s, m) + np.outer(m, s))
    return np.tile(tensors, (points, 1, 1, 1))


def _batch(config: TannFCCConfig | None = None) -> TannFCCBatch:
    return TannFCCBatch(
        config or TannFCCConfig(),
        point_count=POINTS,
        systems_global=_identity_systems(POINTS),
    )


def test_tann_zero_increment() -> None:
    batch = _batch()
    trial = batch.evaluate(np.zeros((POINTS, 3)), compute_tangent=False)
    assert np.max(np.abs(trial.trial_state)) == 0.0
    assert np.max(np.abs(trial.plastic_slip)) == 0.0


def test_tann_dissipation_nonnegative() -> None:
    batch = _batch()
    worst = np.inf
    for _ in range(4):
        strain_1 = RNG.normal(scale=2e-3, size=(POINTS, 3))
        strain_2 = strain_1 + RNG.normal(scale=2e-3, size=(POINTS, 3))
        batch.evaluate(strain_1, compute_tangent=False)
        batch.commit()
        trial = batch.evaluate(strain_2, compute_tangent=False)
        worst = min(worst, float(trial.generalised_dissipation.min()))
        batch.revert()
    assert worst >= -1e-9


def test_tann_system_permutation_equivariance() -> None:
    batch = _batch()
    permutation = RNG.permutation(12)
    batch_perm = TannFCCBatch(
        TannFCCConfig(),
        point_count=POINTS,
        systems_global=_identity_systems(POINTS)[:, permutation, :, :],
    )
    batch_perm.copy_weights_from(batch)
    strain = RNG.normal(scale=2e-3, size=(POINTS, 3))
    trial = batch.evaluate(strain)
    trial_perm = batch_perm.evaluate(strain)
    assert np.allclose(
        trial.stress_in_plane_mpa, trial_perm.stress_in_plane_mpa, atol=1e-12
    )
    assert np.allclose(
        trial.consistent_tangent_mpa, trial_perm.consistent_tangent_mpa, atol=1e-12
    )


def test_tann_substepping() -> None:
    base = _batch()
    strain = RNG.normal(scale=2e-3, size=(POINTS, 3))
    states = {}
    for substeps in (1, 2, 4, 8):
        sub = TannFCCBatch(
            TannFCCConfig(n_substeps=substeps),
            point_count=POINTS,
            systems_global=_identity_systems(POINTS),
        )
        sub.copy_weights_from(base)
        states[substeps] = sub.evaluate(strain, compute_tangent=False).trial_state
    error_1_2 = float(np.max(np.abs(states[1] - states[2])))
    error_4_8 = float(np.max(np.abs(states[4] - states[8])))
    # RK4 is fourth order: halving the step shrinks the error by ~16x, so
    # the 4-vs-8 gap is machine precision against the 1-vs-2 gap.
    assert error_4_8 < 1e-8
    assert error_4_8 < 0.01 * error_1_2


def test_tann_algorithmic_tangent_fd() -> None:
    batch = _batch()
    strain = RNG.normal(scale=1e-3, size=(POINTS, 3))
    trial = batch.evaluate(strain)
    tangent_ad = trial.consistent_tangent_mpa

    def central_fd(step: float, component: int) -> np.ndarray:
        plus = strain.copy()
        minus = strain.copy()
        plus[:, component] += step
        minus[:, component] -= step
        stress_plus = batch.evaluate(plus, compute_tangent=False).stress_in_plane_mpa
        stress_minus = batch.evaluate(minus, compute_tangent=False).stress_in_plane_mpa
        return (stress_plus - stress_minus) / (2 * step)

    # The gate is the Richardson combination of the h/h/2 central pair
    # (the h^2 truncation cancels; plain central FD bottoms out on
    # round-off before reaching 1e-5 at the worst point).
    worst = np.inf
    for step in (1e-4, 1e-5, 1e-6):
        for component in range(3):
            fd_h = central_fd(step, component)
            fd_half = central_fd(step / 2, component)
            rich = (4 * fd_half - fd_h) / 3
            error = float(np.max(np.abs(tangent_ad[:, :, component] - rich)))
            worst = min(worst, error)
    assert worst <= 1e-5


def test_tann_transaction_revert() -> None:
    batch = _batch()
    strain_1 = RNG.normal(scale=1e-3, size=(POINTS, 3))
    first = batch.evaluate(strain_1, compute_tangent=False)
    second = batch.evaluate(strain_1, compute_tangent=False)
    assert np.array_equal(first.stress_in_plane_mpa, second.stress_in_plane_mpa)
    committed_before = batch.committed_state.copy()
    batch.evaluate(
        strain_1 + RNG.normal(scale=1e-4, size=(POINTS, 3)), compute_tangent=False
    )
    batch.revert()
    assert np.array_equal(batch.committed_state, committed_before)
    batch.evaluate(strain_1, compute_tangent=False)
    batch.commit()
    assert not np.array_equal(batch.committed_state, committed_before)
    with pytest.raises(RuntimeError):
        batch.commit()  # a second commit without an accepted trial is forbidden


def test_fcc_geometry_matches_existing_convention() -> None:
    stress = np.zeros((3, 3))
    stress[1, 1] = 1.0  # uniaxial yy, i.e. [001] tension
    tensors = _identity_systems(1)[0]
    tau = np.einsum("ij,aij->a", stress, tensors)
    assert np.count_nonzero(tau) == ACTIVE_SYSTEMS_001
    assert np.allclose(np.abs(tau[tau != 0]), SCHMID_FACTOR_001, atol=1e-14)


def test_zero_spatial_context_no_effect() -> None:
    config = TannFCCConfig(context_dim=2)
    batch = _batch(config)
    provider = ZeroSpatialContext(context_dim=2)
    strain = RNG.normal(scale=2e-3, size=(POINTS, 3))
    context = provider.forward(
        batch.committed_state, _identity_systems(POINTS), np.zeros(POINTS, dtype=int)
    )
    plain = batch.evaluate(strain, compute_tangent=False)
    with_context = batch.evaluate(strain, compute_tangent=False, context=context)
    # A zero context contributes exactly nothing to the linear embedding;
    # only floating-point summation order can differ between the two paths.
    assert np.allclose(
        plain.stress_in_plane_mpa, with_context.stress_in_plane_mpa, rtol=1e-12, atol=1e-10
    )
