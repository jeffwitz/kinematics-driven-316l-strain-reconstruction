from __future__ import annotations

import numpy as np

from fem_inhouse.identification.pointwise_whitening import PointwiseFieldWhitener, null_test


def _non_uniform_noise(realisations: int, seed: int = 1) -> np.ndarray:
    """Noise whose amplitude varies across the field, as the real one does."""

    generator = np.random.default_rng(seed)
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 24), np.linspace(0.0, 1.0, 24), indexing="ij")
    amplitude = 0.2 + 3.0 * np.exp(-8.0 * ((x - 0.3) ** 2 + (y - 0.7) ** 2))
    return generator.normal(size=(realisations, 24, 24, 2)) * amplitude[None, :, :, None]


def test_the_pointwise_whitener_returns_noise_for_unseen_noise() -> None:
    """Fitted on one half, tested on the other: the only check that cannot be faked.

    A whitener validated on the samples it was fitted to reproduces their norm
    by construction. The spectral whitener passed exactly that self-consistency
    check on the propagated residual and still failed to return noise for noise
    on data it had not seen, so the acceptance criterion here is held-out.
    """

    samples = _non_uniform_noise(240)
    whitener = PointwiseFieldWhitener.fit(samples[:180])
    result = null_test(whitener, samples[180:])

    assert 0.95 < result["norm_over_expected"] < 1.05
    assert 0.9 < result["directional_standard_deviation"] < 1.1


def test_the_norm_alone_would_not_have_caught_a_directional_failure() -> None:
    """Why significance must be judged along the direction actually projected on.

    A field with the right total energy concentrated along one direction is
    exactly what turns a projection onto a mode into a spurious detection. The
    global norm is blind to it. Random probes are blind to it too, in a space of
    this dimension -- which is why `null_test` reports the worst direction it
    sampled rather than the average, and why a mode of interest should be among
    the directions checked.
    """

    generator = np.random.default_rng(5)
    samples = generator.normal(size=(120, 16, 16, 2))
    preferred = generator.normal(size=(16, 16, 2))
    preferred /= np.linalg.norm(preferred)
    loading = generator.normal(size=120) * 8.0
    skewed = samples + loading[:, None, None, None] * preferred[None, ...]
    scale = np.linalg.norm(samples.reshape(len(samples), -1), axis=1) / np.linalg.norm(
        skewed.reshape(len(skewed), -1), axis=1
    )
    skewed *= scale[:, None, None, None]

    identity = PointwiseFieldWhitener(
        inverse_deviation=np.ones((16, 16, 2)), spectral=None
    )
    honest = null_test(identity, samples)
    skew = null_test(identity, skewed)
    # The global norm cannot tell them apart.
    assert abs(skew["norm_over_expected"] - honest["norm_over_expected"]) < 0.05

    # Along the planted direction it is obvious.
    flat = preferred.reshape(-1)
    assert float(np.std(skewed.reshape(len(skewed), -1) @ flat)) > 3.0 * float(
        np.std(samples.reshape(len(samples), -1) @ flat)
    )


def test_the_adjoint_is_the_transpose_of_the_forward_action() -> None:
    samples = _non_uniform_noise(60, seed=2)
    whitener = PointwiseFieldWhitener.fit(samples)
    generator = np.random.default_rng(7)
    left = generator.normal(size=(24, 24, 2))
    right = generator.normal(size=(24, 24, 2))
    forward = float((whitener.apply(left) * right).sum())
    backward = float((left * whitener.adjoint(right)).sum())
    assert abs(forward - backward) <= 1e-10 * abs(forward)
