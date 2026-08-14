# Tensor plastic observability on P43 M20

`scripts/qualify_tensor_plastic_observability_m20.py`. Artefact:
`validation/_generated/performance/experimental_oracle_p43_m20/tensor_observability/`.

The unknown is the tensor increment `Delta eps_p` per material point, with no
J2 direction imposed. The mechanics is then linear, the plastic field enters
only through `f = G z` with `G = B^T C`, and the whole observability question
is one SVD of `A = W_D M_D K^-1 L`, where `S = G H^-1 G^T = L L^T`.

Gauge: `H_loc = M^-1 / point_count`. Since `n = M s / q` gives `n^T M^-1 n = 1`,
this is the metric in which the norm of `Delta p n` is `Delta p` itself. `M`
would weight the shear wrongly.

## Checks before the result

| check | value |
|---|---|
| `<s, B u>` against `<B^T s, u>` | `6.0e-15` |
| dense `K, G` against the solver's own residual assembly | `3.3e-16` |
| modes `H`-orthonormal by construction | `5.8e-15` |
| `S` and `K` positive definite | Cholesky succeeded |

The second row is the one that matters: the assembled operators reproduce
`pack(divergence(C(strain(u) - eps_p)))` as the qualified code computes it.

## The quotient is as large as expected

```text
plastic components   2400
free mechanical dofs  722
rank(G)               722      (full row rank, so S is SPD and no pseudo-inverse is needed)
dim ker(G)           1678      70 % of the space is self-equilibrated and strictly invisible
```

## The spectrum falls off a cliff after two modes

| j | `sqrt(lambda)` | SNR at increment RMS | SNR at accumulated RMS | wavelength mm | uniform share |
|---:|---:|---:|---:|---:|---:|
| 1 | `9.59e+02` | `0.0076` | `0.183` | `0.0144` | `0.000` |
| 2 | `7.62e+02` | `0.0060` | `0.145` | `0.0144` | `0.000` |
| 3 | `2.31e+00` | `0.0000` | `0.0004` | `0.0207` | `0.000` |
| 4 | `2.08e+00` | `0.0000` | `0.0004` | `0.0295` | `0.000` |

Effective rank `1.90`. Between mode 2 and mode 3 the singular value drops by a
factor of `330`.

The two amplitudes are the archived oracle's own: `2.35e-4` root-mean-square
for a single increment, `5.67e-3` for the accumulated equivalent plastic strain
at the final state. Judging an accumulated field at the increment amplitude
would understate observability by a factor of 24, so both are reported.

## Two predictions, both wrong

**The low-pass ordering is refuted.** I expected the modes to be ordered by
spatial wavelength, the DIC transfer damping high wavenumbers. They are not:
modes 1 and 2 share `0.0144 mm`, mode 7 sits at `0.0033 mm` and mode 12 at
`0.0147 mm`. There is no monotone relation between rank and wavelength, and the
structure is a two-mode cliff rather than a graded filter.

**They are not uniform fields either.** The share of each archived mode carried
by its spatial mean is `0.000` throughout, so the cliff is not "the two uniform
tensor components dominate". What the first two modes are remains to be named.

## What this says, and what it does not

**No mode reaches the noise.** The best has `SNR = 0.18` at the accumulated
amplitude, so in its most observable direction the tensor plastic field
produces a whitened DIC signal five times below the measurement noise. Removing
the J2 restriction does not open an observable direction that the scalar
parametrisation was hiding.

**This is M20 only, and the size dependence is not measured.** The crop is
`20 x 20` pixels, `0.0368 mm` across. An eigenstrain produces a displacement of
order `strain x length`, so the same plastic amplitude on a larger domain
produces a larger signal against an unchanged per-node noise. Whether the
conclusion survives at M100 or on the full crop is the decisive open question
and is *not* answered here. The script accepts `--pixels`, but the dense `G`
assembly and its SVD dominate the cost beyond `n = 20` and need an efficiency
pass before the sweep is affordable.

**A coincidence checked and rejected.** The first two SNR values, `0.183` and
`0.145`, are numerically close to the archived scalar spectrum's `0.181` and
`0.142`. They are not the same quantity: those are quoted at a reference scale
of `2.96e-4` while these are at `5.67e-3`, a factor of 19, and the scalar run
sums over 40 states with `sum_to_one` weighting whereas this operator is
state-independent. The agreement is accidental until the scalar normalisation
is unpicked, and nothing here rests on it.
