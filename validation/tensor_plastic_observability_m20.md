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

## Matrix-free, and what the scaling changes

The dense prototype above is a qualification oracle, not an algorithm: at M100
its forcing operator alone would be `19602 x 60000`, about 9.4 GB. None of it is
needed. `fem_inhouse.identification.tensor_plastic_observability` applies
`A = W_D M_D K^-1 B^T C H^-1/2` and its adjoint through existing field
operators, with `K` recovered as a sparse matrix by colouring its `3 x 3` nodal
stencil in eighteen applications, at any mesh size.

Validated against the dense M20 before use:

| check | value |
|---|---|
| six leading singular values | `2.2e-07` relative |
| rank-2 principal angles | `0.0` and `8.5e-07` degrees |
| `<A x, y>` against `<x, A^T y>` | `5.3e-14` |

M20 falls from minutes to `0.09 s`; M100 with six modes takes `5 s`.

`scripts/scale_tensor_plastic_observability.py`, artefact
`tensor_observability_scaling.json`:

| pixels | window | `sigma_1` | gap after mode 2 | SNR increment | SNR accumulated |
|---:|---:|---:|---:|---:|---:|
| 20 | `36.8 um` | `9.59e+02` | `329.8` | `0.0076` | `0.183` |
| 30 | `55.2 um` | `2.07e+03` | `2.4` | `0.0111` | `0.268` |
| 40 | `73.6 um` | `3.73e+03` | `1.1` | `0.0151` | `0.365` |
| 60 | `110.4 um` | `1.17e+04` | `1.1` | `0.0319` | `0.769` |
| 80 | `147.2 um` | `2.42e+04` | `1.1` | `0.0496` | `1.197` |
| 100 | `184.0 um` | `3.72e+04` | `1.2` | `0.0612` | `1.476` |

**The M20 conclusion does not survive, and neither does the M20 structure.**

The accumulated plastic field becomes detectable: `SNR` crosses 1 between M60
and M80 and reaches `1.48` at M100. The growth is faster than linear in the
window — a factor of `8.1` for a factor of `5` in size — so extrapolating the
M20 number linearly would have understated it.

But the factor-330 cliff after two modes is **an artefact of the small window**.
The gap collapses to `2.4` at M30 and `1.1` from M40 onward: at M100 the leading
modes are nearly degenerate and there is no rank-2 structure to exploit.

Sixty leading modes at M100 confirm a slow, graded decay rather than a cliff:
`sigma_1 / sigma_j` reaches only `1.7` at `j = 10`, `2.5` at `j = 20` and `6.7`
at `j = 60`. Seven modes exceed one noise sigma at the accumulated amplitude and
**none exceeds three**.

So the reduction is real but it is a noise truncation, not a structural rank:
the useful dimension is set by where the threshold is placed, and at M100 it is
about seven marginally detectable directions out of 60 000 components.

**Increment-by-increment identification stays out of reach at every size.** Its
`SNR` is `0.061` at M100, sixteen times below the noise. Only the accumulated
field is accessible, which makes the reconstruction target `eps_p^n` rather than
each `Delta eps_p` — the stress at state `n` depends on the total anyway.
