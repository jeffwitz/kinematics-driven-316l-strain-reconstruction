# Are the observable plastic modes actually excited by the experiment?

The observability spectrum says what the DIC chain *could* see if a mode were
present at a reference amplitude. It says nothing about whether the experiment
contains it. This projects the residuals of the 40 measured states onto the
left singular vectors of the observability operator.

Scripts: `scripts/project_dic_residuals_on_observable_modes.py`,
`scripts/diagnose_elastic_model_residual_p43.py`. Artefacts:
`dic_excitation_m20.json`, `dic_excitation_m100.json`,
`elastic_model_residual.json`.

## Two corrections to the construction

**The transfer applies to the model only.** The measured field has already been
through the instrument; applying `M_D` to both sides blurs the data twice. The
residual is `r_n = W_D (u_DIC,n - M_D u_elastic,n)`.

**The elastic reference needs no boundary-coupling block.** The measured field
satisfies `K u_int + K_ib u_b = f_int` with `f_int` its own interior
out-of-balance force, and the elastic field satisfies the same with a zero
right-hand side, so `u_elastic,int = u_DIC,int - K^-1 f_int` exactly.

Because `W_D` whitens, a projection onto a unit left singular vector has unit
variance under pure noise: the coefficients are **already z-scores**, with no
separate uncertainty propagation.

## The noise model is right

Whitened norm of real noise realisations against `sqrt(interior components)`:
`1.109` at M20 and `1.025` at M100. The whitener is calibrated, so the sigma
units below mean what they say.

## M20: the plastic signature is below the noise

| state | deviation from elastic, per node | whitened norm / pure noise |
|---:|---:|---:|
| 1 | `0.005` sigma | `0.02` |
| 10 | `0.040` sigma | `0.15` |
| 40 | `0.543` sigma | `1.49` |

The measured field on a `36.8 um` crop is elastic to within a fraction of the
DIC noise at every state. The two best-observed modes carry `0.27` and `0.07`
sigma. What does rise above noise sits in modes 3 and 10, whose singular values
are four hundred times smaller, and converting those coefficients to a plastic
amplitude gives `a = c / sigma ~ 2.6`, an RMS equivalent plastic strain of
260 %. That is not plasticity; it is high-frequency discrepancy landing near the
null space of the operator.

## M100: the signature is detected, at physically right amplitudes

Coefficients in noise sigma, `crop (1610, 1710, 1075, 1175)`:

| mode | s1 | s2 | s3 | s5 | s10 | s20 | s30 | s40 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `0.67` | `-2.12` | `-3.98` | `-4.89` | `-14.3` | `-29.4` | `-71.5` | `-118.1` |
| 3 | `1.11` | `1.48` | `0.35` | `1.32` | `2.48` | `9.84` | `44.5` | `72.3` |
| 7 | `-0.27` | `-2.24` | `-2.54` | `-4.71` | `-5.71` | `5.73` | `62.0` | `124.6` |
| 8 | `-0.41` | `3.65` | `5.83` | `9.69` | `20.4` | `39.0` | `110.1` | `166.5` |

**The null test passes.** At state 1 the largest coefficient over twenty modes
is `1.11` sigma, which is what twenty draws of unit-variance noise give.

**The signal grows monotonically with the load** and reaches `167` sigma.

**And the amplitudes are physically right.** Converting through
`a_j = c_j / sigma_j` gives RMS equivalent plastic strains of `2.1e-3` to
`6.6e-3` at the final state, against the archived oracle's accumulated
`5.67e-3`. The modes that the mechanics and the measurement chain single out,
projected onto data they never saw, return the plastic strain level the
material is independently known to have reached.

This reverses the M20 reading. Plastic reconstruction from DIC alone is not out
of reach; it was out of reach *on a 37 micrometre window*.

## What is not established

**The growth is superlinear but not sharply so.** Fitting the coefficient
against the load amplitude gives an exponent of `1.33` for mode 1, `1.29` for
mode 8, and `1.41` for the total residual norm. Pure elastic heterogeneity
would give exactly `1`, so the residual is not only heterogeneity; but a clean
plastic onset would give much more than `1.4`, so it is not only plasticity
either. A mixture is the honest reading, and the two are not separated here.

**And they may not be separable in this parametrisation at all.** A free tensor
eigenstrain reproduces the effect of an elastic inclusion exactly -- Eshelby's
equivalent inclusion -- so elastic heterogeneity and plasticity produce the same
class of forcing `f = G z`. This crop spans many grains of a polycrystal, whose
elastic anisotropy is real. No regularisation inside the plastic space can
separate the two; separating them needs either the EBSD orientations in the
elastic operator, or a loading path that distinguishes them.

Until that is settled, `a_j` should be read as *equivalent eigenstrain*, not as
plastic strain, however well the number matches.
