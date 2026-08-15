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

## The two components separate on the data, without EBSD

The normalised residual fields are mutually parallel early and rotate away
later — cosine similarity `0.93` to `0.98` among states 5, 10 and 20, `0.909`
between states 30 and 40, but only `0.27` between states 5 and 40. Two regimes,
with a fixed pattern in each.

That is enough to separate them without grain data and without assuming
proportionality. The early states span the heterogeneity subspace directly;
removing it from the late residuals leaves what elasticity cannot produce at a
fixed shape. `scripts/separate_elastic_heterogeneity_from_plasticity_p43.py`,
artefact `heterogeneity_plasticity_separation_m100.json`.

A rank-3 subspace fitted on states 3-20 captures **99.70 %** of their variance:

| state | raw norm / noise | corrected | max abs c raw | max abs c corrected |
|---:|---:|---:|---:|---:|
| 1 | `0.112` | `0.077` | `1.11` | `1.11` |
| 5 | `0.507` | `0.064` | `9.69` | `1.28` |
| 10 | `1.000` | `0.090` | `20.4` | `1.28` |
| 20 | `1.951` | `0.047` | `39.0` | `0.63` |
| 25 | `2.780` | `0.216` | `61.7` | `4.70` |
| 30 | `4.505` | `1.317` | `110.1` | `14.45` |
| 35 | `6.047` | `2.015` | `140.5` | `18.11` |
| 40 | `7.364` | `1.131` | `166.5` | `21.59` |

**Three fixed patterns explain the entire pre-yield residual.** Up to state 20
the corrected residual is `0.05` to `0.09` of the noise norm and no mode exceeds
`1.3` sigma. The load-proportional interpretation is not an assumption any more;
it is measured.

**A second component appears between states 20 and 25** and grows to `21.6`
sigma. That onset is sharp, which is what a yield point looks like and what pure
heterogeneity cannot produce.

**And it is an order of magnitude smaller than it looked.** The equivalent
eigenstrain of the leading modes falls from `2.1e-3 ... 6.6e-3` before the
correction to `1.1e-4 ... 5.8e-4` after it. Nine tenths of the apparent plastic
amplitude was elastic heterogeneity.

## What this leaves

The corrected amplitudes are far below the accumulated plastic strain the
material reached, `5.67e-3` RMS. The observable subspace therefore sees only a
small projection of the real plastic field — consistent with the spectrum, where
seven modes clear one noise sigma out of sixty thousand components.

One anomaly is recorded rather than explained: the corrected residual norm falls
from `2.015` at state 35 to `1.131` at state 40, where accumulated plasticity
should grow. Either the late pattern rotates further out of the twenty modes
retained, or state 40 carries something the earlier states do not.

The separation is a **lower bound** on the plastic content: any plastic
component lying inside the early subspace is removed with it. And the early
states are assumed plasticity-free, which the state-1 null test and the pattern
stability to state 20 support but do not prove.

## The modes are edge-dominated, and the reconstruction misses the localisation

`scripts/anatomy_of_the_observable_plastic_modes.py`, artefact
`mode_anatomy_m100/`. Each mode is described before anything is built on it: how
much of its energy sits away from the border, how it splits across the tensor
components, and at what spatial concentration.

**Every leading mode is concentrated near the boundary.** With a `15`-pixel
border, the interior covers `49 %` of the area, but the interior share of the
mode energy is `0.094` at worst, `0.197` at the median, and only one mode of
twenty exceeds the area fraction. Mode 1 puts `10.5 %` of its energy in half the
domain. Under Dirichlet boundaries a near-edge eigenstrain has the strongest
lever on the interior displacement, so the operator ranks those directions
first — mathematically observable, physically the boundary talking.

They are also **shear-dominated**: mean component shares
`0.24 / 0.22 / 0.54` for `e11 / e22 / g12`.

**And the reconstruction does not land where the material yields.** Combining
the modes with the heterogeneity-corrected coefficients gives a field whose
equivalent measure peaks at `7.08e-3`, against a measured peak equivalent strain
of `1.59e-2` — the right order. But its correlation with the DIC equivalent
strain map is only `+0.149`, and the share of it falling inside the DIC top
decile is `0.134`, against `0.10` for an unstructured field. Barely above
chance.

So the `21.6` sigma detection is real and it is not the bulk plastic
localisation. The observable subspace is dominated by near-boundary directions,
and that is what the data lights up in it.

This is the third reading of the same measurement, and each one removed a layer:
the raw coefficients looked like plasticity at the right amplitude; removing the
elastic-heterogeneity subspace cut nine tenths of it; and the anatomy shows that
what remains lives at the edges rather than in the band. Whatever is being
detected, calling it a reconstruction of the plastic field is not supported.

**What would decide it.** The edge dominance is a property of the Dirichlet
window, not of the specimen: the crop boundary is an artefact of choosing a
window inside a larger field. Re-running with the observation restricted to the
interior — masking the border out of `W_D` rather than only its outermost node —
would re-rank the modes on what the bulk can show, and is the cheapest next
test. It costs one extra mask and no new machinery.

### Masking the boundary band does not rescue it

Re-running with a `15`-node band removed from the observation, so the operator
is ranked on what the bulk can show:

| | border ring only | 15-node band masked |
|---|---:|---:|
| interior share, modes 1-6 | `0.11 ... 0.27` | `0.14 ... 0.21` |
| interior share, modes 8-11 | `0.09 ... 0.30` | `0.40 ... 0.46` |
| correlation with the DIC equivalent strain | `+0.149` | `-0.150` |
| share in the DIC top decile | `0.134` | `0.112` |

Masking lifts the middle modes to roughly the interior area fraction, so part of
the edge concentration was indeed the window. But the leading six modes stay
edge-weighted, and the correlation moves from `+0.15` to `-0.15` — two
independent observation geometries returning values consistent with zero.

The negative is therefore robust to the geometry: at M100 the tensor eigenstrain
reconstruction does not recover the plastic localisation the DIC shows, whether
or not the boundary band is observed.

## The defect is not crystallographic elastic heterogeneity

`scripts/test_ebsd_elastic_reference_p43.py`, artefacts
`ebsd_elastic_reference_m100.json` and `..._polycrystal.json`.

The reference elasticity is replaced by the real one: at each pixel the cubic
tensor of the FCC law this repository already declares — `E = 99950.3 MPa`,
`nu = 0.388`, `G = 122000 MPa`, Zener anisotropy `3.39` — is rotated in three
dimensions by the recorded EBSD orientation and then condensed exactly,
`C_ps = C_aa - C_ab C_bb^-1 C_ba`. Gauge, measurement chain, boundary
conditions, crop and diagnostics are unchanged. No plasticity model, no
crystal plasticity, no fitting.

**The chain is verified first.** An isotropic crystal condenses to
`plane_stress_elasticity` at any Euler angle, to `4e-16` relative — the 3D
rotation, the Voigt mapping and the Schur complement together. A cubic symmetry
leaves the tensor unchanged to `7e-17`. And on an affine boundary at 1 % strain,
the isotropic extension has an interior fluctuation of `6e-20 mm`, exactly the
zero theory requires, while the EBSD extension fluctuates by `6.18e-5 mm` RMS,
`0.66` DIC sigma. The anisotropy is real and it reaches the operator, whose
leading singular values move by 9 %.

**The first crop was a poor test and was replaced.** At `(1610, 1075)` a single
grain covers `70.4 %` of the window and three cover `94.5 %`; under a
near-affine Dirichlet boundary a uniform stiffness — anisotropic or not —
returns the same field, so there was little heterogeneity to find. The crop was
rescanned for grain diversity and `(1580, 1030)` retained: dominant grain
`20.6 %`, `7.2` effective grains.

**On that genuinely polycrystalline crop, the residual does not move:**

| state | isotropic | EBSD | shuffled |
|---:|---:|---:|---:|
| 20 | `1.180` | `1.180` | `1.180` |
| 30 | `2.944` | `2.948` | `2.944` |
| 40 | `5.873` | `5.875` | `5.873` |

And the shuffled control is identical, so this is not a question of the wrong
spatial arrangement.

**The reason is orthogonality, not smallness.** The correction the EBSD
reference makes to the residual is not negligible — `2.6 %` of it at state 20,
`3.0 %` at state 40 — but its cosine with the residual is `+0.0015` and
`+0.0057`. It points somewhere else entirely. Subtracting an orthogonal
component cannot reduce a norm, and indeed the norm rises by the `sqrt(1+e^2)`
that predicts, `5.8734` to `5.8750`. Only `1 %` of the correction lands in the
empirical early rank-3 subspace, which therefore keeps its unexplained origin.

**So the principal confounder identified last night is eliminated.** Eshelby
equivalence remains true in principle — an eigenstrain can imitate an inclusion
— but the actual crystallographic elasticity of this specimen does not produce
the observed defect, and cannot be what the early rank-3 subspace represents.

What that subspace *is* remains open, along with why the late component does not
coincide with the DIC localisation.
