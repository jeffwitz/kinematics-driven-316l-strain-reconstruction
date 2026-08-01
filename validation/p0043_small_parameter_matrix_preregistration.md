# Small (ell, alpha) matrix on P43 and selection tool — preregistration

Date: 2026-08-01
Specification: short specification of 2026-08-01, "Matrice (ell, alpha) sur P43
et outil de sélection".

**Validated 2026-08-01, including corrections C1 to C4. Execution started the
same day; the 13 missing points plus the reproducibility replicate are driven by
`scripts/run_p0043_parameter_matrix.py`.**

This campaign closes the P43 exploration phase with one of three registered
outcomes: a provisional parameterisation, a zone of indistinguishable
parameters, or an explicit statement that the observable cannot select. It is
forbidden to conclude that the nonlocal formulation works or fails in general.

## Four corrections to the specification, and why

Written before the matrix exists, so none of them can be a reaction to a result.

### C1 — There is no reduced-fidelity tier. The whole matrix runs at full fidelity

Specification section 3 asks for a reduced-mesh first pass; section 4 requires
the symmetric observation operator. **These are incompatible**, and the
incompatibility is already settled in
`micromorphic_symmetric_identification_preregistration.md`:

> The symmetric operator warps a displacement field onto the reference image
> and re-observes it through DISFlow. That is only defined when **one element is
> one pixel**. The configured `spatial_reduction: 2` low-fidelity tier is
> therefore incompatible with the registered objective and is **not used**.

A reduced mesh forces the raw comparison that lot V3 showed changes amplitude,
morphology **and the ranking of coupling candidates**. Using it would reproduce
the error that suspended identification in the first place.

The reduced tier is not needed. Measured on the archive: a coupled P43 pad150
run at 20 increments takes `2171 s`, a local run `939 s`. **Thirteen new
coupled runs is about 7.8 h sequential**, or roughly 40 min of wall clock on a
13-way job array. The DISFlow replays cost about a minute each.

Consequence: the two-stage design of section 3 is dropped. There is no
promotion step, and section 16's "at most four promoted cases" does not apply.
This also removes the fourth promotion rule, "a candidate showing a different
trade-off on the front", which is a judgement rather than a rule and could not
have been preregistered honestly.

### C2 — `D_self` is the repetition residual, not the DIC against itself

Section 10.1 offers either. They are different campaigns: the DIC against
itself gives exactly `0` on all four indicators, the repetition residual gives
the achievable floor. **The repetition residual is used**, because a
normalisation anchored at an unreachable zero overstates how far every
candidate is from the best possible.

Registered construction, from `dic_measurement_chain_results.md`: a synthetic
displacement perturbation with per-component standard deviations of
`0.0403 px` in row and `0.0624 px` in column, spatially correlated with a
`1/e` length of `38.2 px`, added to the DIC displacement. The correlation
length matters — that residual is **not** white noise, and a white perturbation
would be filtered away by `H_49` and would understate the floor. Ten
realisations, seed `20260801`, `D_self` is their median.

Recorded honestly: the measurement-chain report calls this pair a
noise-and-drift **upper bound**, not a certified noise floor. `D_self` is
therefore an upper bound on the floor, which makes every `Z` a **lower** bound
on the normalised defect. Stated so it is not read as a certified distance.

### C3 — The normalisation is anchored on the controls, and that is a declared weakness

Section 10.1 sets `D_null` from the best of the negative controls. Each
indicator is then divided by a different arbitrary number, and section 10.3
takes a maximum across them. **The minimax answer therefore depends on how bad
the controls happen to be**, not only on the candidates.

Concretely: the homogeneous control carries `5.8 %` of the DIC high-pass strain
energy at `8 px`, so its `D_presence` is about `|log 0.058| = 2.85`, while its
shape defect is moderate. `Z_presence` is compressed and `Z_shape` stretched by
construction.

This is kept, because the alternative — an absolute scale — does not exist for
these indicators. It is bounded three ways:

1. **the control used is declared per indicator** and reported in the output,
   as the specification requires;
2. **raw values are always reported** beside the normalised ones, and the
   Pareto front of section 10.2 is computed on the **raw** defects, where the
   normalisation cannot act. Only the minimax tie-break of 10.3 uses `Z`;
3. **the selection is repeated with `D_null` taken from the other control** for
   every indicator. If the retained candidate changes, the selection is
   reported as **normalisation-dependent** and the outcome is Case B or C.

Point 2 is a deliberate departure from section 10.2, which puts the front on
`Z`. Domination is invariant under any per-indicator monotone rescaling, so the
front is identical either way; computing it on raw values makes that invariance
visible instead of implicit.

### C4 — Registered prediction: `D_localisation` will discriminate poorly and is inverted

Measured on the already archived fields, FSS at the q90 DIC threshold and
`48 px`, the scale nearest the registered `49`:

| | local | alpha = 1 | alpha = 2 | alpha = 4 | translated |
|---|---:|---:|---:|---:|---:|
| FSS q90 (48 px) | `0.687` | `0.689` | `0.670` | `0.599` | `0.504` |

`D_localisation = 1 - FSS` therefore separates the local model from `alpha = 1`
by `0.002`, and **prefers less coupling**, the same inversion already found on
IoU q90. In a minimax it will become the worst defect of the high-`alpha`
candidates and push selection downwards, against `D_amplitude`.

**This is registered now so that the resulting tension is not later reported as
a physical result.** Part of it is a property of the indicator. `D_localisation`
is kept, because section 9 decides what is kept and it has not been run yet;
but if the campaign ends on a trade-off between amplitude and localisation, the
report must cite this paragraph.

## Amendment A1, written 2026-08-01 while the matrix runs, before it is read

**Which field `D_localisation` scores.** Section 6.3 of the specification says
"FSS at the DIC absolute q90 threshold, scale 49 px" without naming the field,
while 6.1 and 6.2 name `g_49` explicitly. The ambiguity is resolved **in favour
of `g_49`**, so all four indicators read one observable and the fluctuation
framing of section 5 is not broken halfway through.

Recorded honestly: **correction C4 was calibrated on the other reading.** Its
numbers — FSS q90 of `0.687` for the local model against `0.689` for
`alpha = 1` — come from the archived FSS on the **EVM** field, not on `g_49`.
The prediction that `D_localisation` discriminates poorly and is inverted in
`alpha` therefore rests on the EVM variant and may not transfer.

Both are computed. `g_49` is the registered primary; the EVM variant is
reported beside it as a secondary, so C4 can be checked on the data it was
formed on. If the two disagree about which candidates are preferred, that
disagreement is reported and no ranking rests on `D_localisation` alone.

## Amendment A2, written 2026-08-01 while the matrix runs, before it is read

**The spatial bootstrap uses 49 px square tiles, not blocks of 8.** Section
10.4 carries "block 8" over from the section bootstrap of the v1 and v2
campaigns, where a block was 8 consecutive centreline sections spaced 4 px, so
about 32 px of band. Here the resampling is two-dimensional over the core, and
an 8 px tile is far below the measured `38.2 px` coherence of this observable:
resampling at that size would treat correlated pixels as independent and
understate the uncertainty, which is the one direction a stability criterion
must not err in.

Tiles are therefore squares of `49 px`, the principal scale and the nearest
round figure above the coherence length, giving 42 whole tiles on the
`360 x 310` core. Tile sizes `32` and `96 px` are reported as sensitivities.
Draws, `10 000`, and seed, `20260801`, are unchanged.

Each defect is recomputed on the resampled pixel multiset. This is exact rather
than approximate for all four: shape is a correlation, amplitude a quantile,
presence a ratio of sums of squares, and localisation a ratio of spatial means
of the fraction fields, which are computed once on the whole core at the
registered neighbourhood size and only then averaged over the drawn tiles.

## Amendment A3, written 2026-08-01 after the fourth point failed to converge

`(alpha = 4, ell = 20 um)` does not converge under the registered settings. It
cut back from increment 9 onwards, halving the step until it fell below the
registered minimum divisor, and stopped with
`NonlinearConvergenceError` after 39.6 min. The failure is the solver's, not the
tooling's: the log shows a clean cutback cascade.

**A non-converged point is reported and excluded from the selection. Its solver
settings are not changed to make it converge.** Retuning increments or
tolerances for one point would make it incomparable with the fifteen others and
would be precisely the per-point adjustment this protocol exists to prevent. A
parameterisation that cannot be computed under the settings every other point
uses is not a candidate for a provisional parameterisation.

Consequences, stated now:

- the count of usable points drops, and the report gives the converged and
  non-converged sets explicitly;
- **the iso-`Achi` test at `1600` loses one of its two members**, since that
  pair is `(alpha = 4, ell = 20)` and `(alpha = 1, ell = 40)`. If the point
  stays non-converged the pair cannot be compared, and only the `800` pair
  remains;
- non-convergence is itself reportable. Where it falls in the `(ell, alpha)`
  plane is a property of the formulation at these settings and belongs in the
  heat maps, marked as such rather than left blank without comment.

This amendment is written before any indicator has been computed on any matrix
point. What is known at this moment is a solver outcome, not a score.

## Amendment A4, written 2026-08-01 during the end-to-end rehearsal

**The zone is built from paired differences, not from overlapping bands.**

The first implementation put a candidate in the zone when its `5 %` quantile
sat below the best candidate's `95 %` quantile. That is the interval-overlap
fallacy: the draws share their resampled tiles, so
`J_inf(candidate) - J_inf(best)` has far less spread than either score alone,
and two bands can overlap comfortably while the paired difference never comes
near zero.

A candidate now joins the zone when the bootstrap interval of that paired
difference contains zero.

Recorded plainly because of the direction it cuts: the paired test is stricter,
so it **shrinks** the zone and makes the campaign more decisive. It was decided
while rehearsing the pipeline on the points computed so far, which means partial
scores were on screen. It is adopted because comparing marginal quantiles of
paired samples is wrong whatever it returns, not because of what it returned;
no indicator, threshold or bound is touched.

## Inventory of reusable computations

| Point | Status | Source |
|---|---|---|
| `ell = 58.88`, `alpha = 1` | **archived** | `results/constitutive-nonlocal-p0043-pad150-a100` |
| `ell = 58.88`, `alpha = 2` | **archived** | `.../-a200` |
| `ell = 58.88`, `alpha = 4` | **archived** | `.../-a400` |
| local reference | **archived** | `results/constitutive-local-p0043-pad150` |
| homogeneous control | **archived** | `results/control-homogeneous-local-p0043-pad150` |
| translated control | **archived** | `results/control-translated-maps-local-p0043-pad150` |
| the other 13 matrix points | to compute | — |

Observations under both DISFlow profiles are archived for the four models in
`dic_symmetric_observation_p0043_v1`, and for both controls in
`observed_evm_controls_p0043_v1`. **Nothing already computed is recomputed.**

## The matrix

`alpha` is a multiplier of the reference coupling modulus
`Hchi_ref = 5168.147582748343 MPa`, which is how the archived runs are
parameterised. `ell` is `length_scale_mm`.

| `ell` (um) | `alpha = 0.5` | `alpha = 1` | `alpha = 2` | `alpha = 4` |
|---:|:---:|:---:|:---:|:---:|
| 20 | compute | compute | compute | compute |
| 40 | compute | compute | compute | compute |
| 58.88 | compute | **archived** | **archived** | **archived** |
| 90 | compute | compute | compute | compute |

`Hchi` in MPa: `2584.0738`, `5168.1476`, `10336.2952`, `20672.5903`.
`length_scale_mm`: `0.020`, `0.040`, `0.05888`, `0.090`.

Plus the local model as reference, and the two negative controls. Increments
fixed at **20**, matching the archived runs so their three points are directly
comparable, and padding 150 on P43 throughout.

### The matrix contains two exact iso-`Achi` pairs, and that is a registered test

The degeneracy feared by the identification preregistration is along
`Achi = Hchi * ell^2`. On this grid, `Achi` is proportional to `alpha * ell^2`:

| `Achi` (arbitrary units) | points |
|---:|---|
| `800` | `(alpha = 2, ell = 20)` and `(alpha = 0.5, ell = 40)` |
| `1600` | `(alpha = 4, ell = 20)` and `(alpha = 1, ell = 40)` |

Two further pairs sit `8.3 %` apart: `(2, 40)` against `(1, 58.88)`, and
`(4, 40)` against `(2, 58.88)`.

**Registered prediction**: if the observable constrains only `Achi`, the two
exact pairs are indistinguishable under every indicator, within the stability
bands of section 10.4. If they separate, `Hchi` and `ell` act separately on
this observable. Either answer is reported. This does not replace the 22-point
identifiability campaign; it is a partial and cheap test of the same question,
obtained for free from a grid designed for something else.

## Observation operator and comparison support

Primary comparison through the symmetric operator only:
`U_FEM -> synthetic image -> DISFlow -> U_FEM,obs`. **No primary indicator
compares the DIC to a raw FEM field.** Same support, same mask
(`declared_all_valid`), same axis convention, same pixel size `1.84 um`, same
differentiation as the historical EVM operator: `np.gradient` with array axis
0 = canonical x, then `cell_average`, fields differentiated on the solve grid
and cropped to the core afterwards.

Primary DISFlow profile `legacy_script_2021`, patch 4 stride 1, by provenance.
`declared_medium_v4` is a registered sensitivity, not a tie-break.

## The scalar fluctuation field

`epsilon = sym(grad u)` on the observed displacement, then the declared
high-pass `H_49(f) = f - G_49 * f` with a Gaussian of standard deviation
`sigma = 0.5 * s`, and `g_49 = ||H_49(epsilon)||_F`.

Principal scale **`49 px`**, the measured MTF-50 of the chain. Sensitivities at
`32` and `96 px`. **Scales `8` and `16 px` play no part in selection**: the
coupled fields carry `10` to `18 %` of the DIC high-pass energy there and the
replay adds no speckle-decorrelation noise, so those scales measure missing
measurement noise as much as missing model content.

No filter is ever tuned per candidate.

## Indicators

Four defects, all lower-is-better.

| Symbol | Definition |
|---|---|
| `D_shape` | `1 - pearson(g_49 FEM, g_49 DIC)` |
| `D_amplitude` | `abs(log(q95(g_49 FEM) / q95(g_49 DIC)))` |
| `D_localisation` | `1 - FSS_q90(49)`, threshold frozen from the DIC |
| `D_presence` | `abs(log(R))`, `R = ||H_49(eps FEM)||^2 / ||H_49(eps DIC)||^2` |

`D_presence` is the guard rail that the Frobenius distances lacked. It is **not
merged** with the other three: it enters the Pareto front as a fourth
coordinate and the minimax as a fourth term, but no weighted sum is formed
anywhere.

Secondary, reported and never used for selection: EVM relative L2, EVM
correlation, IoU q90, q90 and q95 of `g_49`, FSS at `32` and `96 px`, high-pass
energy at `32` and `96 px`, iteration counts and cutbacks, PEEQ mean, q95 and
maximum.

## Indicator validation, before the matrix is read

The four indicators are applied to nine cases first:

1. DIC against itself;
2. DIC perturbed by the repetition residual (C2);
3. homogeneous control;
4. translated-map control;
5. amplitude scaled by `0.8` and by `1.2`;
6. one band displaced by `16 px`;
7. one band removed;
8. two bands merged;
9. a spurious band added.

Accepted only if all of the following hold:

- identity is optimal on every indicator;
- **the homogeneous control fails on presence and on amplitude**;
- **the translated control fails on shape or on localisation**;
- an amplitude error does not look like a position error, and conversely;
- a removed band scores worse than a moderate amplitude error;
- conclusions are qualitatively stable across `32`, `49` and `96 px`.

An indicator that fails stays in the report as a diagnostic and **leaves the
selection**. Which ones survive is not known now.

The band-level perturbations act on the displacement inside a hard-edged
corridor, whose boundary step inflates the score; as in the gradient
diagnostic, **only the sign of those cases is read**, never the magnitude.

## Normalisation, front and selection

`Z_k = (D_k - D_k^self) / (D_k^null - D_k^self)`, with `D_self` from C2 and
`D_null` declared per indicator, subject to C3.

Pareto front on the **raw** four defects; a dominated candidate is discarded.
Among the non-dominated, the minimax `J_inf = max(Z_shape, Z_amplitude,
Z_localisation, Z_presence)`; the retained candidate minimises its worst
normalised defect. No weighted sum, at any point.

## Stability, and what a "zone" means

The selection is repeated under: paired spatial bootstrap, block 8, 10 000
draws, seed `20260801`; both DISFlow profiles; scales `32`, `49`, `96 px`.

- **robustly preferred**: minimises `J_inf` in at least `95 %` of draws;
- **preferred**: at least `80 %`;
- otherwise, a zone of indistinguishable parameters.

**Expected outcome, registered in advance: Case B.** An argmin over seventeen
neighbouring candidates in a smooth parameter space is a fragile statistic —
in v1, `alpha = 2` against `alpha = 4` already gave `P = 0.937` on a single
metric with 10 000 draws. Reaching `95 %` here would be surprising.

A zone is reported as **the set of points reaching `J_inf` within the
bootstrap band of the minimum**, listed point by point. It is **not** reported
as a range of `ell` crossed with a range of `alpha`: a non-dominated set on a
grid need not be a rectangle, and a bounding box would claim points that were
never preferred.

### Solver reproducibility floor

The bootstrap is spatial and says nothing about the solver's own
reproducibility, which on this partition reaches `15.82 %` on core PEEQ between
loading paths and `0.20 %` between 20 and 40 increments. One matrix point,
`(alpha = 2, ell = 40)`, is therefore recomputed at 40 increments, and the
resulting spread on each indicator is reported as a floor drawn on every heat
map. **Neighbouring points differing by less than that floor are declared
indistinguishable regardless of the bootstrap.**

## Matrix analysis

Per indicator: an `(ell, alpha)` heat map, cuts at fixed `ell`, cuts at fixed
`alpha`, finite differences along both, presence or absence of an interior
optimum, monotonicity, and interaction. The report answers explicitly: does
`alpha` act mainly on amplitude; does `ell` act mainly on spatial organisation;
does the best `alpha` depend on `ell`; is there a valley rather than a point;
is the preferred zone stable across indicators and profiles; can the local
model or a negative control still beat the micromorphic solutions.

The question about separating `ell` from `alpha` must be answered **through**
the iso-`Achi` structure above, not as if the grid were a clean factorial: at
fixed `alpha`, changing `ell` changes `Achi` as `ell^2`.

## The three allowed conclusions

**Case A, robust optimum** — one configuration stably minimises the worst
normalised defect and rejects both negative controls. It becomes the
provisional P43 parameterisation.

**Case B, robust zone** — several neighbouring configurations are
indistinguishable. A point list is reported, with no artificial choice.

**Case C, non-selective indicators** — the indicators do not distinguish the
configurations, or do not reject the controls. No parameterisation is selected,
**and no new criterion development is opened in this campaign.**

## Limits, to be repeated in the report

P43 only. The selection is a provisional reconstruction parameterisation. It
does not demonstrate an internal length of 316L, does not demonstrate
transferability, and neither validates nor refutes the nonlocal formulation.
An independent validation remains necessary for any material conclusion.
Everything rests on one loading path and one ROI, and `D_self` is an upper
bound on the measurement floor.

## Relationship to the unlaunched 22-point campaign

`campaigns/mm_id_points.tsv` holds 22 points at `alpha` in `{1,2,3,4,6}` and
`ell` in `{20,30,40,50,58.88}`, preregistered for **identifiability** and never
launched. Six of its points coincide with this matrix.

Neither supersedes the other: that campaign asks whether `Hchi` and `ell` are
separately identifiable, this one asks whether a parameterisation can be
selected. Whichever runs first, its runs are reused by the other — the
configurations are byte-comparable, same padding, same increments. This
document does not authorise the 22-point campaign.

## Deliverables

`parameter_matrix.csv`, `indicator_matrix.csv`,
`normalised_indicator_matrix.csv`, `pareto_front.json`,
`bootstrap_selection.csv`, `selection_report.json`, `selection_report.md`, and
the seven registered figures, under
`reference_data/p0043_small_parameter_matrix_v1/`, in a commit separate from
this one.
