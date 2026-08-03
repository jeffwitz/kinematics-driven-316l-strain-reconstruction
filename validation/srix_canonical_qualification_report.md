# SRIX against Forest and Rubin: canonical qualification report

Date: 2026-08-03
Sections 8, 9, 10 and 13 of the 2026-08-03 specification.
Tests: `tests/unit/core/test_srix_canonical.py`,
`tests/unit/core/test_srix_symmetry_and_plane_stress.py`,
`tests/unit/core/test_fcc_interaction_matrix.py`
Archive: `validation/_generated/srix/srix_canonical_tests.json`

Everything here is independent of 316L. It qualifies the *model and its
integration*, so it can falsify the implementation while every material
parameter is still unsettled.

## Verdict

**Case B of section 19: the formulation and its integration are qualified; the
parameters are not.**

> The implementation reproduces the closed-form Forest-Rubin solution to machine
> precision, satisfies the thermodynamic and numerical requirements, respects
> cubic symmetry, and closes the plane-stress condition on an arbitrarily
> oriented crystal. Its parameters are configurable and traceable. What is not
> established is any 316L value: every registered set is a prior or a
> transposition, and the calibration is preregistered but not performed.

Two limitations are recorded below rather than smoothed over: an unresolved
index map in the symmetry comparison, and a step-size threshold below which a
reversal is qualitatively wrong.

## Section 8.1 — the analytical `[001]` solution

With hardening switched off, `[001]` tension admits an exact plateau. Eight
systems are active, each with `|m| = 1/sqrt(6)`, and together they produce the
axial plastic rate, so `gammadot = (sqrt(6)/8) eqrate`. Substituting into the
flow rule and cancelling the equivalent rate — which is what makes the law rate
independent — gives

$$\sigma = \sqrt{6}\,\tau_0 + \tfrac{6}{8}R .$$

The elastic constants do not appear.

| checked | result |
|---|---|
| plateau, four `(R, tau0)` pairs | relative `1e-16` |
| active systems | exactly 8; the other four exactly zero |
| slip per active system | identical to `1e-12` |
| `gamma = (sqrt(6)/8) x` axial plastic strain | to `1e-6` |
| convergence under refinement | monotone |

The relative overstress on that plateau is exactly the registered
`O_R = (sqrt(6)/8) R / tau_0`. That is not a coincidence of definitions: `O_R`
was defined to be this number, so the dimensionless label attached to a parameter
set is the relative overstress the model actually runs at in the reference
configuration.

## Section 8.2 — correspondence with Méric-Cailletaud

Covered by `tests/unit/core/test_forest_rubin_srix.py`: equation (16) reproduces
`18.7819100705 MPa` for `K = 12`, `n = 11`, `1e-3 s⁻¹`; the two laws agree at
`[001]` and are asserted to **disagree** away from it — 0.32 percent at `[001]`,
7.1 at `[111]`, 14.2 at `[123]`. The disagreement is asserted deliberately so the
correspondence cannot be quietly overstated into a general equivalence.

## Section 8.3 — published cases

**Not performed, and not faked.** Reproducing the copper and PWA1489 cases of
Forest and Rubin requires their parameter sets, which are not in hand.
Prohibition 16 forbids inventing missing coefficients, and inventing them to
produce a matching figure would be worse than the omission. This remains open
and is the most valuable single addition to this report.

## Section 9.1 — dissipation

`(tau_s - X_s) Dgamma_s >= 0` holds on every system at every increment, on
monotonic paths across three overstress moduli and through a full reversal. It
is not an assumption of the model but a consequence of its flow rule, so a
negative value would signal an implementation defect rather than an unusual
material — the check exists for that, and a deliberately sign-flipped input is
tested to confirm the check can fail.

The energy balance at 2 percent, historical set, in MPa (energy per unit volume):

| axis | elastic | stored isotropic | stored kinematic | dissipated |
|---|---:|---:|---:|---:|
| `[001]` | 0.160 | 0.004 | 0.071 | 2.116 |
| `[111]` | 0.132 | 0.014 | 0.053 | 3.727 |

The kinematic term is the recoverable `sum (C/2) a_s^2`, not the integral of
`X_s dgamma_s`: dynamic recovery dissipates the rest.

## Section 9.2 — overstress diagnostic

`eta_s = <|tau_s - X_s| - r_s> / max(r_s, eps)` is archived per increment with
its maximum, `q99`, `q95`, mean over active systems, the fractions above 1, 5
and 10 percent, and the active count. At 2 percent, historical set: `[001]` is
uniform at `0.132` by symmetry, `[123]` spreads from `0.382` at `q95` to `0.470`
at the maximum.

These are descriptive. The flow rule is linear in the overstress, so a large
value means the increment demanded a lot of slip, not that anything is wrong.

## Section 9.3 — time independence

Bit for bit, on identical strain discretisations with uniform, linearly ramped
and randomised pseudo-time, and across a thousandfold change of rate. Stresses
and all twelve slips compare with `assert_array_equal`, not with a tolerance.

## Section 9.4 — time convergence, and a finding

The monotonic branch converges cleanly at first order from the coarsest step, on
all four named orientations, already below `1e-3` at ten increments.

**The reversal has a step size below which it is qualitatively wrong.** Below
about twenty increments over the whole path, it produces **no reverse slip at
all**: the total slip stays exactly at its forward value and the back strain
never relaxes, instead of relaxing by an order of magnitude. This is not a large
error, it is a different solution, and it is why the specification asks for
monotonicity on the *last* refinements. Once the reverse yield point is resolved
— from about forty increments — convergence is monotone to `1e-2` on stress,
slip, back strain and dissipation.

Recorded rather than tuned away, with its own test. A campaign that reverses its
loading must check its increment count against this instead of assuming that
fewer increments merely cost accuracy.

## Section 9.5 — consistent tangent

Against global finite differences over a plateau of three perturbation
amplitudes, at a non-zero strain increment and on the condensed plane-stress
operator: below `1e-5` relative for the identity orientation and for a general
Bunge orientation. Several amplitudes are used because a single one cannot
distinguish a wrong tangent from a badly conditioned difference.

## Section 10 — cubic symmetry

The 24 proper rotations are generated as integer signed permutations with
determinant `+1`, their action on the twelve slip systems is derived from the
geometry, and that action is shown to be a bijection leaving the interaction
matrix **exactly invariant**.

On the behaviour, over `[001]`, `[011]`, `[111]` and `[123]`:

| checked | result |
|---|---|
| axial response under all 24 symmetries | invariant to `1e-9` |
| sorted spectrum of per-system slip | preserved to `1e-12` |
| number of active systems | preserved |
| small perturbation around each axis | below 0.5 percent |

**The open point.** The specification asks to compare slip families after
canonical reordering. The catalogue-level permutation is derived and correct,
but reconciling it with the order MFront reports slips in did not work out: no
rule tried — forward or inverse permutation, sign applied on either side — held
for all twenty-four symmetries, while the slip multiset was preserved to
`3.8e-13` throughout. The physics is symmetric; the index map is not derived.
The property is therefore asserted in its permutation-invariant form and the
limitation is written into the test. Establishing the index map would strengthen
this section; fitting whichever rule happened to pass would not.

## Section 13 — plane stress

Four orientations against four loadings, plus three overstress moduli: the three
out-of-plane components stay below `1e-6 MPa` throughout. An off-axis crystal
couples extension and shear where an aligned one does not — which is what makes
the condensation non-trivial and worth testing.

## Section 7 — the interaction matrix

Derived from geometry and checked against `mfront-query --interaction-matrix`
entry for entry. The finding: MFront splits the glissile junction into two ranks
by which system can glide it, so the **rank matrix is not symmetric** and the
numerical matrix is symmetric only because the publication's single glissile
coefficient is written into both slots. The shipped TFEL gallery literal is
identical to ours, slot for slot, in all three variants — a corroboration, not a
proof.

## Defects found and fixed during this qualification

**In the shipped code.** `mgis.load` does not give a private behaviour: two
loads of the same library, name and hypothesis return handles onto the same
object, so a `setParameter` through one is visible through the other
process-wide. The parameter-set feature had applied each batch's values once at
construction, so two batches with different sets would have silently shared
whichever was applied last. Parameters are now re-asserted before every
integration.

**In the compiled law.** The Young's modulus was `99950.29765841035` where
`C11 = 197000` and `C12 = 125000` give `99950.31055900622`, a `1.3e-7`
transcription error. Small, but it meant "apply the historical set" was not a
no-op. Corrected in both crystal laws; no test moved.

**In my own test code**, three times, each of which first looked like a broken
law: composing a symmetry as `Q S^T` instead of `S Q`; reading
`thermodynamic_forces` without rotating back, which reads the material frame and
reported `-sigma/2` for a symmetry sending `z` onto `y`; and evaluating the
tangent at a committed state, which returns the elastic operator and stops
testing the plastic branch.

## What remains open

1. The published copper and PWA1489 cases (§8.3), pending their parameter sets.
2. The slip-family index map under cubic symmetry (§10).
3. A comparison against a thin three-dimensional model with several layers
   through the thickness (§13's forward reference).
4. The 316L calibration itself, preregistered in
   `validation/srix_316l_calibration_preregistration.md` and not performed.
