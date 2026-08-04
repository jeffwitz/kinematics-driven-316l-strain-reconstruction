# Limited-memory Broyden for CPS4R-AS — results

Date: 2026-08-04
Preregistration: `cps4r_as_broyden_preregistration.md`, frozen before any
accelerated solve existed and not touched since.

**Verdict: rejected.** Three falsifiers fire. The correction is not adopted, the
default of `jacobian_correction` stays `none`, and CPS4R-AS keeps the 47
iterations this work set out to reduce.

## What was measured

The registered SRIX case: Bunge (35, 20, 15), 12x12, non-affine boundary, eight
increments, `assumed_strain_energy` with the ASMD projection.

```bash
MFRONT_BEHAVIOUR_LIBRARY=$PWD/build/mfront/src/libBehaviour.so \
PYTHONPATH="$HOME/.local/lib/python3.12/site-packages" \
LD_LIBRARY_PATH="$HOME/.local/lib" \
.venv/bin/python scripts/qualify_broyden_correction.py --repeats 5
```

Raw data in `validation/_generated/cps4r_as/broyden_correction_qualification.json`
and `..._tol1e-08.json`.

## Newton iterations, the criterion the work exists for

| variant | iterations | against the un-accelerated run |
|---|---|---|
| CPS4 (reference) | 37 | |
| CPS4R-AS, no correction | 47 | |
| CPS4R-AS + Broyden, m = 1 | **50** | +6.4 % |
| CPS4R-AS + Broyden, m = 3 | **57** | +21.3 % |
| CPS4R-AS + Broyden, m = 5 | **64** | +36.2 % |

The bound was 40 or fewer. Every memory makes it worse, and monotonically: more
secant pairs, more iterations. That monotonicity is the finding, not noise —
iteration counts here are deterministic, and the same ordering reappears at a
residual tolerance of `1e-8` (56 without, then 59, 60, 71) and on a 6x6 mesh
with four increments (20 without, then 20, 21, 22).

## Every preregistered criterion

Five repetitions, medians, as preregistered.

| criterion | bound | no correction | m = 1 | m = 3 | m = 5 |
|---|---|---|---|---|---|
| `E_u` against the un-accelerated run | < 1e-6 | — | 6.6e-10 ✓ | **2.3e-5 ✗** | **2.3e-5 ✗** |
| errors against CPS4 | within 0.02 pt | — | 0.0000 ✓ | 0.0082 ✓ | 0.0082 ✓ |
| Newton iterations | <= 40 | 47 | **50 ✗** | **57 ✗** | **64 ✗** |
| additional cutbacks | none | 0 | 0 ✓ | 0 ✓ | 0 ✓ |
| additional constitutive calls | zero | — | 0 ✓ | 0 ✓ | 0 ✓ |
| constitutive speed-up | > 3.5 | 2.44 | **2.72 ✗** | **2.21 ✗** | **1.85 ✗** |
| total speed-up | > 1.8 | 2.10 | **1.79 ✗** | **1.39 ✗** | **1.25 ✗** |

Two remarks on the speed rows, so they are read for what they are.

The constitutive bound of 3.5 was never within reach: the **un-accelerated**
CPS4R-AS is already at 2.44, and that shortfall is the fact this whole work was
meant to repair. The row therefore records that the repair did not happen, not
that the correction caused the shortfall.

And `m = 1` shows a constitutive speed-up *above* the baseline while running
three more iterations, which is impossible if the timings were exact. Constitutive
time scales with the iteration count, so that inversion bounds the timing
repeatability of this machine at roughly 15 to 20 % even over medians of five.
The verdict rests on the deterministic criteria — iterations, cutbacks, field
errors — and the timings are recorded rather than relied on.

## Falsifiers

- **F3 — iterations increase.** Fires on all three memories.
- **F1 — the converged solution moves.** Fires on `m = 3` and `m = 5`.

F1 deserves its own paragraph, because the first reading of it is wrong. Both
runs stop on the same relative-residual test, so they land at two points of one
convergence ball, and a difference of the order of the tolerance is expected
rather than alarming. That is what `m = 1` does: `6.6e-10` at a tolerance of
`1e-6`, and `7.8e-12` at `1e-8` — it scales with the tolerance, so it is a path
difference and nothing more.

`m = 3` and `m = 5` give **`2.26e-5` at both tolerances**, unchanged by
tightening the stopping test a hundredfold, and identical to each other to three
digits. That is not a convergence-ball artefact: the accelerated runs settle on a
genuinely different equilibrium of the same discrete problem, about `2e-5` away
in displacement and `0.008` point away in the error against CPS4. The rate-
independent crystal law admits nearby equilibria differing in their active slip
set, and a different iteration path can land on a different one — that is the
explanation the evidence *suggests*, and it has not been demonstrated. It is
recorded as an open point, not as a conclusion.

- **F8 — the assembly cost eats the constitutive gain.** Fires, but it is
  redundant here: with more iterations the gain was never there to eat.

No falsifier fired for F2 (constitutive calls), F4 (cutbacks), F5 (rigid modes),
F6 (a singular reduced matrix without a clean fallback) or F7 (a memory chosen
after seeing results — the sweep was `1, 3, 5` as frozen, and nothing is adopted).

## Why it fails: the correction memorises the path

Section 23, `scripts/diagnose_broyden_directional_prediction.py`, one element,
the constitutive law re-integrated at every evaluation so `C` moves as it does in
a real iteration. A contracting sequence of iterates fills the memory; the
prediction of `dr` is then compared against the truth in two regimes.

| relative prediction error of `dr` | base `G_0` | corrected `G_0 + dG` |
|---|---|---|
| along a direction **in** the memory | 0.90 | **1.6e-15** |
| along a **fresh** direction | 0.99 | **1.52** |

The correction is exact where it was fitted and worse than useless where the
next step goes. It improved 41 % of fresh directions — indistinguishable from a
coin toss. Repeated over contractions `0.3, 0.6, 0.8` and memories `1, 3, 5`,
the pattern never breaks:

| memory | out-of-sample, base | out-of-sample, corrected | fresh directions improved |
|---|---|---|---|
| 1 | 0.85 – 0.99 | 0.81 – 0.93 | 47 – 62 % |
| 3 | 0.85 – 0.99 | 8.5 – 8.6 | 9 – 16 % |
| 5 | 0.85 – 0.99 | 1.5 – 29.6 | 9 – 41 % |

`m = 1` is roughly neutral out of sample and roughly neutral in the solver
(50 against 47). `m = 5` degrades out-of-sample prediction by up to thirty times
and is the worst in the solver (64 against 47). The element-level diagnostic and
the solver campaign agree, in ordering and in magnitude, which is what makes this
an explanation rather than a coincidence.

The reason is the nature of the term being learned. `(df_stab/dC)(dC/du)` is not
a fixed linear operator that five secant pairs can identify: `C` is the
algorithmic tangent of a **rate-independent** crystal law, so it jumps with the
active slip set rather than varying smoothly along the path. Fitting a `2 x 5`
operator to a contracting sequence in which `C` has changed discontinuously
produces a matrix that satisfies every stored condition and describes nothing.
The minimum-Frobenius solve does its job correctly — it is the premise that a
secant fit can capture this term which is wrong.

## What survives

Nothing here contradicts the measurement that motivated it: the assumed-strain
tangent *is* inconsistent, by 370 % on the stabilisation and 36 % on the total,
and that inconsistency *is* what costs CPS4R-AS its ten extra iterations. The
base reduced Jacobian is wrong by 0.85 to 1.2 in relative prediction error, which
this diagnostic confirms independently, from a different direction, on the real
crystal law.

What is refuted is one route to repairing it. A quasi-Newton correction learned
from the iteration's own history cannot do this job, because the thing to be
learned is not stationary over the history. Any future attempt should therefore
supply the missing term rather than fit it — an actual `dC/du`, or a lagged
formulation whose matrix is the exact derivative of the force it uses. The
lagged variant was implemented and does not converge (`f95d351`); that remains
the other open route, and it is open for a different reason.

## Code retained

`jacobian_correction` stays in the configuration with `none` as its default. The
correction, the reduced coordinates and the multisecant algebra are kept: they
are what makes this result reproducible, the directional diagnostic is reusable
against any future candidate, and a mechanism deleted after a negative result is
a negative result nobody can check.
