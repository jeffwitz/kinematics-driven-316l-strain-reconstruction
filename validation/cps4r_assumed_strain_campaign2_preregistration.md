# CPS4R-AS — campaign 2 preregistration

Date: 2026-08-04
Written before any campaign-2 result exists.
Campaign 1 is frozen as **a preregistered near miss with no qualification**; see
`validation/cps4r_assumed_strain_report.md`.

## What campaign 1 left

Two quantitative SRIX criteria missed, both narrowly: cumulated slip at 1.17 %
against 1 %, and constitutive speed-up at 2.72–2.87 against 3.5. Several criteria
unevaluated. **The thresholds are not moved.** The result is good enough to
justify a second, independent campaign; it is not good enough to rewrite the
first.

## Four questions, and nothing else

### Q1 — is 1.17 % a formulation floor or a discretisation error?

Same domain, same perturbation, same increments, at `12x12`, `24x24`, `48x48`.
CPS4 and CPS4R-AS at each level.

- error falls below 1 % → the miss was spatial;
- error settles near 1.1–1.2 % → **that is the formulation floor**;
- error grows → campaign 1 was a favourable case.

All three outcomes are publishable and the third is the one worth naming in
advance.

### Q2 — how much of it is temporal?

`N_inc = 8, 16, 32`, **identical for both formulations**. The purpose is to
attribute the error, not to buy a pass with more increments: a variant that only
meets the bound at a higher increment count than CPS4 is not qualified, and that
was already registered as falsifier F3.

### Q3 — is the spectral floor conditioning or mechanics?

`relative_floor` over `1e-8`, `1e-6`, `1e-4`. **No optimum is sought.** Recorded
at each value: `E_Gamma`, Newton iterations, count of eigenvalues lifted, norm of
the stabilising force, reactions, and the slip localisation. If the verdict moves
across that range the floor is doing mechanical work and the formulation must be
re-derived rather than tuned.

### Q4 — does it hold on real crystal heterogeneity?

The decisive gap in campaign 1: its SRIX case is a **homogeneous** crystal. At
minimum a two-orientation checkerboard, four synthetic grains, an oblique
interface aligned on element boundaries, and a synthetic polycrystal of ten to
twenty orientations. Governing criterion stays `E_Gamma < 1 %`, and beyond it:
active-system count, sorted spectrum of the twelve slips, maxima near the grain
boundaries, reactions, band orientation.

No case places two orientations inside one element. That limit belongs to mesh
refinement, not to hourglass control, and was registered as such.

## The Newton question, tested before campaign 2 concludes

CPS4R-AS needs 47 Newton iterations against 37, which is what pulls the
constitutive speed-up below its bound while the per-call count stays exactly one.
The stabilisation force depends on the projected current tangent,
`f_stab(u, C(u))`, but the matrix handed to Newton differentiates it holding `C`
fixed and so drops `(df_stab/dC)(dC/du)`.

The test: at a **plasticised SRIX state**, compare the finite-difference
derivative of the *complete element internal force* — physical plus stabilisation,
with the constitutive law re-integrated at each perturbation so `C` moves — against
the matrix actually assembled. The existing element tests check the algebra for a
*given* tangent and cannot see this term. This quantifies the 47 iterations
instead of hypothesising about them.

## A variant, in its own preregistration and not a silent substitution

`assumed_strain_energy_lagged`: the projected tangent evaluated once at the start
of the increment from the converged previous state, **frozen through Newton**, and
refreshed after the increment converges. Then `dC_stab/du = 0` during Newton and
the stabilisation force has an exactly consistent derivative.

Registered predictions: fewer iterations, a better constitutive ratio, and no
spectral decomposition per iteration. Registered risk: it lags the onset of
plasticity, so `E_Gamma` may worsen. It is compared **without post-hoc tuning**
and does not replace the current formulation unless it wins on both.

## What does not change

CPS4 remains the reference. CPS4R-AS is not authorised for any scientific
conclusion. Thresholds are those of
`validation/cps4r_assumed_strain_preregistration.md`, unmodified.
