# SRIX generalised plane stress via UMAT closure — qualification preregistration

Date: 2026-08-07
Written before any UMAT-closure run has been executed.

## Purpose

The monolithic generalised-plane-stress contract — plane stress enforced inside
the law's own local Newton — has two candidate implementations:

1. the TFEL generator fork `jeffwitz/tfel-generalised-plane-stress`
   (prototype `ffcdcb3`), which modifies MFront itself; and
2. a stock-MFront behaviour variant that receives the rotation
   `Q_global_to_material` as nine per-point material properties, applies `Q` to
   the imposed gradient itself, and enforces the three transverse stress
   equations on the **global** stress inside its implicit Newton.

This document freezes the qualification of route 2 before any number exists.
Route 1 is parked, for a scientific reason this campaign also measures: the
generator prototype enforces the closure in the law's frame, which the law
alone cannot reconcile with the structural frame. The reference backend
`MFront3DCondensedPlaneStressBatch` enforces the closure in the **global**
frame — verified in `src/fem_inhouse/core/mfront.py`: the bridge rotates the
strain into the crystal frame before integration (`mfront.py:1446-1457`),
rotates stress and tangent back to global after (`mfront.py:1489-1502`), and
the closure Newton iterates on the global-frame stress and `Cbb`
(`mfront.py:1962-2012`). Any replacement backend must reproduce that fixed
point.

The FEM bridge for the UMAT route is passive: it applies **no** rotation, runs
**no** Python closure, and only discards the out-of-plane block of the
returned stress and tangent. This is "do the global closure in the UMat" made
correct: the law is the only component that knows `Q`.

## Design frozen for this campaign

- New behaviour file `mfront/Fcc316LForestRubinSrixGps.mfront`, compiled under
  `Tridimensional` (unchanged law body and parameter sets), with:
  1. nine dimensionless material properties `Q11` … `Q33`, convention
     `Q_global_to_material` (the repo convention in
     `core/crystal_orientation.py`; the law applies `Q deto Q^T` to the
     imposed gradient inside the integrator);
  2. three additional state variables: the **global** transverse strains
     `ezz`, `eyz`, `exz`;
  3. three residual equations
     `(Q^T sigma Q)_zz = (Q^T sigma Q)_xz = (Q^T sigma Q)_yz = 0` in the same
     `@Integrator`, with analytic derivatives (the transverse rows of the law's
     3D tangent, rotated by `Q`);
  4. consistent tangent via the DSL machinery (no hand-written
     `@TangentOperator`), like the current law.
  Local system dimension: 21 (`deel` 6 + `dg[12]` 12 + closure 3).
- The bridge variant rejects a rotation argument (no double rotation: the
  batch must never rotate what the law rotates itself) and sets `Q` per point
  from the existing provider
  (`provider.rotations_global_to_material(point_count)`).
- `Fcc316LForestRubinSrix.mfront` and `Fcc316LMericCailletaud.mfront` are not
  modified. The generator fork is not used.

## What is already established, and therefore not under test

- The SRIX law is qualified as case B: formulation and integration are sound,
  **no 316L parameter is identified** (2026-08-03 journal, §9). The parameter
  sets used below are the registered ones
  (`316l_srix_transposed_from_nasri2018_rate_1e-3` and the paired set used by
  `scripts/qualify_crystal_tet2_p43.py`), provenance by group, none claiming
  an identification.
- `MFront3DCondensedPlaneStressBatch` is the reference and enforces the global
  closure (code references above). Its batch-closure sibling matched the
  reference at about `1e-11` relative on P43 M100 EBSD
  (`srix_monolithic_plane_stress_architecture.md`).
- The Kelvin partition and the Schur complement are established
  (`condense_kelvin_tangent_to_engineering`,
  `test_schur_complement_recovers_isotropic_plane_stress_elasticity`).
- The batch machinery already sets per-point material properties
  (`create_plane_stress_material_batch` passes per-point arrays).
- The SRIX integrator was written with this closure in mind: its `Deq` is
  built from the unknowns so that "every update of the transverse strain
  changes `deel` and therefore `Deq`" (comment in
  `Fcc316LForestRubinSrix.mfront`).
- Orientations are exercised through `rotation_from_euler_bunge_deg` in
  `tests/unit/core/test_srix_symmetry_and_plane_stress.py`.

## Hypotheses

**H1 — fixed-point equivalence.** On the same material point, the UMAT-closure
variant returns the same transverse strains, in-plane stress and full 3D
stress as the reference backend.

**H2 — tangent equivalence.** The DSL-returned condensed tangent equals the
reference Schur complement, and agrees with finite differences at the law's
established floor.

**H3 — frame correctness.** With `Q` known, the **global** transverse stress
is below tolerance at convergence, and the difference between a global-frame
closure and a material-frame closure on a rotated crystal is measured and
interpreted as preregistered below (F6).

**H4 — robustness.** The 21-unknown Newton converges wherever the reference
nested Newton converges, with the same set of accepted increments and no
additional cutbacks.

## Registered test cases

**C1 — single point, identity.** `Q = I`, registered parameter set
`316l_srix_transposed_from_nasri2018_rate_1e-3`, temperature `293.15 K`. Load
history covering, in order: purely elastic states, first slip activation, and
established multi-system plasticity (including the analytically known plateau
`sigma = sqrt(6) tau0 + (6/8) R` of the law's §8 qualification). The exact
history is frozen in the generating script.

**C2 — single point, tilted.** Same history, `Q` from Bunge
`(35, 20, 15)` degrees — the homogeneous orientation of the archived P43
profiles. This is the frame-sensitive case.

**C3 — single point, generic tilt.** Same history, `Q` from Bunge
`(54.7, 45, 10)` degrees (already used in the symmetry suite), so that the
rotation is not axis-aligned.

**C2b — frame measurement.** On C2 and C3 only: the same law **without** `Q`
(material-frame closure — the naive UMAT variant, and what the generator
prototype would produce) is run and the deviation it leaves in the global
frame is measured. This is a registered measurement, not a candidate.

**C4 — small field.** A small structured field (12×12) with two orientations
(identity and `bunge_35_20_15` in halves), eight increments of a Dirichlet
case, compared at field level against the reference backend. Mirrors
`test_3d_condensed_backend_matches_native_plane_stress_fem`.

**C5 — P43 M100 EBSD, 20 increments (M20).** The qualified configuration —
four MFront threads, one FFTW thread, LGMRES, Eisenstat–Walker, tangent
transverse predictor, production trial promotion — on the registered P43 crop
and the **EBSD map** (the EBSD map must not be replaced by the homogeneous
orientation `[35,20,15]`, per the 2026-08-07 reprise state). Compared against
the reference backend on the same case. This is the M20 gate of the
reprise document; M100 and M200 are explicitly out of scope.

## Registered metrics

Against the reference backend, per case:

- in-plane stress, relative `L2` and maximum absolute (MPa);
- full 3D stress tensor, relative `L2`;
- transverse strains (Kelvin components 2, 4, 5), relative `L2` and maximum
  absolute;
- condensed tangent `3x3`, relative `L2` and maximum absolute;
- **global** plane-stress residual at convergence, maximum absolute (MPa);
- local Newton iterations per point and local failures (the law's own);
- C4/C5 field level: displacement, reaction forces, in-plane stress, plastic
  slip, accumulated slip, equivalent slip — relative `L2` (the field set of
  `scripts/benchmark_srix_ebsd_condensation_blocks.py`);
- C4/C5: set of accepted increments and cutbacks;
- material seconds (reported only; the backend-selection decision, including
  performance, is separate);
- C2b: `delta`, the maximum absolute global transverse stress left by the
  material-frame closure solution, and its ratio to the local tolerance.

## Registered acceptance criteria

Derived, not chosen by taste. The batch-closure sibling of the reference
matched at about `1e-11` relative on the same case; the existing unit tests
accept `rtol 1e-7` on J2 histories.

**A1** — C1–C3: in-plane stress and transverse strains, relative `L2`
`<= 1e-9` — two decades of margin above the `1e-11` demonstrated level, for a
different Newton path, and one decade tighter than the unit-test level.

**A2** — C1–C3: condensed tangent, relative `L2` `<= 1e-8` — one decade
looser than A1, tangents being derivatives.

**A3** — every converged point: maximum absolute global transverse stress
`<= 1e-6 MPa` — the level the existing native-versus-condensed test asserts
(`max |plane_stress_residual_mpa| < 1e-6`).

**A4** — C4/C5 field level: displacement, stresses and slips, relative `L2`
`<= 1e-9` (same derivation as A1).

**A5** — identical accepted-increment sets on C4/C5; zero local failures on
any point where the reference converged.

**A6** — finite-difference cross-check of the returned condensed tangent
agrees to relative `1e-6` — one decade of margin above the `7e-7` floor of
the law's own tangent qualification.

## Registered falsifiers

**F1 — fixed-point or tangent mismatch.** Any C1–C3 point exceeding A1, A2 or
A3 refutes H1/H2. The report must state which: a stress agreement with a
tangent disagreement points to the derivative blocks; a stress disagreement
points to the closure assembly.

**F2 — robustness.** The 21-unknown Newton fails (non-convergence or failed
solve) on any point where the reference nested Newton converges: H4 refuted,
not qualified.

**F3 — field drift.** Any difference in accepted increments or additional
cutback on C4/C5 (A5 fails): not qualified.

**F4 — derivative defect.** The finite-difference check (A6) fails while A1
passes: the closure residual derivatives are wrong; the tangent is
untrustworthy and the variant is not qualified.

**F5 — double rotation.** The passive bridge must reject a rotation argument;
an automated test asserts it. If the guard is absent or a run silently
double-rotates, the campaign is invalid — a design defect, not a measurement.

**F6 — frame measurement, preregistered interpretation.** If `delta` (C2b)
is above `1e-6 MPa`, the frame distinction is material at the closure
tolerance: any material-frame closure — including the generator prototype as
it stands — is invalid for rotated crystals and must not be presented as
generalised plane stress. If `delta` is at or below `1e-6 MPa`, the
distinction is below tolerance for these orientations; this is reported as
such, and the global-frame requirement is not relaxed.

## What this campaign cannot conclude

- It does not select M100 or M200. C5 is the M20 gate of the 2026-08-07
  reprise document; M100/M200 remain forbidden until this campaign passes and
  a separate decision is recorded.
- It does not identify SRIX parameters. Case B stands: the parameter sets are
  registered priors, nothing more.
- It does not validate the thin-sheet plane-stress model itself. It only
  verifies that the new backend enforces the same structural condition as the
  reference.
- It does not authorise the generator fork. A pass is evidence that the fork
  is unnecessary for this goal; the frame measurement (F6) is the registered
  evidence about the fork's own closure frame.
- No wall-clock claim is made. Timings are reported; the selection decision
  (including performance) is a separate, later decision.
- A pass authorises the variant for plane-stress SRIX campaigns of the P43
  kind. It does not authorise the micromorphic coupling with crystal
  behaviours (refused by design) nor any 3D solver.

## Recording

Results go to `validation/srix_umat_gps_closure_results.md`, including
refuting outcomes and `delta`. The generating script is
`scripts/qualify_srix_umat_gps_closure.py`; its JSON output and the archived
fields are stored beside the report with a manifest. The behaviour file is
`mfront/Fcc316LForestRubinSrixGps.mfront`, added to the explicit build list.
The passive bridge replaces the current experimental
`MFrontNativeGeneralisedPlaneStressBatch` gate (which consumed a
non-existent MGIS binding) or is added alongside it; the factory key is
unchanged. `Claude.md` is updated and the work committed, per repo rule.
