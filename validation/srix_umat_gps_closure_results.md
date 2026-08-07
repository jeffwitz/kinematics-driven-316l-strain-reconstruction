# SRIX UMAT GPS closure — qualification results

Date: 2026-08-07
Preregistration: `validation/srix_umat_gps_closure_preregistration.md`.

**Verdict: F1 triggered. The UMAT-closure backend is not qualified for
fixed-point agreement with the reference.** The closure machinery itself is
correct and verified; the failure is a property of the SRIX law, not of the
implementation.

## What was verified and works

- The behaviour `Fcc316LForestRubinSrixGps` compiles and runs (21-unknown
  local system: `deel` 6 + `dg[12]` 12 + closure 3).
- **The closure is enforced in the structural frame**: the converged final
  stress satisfies `sigma_zz = sigma_xz = sigma_yz = 0` to `1e-14 MPa`
  (probe and MGIS runs agree).
- The passive bridge applies no gradient rotation, sets the nine `Q`
  components as per-point material properties, and its in-plane stress,
  transverse strain and closure residual match the reference at the
  **elastic** increment (`C1`): stress `7e-15`, tangent `4e-11`, residual
  `2e-15 MPa`.
- The tangent correction (in-plane rotation operator) is implemented and
  verified at the elastic state.

## What failed — F1 (fixed-point agreement)

From the **first plastic increment** on, the UMAT solution diverges from the
reference and the difference grows with the number of increments:

| increments | max in-plane stress error (MPa) |
| --- | ---: |
| 12 | 18 |
| 24 | 37 |
| 48 | 77 |
| 96 | 160 (and a local Newton failure) |

At increment 2 of the frozen history (identity orientation):

| quantity | reference (condensed) | UMAT closure |
| --- | ---: | ---: |
| in-plane stress (MPa) | `[147.15, 7.39]` | `[159.04, 8.89]` |
| `eps_zz` | `-0.00165` | `-0.00262` |
| accumulated slip | `0.0046` | `0.0084` |
| own closure residual | `~1e-8 MPa` | `1e-14 MPa` |

**Both states are roots of the same discrete equations.** The raw 3D law
(`Fcc316LForestRubinSrix`, driven through the same history with the UMAT
transverse strains imposed) converges to a state with `sigma_zz = -154.7 MPa`
at the UMAT closure point — a different root. A direct C++ probe of the
generated UMAT behaviour confirms its converged state satisfies its own
residual system (status 0, `sigma_zz = -2.8e-14`). The SRIX law therefore
admits **multiple roots at the first plastic increment** (different active
system sets under the Macaulay clamp); the block Newton of the reference and
the joint 21-unknown Newton of the UMAT select different branches. This is
the multiple-solution behaviour already documented for this law in the
2026-08-03 journal ("un renversement ne produit aucun glissement inverse —
pas une grosse erreur, une autre solution").

## Consequences

- A1/A2 (fixed-point and tangent agreement at `1e-9`/`1e-8`) fail from the
  first plastic increment. The backend is **not** authorised for production
  campaigns.
- The negative result is kept: the UMAT route is not a drop-in replacement
  for `MFront3DCondensedPlaneStressBatch` while the branch selection differs.
- The closure-in-the-law strategy itself is not refuted: it enforces the
  structural closure in a single Newton, exactly as designed. The open
  question is the branch selection of the SRIX law under a joint Newton, a
  property of the law, not of the closure.

## Update 2 — 2026-08-07: two rotation bugs found and fixed; robustness wall

Re-examination after the amendment (preregistration Amendment 1) uncovered
two storage-convention bugs that had corrupted the rotated orientations
(the identity cases were blind to both):

1. **The law's rotation formula used the engineering shear storage.** MGIS
   stores gradients, thermodynamic forces and state variables in the Kelvin
   storage (shear = `gamma/sqrt(2)`). The tensor rotation in the stored
   components mixes diagonal and shear with `sqrt(2)` factors, not `1` (and
   the shear-to-shear mixing carries no factor). The corrected formula
   (`gpsRotate` in `Fcc316LForestRubinSrixGps.mfront`) matches the MGIS
   rotation on all six components to machine precision. The user's hint to
   re-check the component ordering was verified: the ordering is the
   standard `[11, 22, 33, 12, 13, 23]` (confirmed by the qualified
   `kelvin_3d_to_tensor`), and the `sqrt(2)` factor was the error.
2. **The bridge's in-plane operator was transposed.** The MGIS
   `rotateGradients` output has the rotated unit vectors as ROWS; the
   derivative operator needs them as COLUMNS. At the identity the transpose
   is a no-op, which is why the error only showed on rotated orientations.

After both fixes, at a rotated (`[35, 20, 15]`) plastic material point: the
closure residual is `1.3e-14 MPa` and the finite-difference tangent agrees
to `1.3e-7` relative.

**Performance.** Per material evaluation (single point, four threads, the
working range): the UMAT backend takes `0.42 ms` against `2.81 ms` for the
condensed reference — **about 6.7x faster**, as expected for one Newton
instead of a nested one.

**Robustness wall (open).** The joint 21-unknown Newton fails at the deeper
plastic states (increment 8 of the frozen 12-increment history, and the deep
points of the P43 20x20 case at 8 increments). The failure is independent of
the start (the GPS fails at increment 8 even from the reference's committed
state), of the Jacobian (analytic or finite-difference), of `@IterMax` (200)
and of the closure normalisation (modulus `1e6`). The nested reference
converges there. The wall is a property of the joint Newton structure in the
Implicit DSL, not of the closure formulation; it limits the backend to the
moderate plastic range until addressed (or documented as the acceptance
range of Amendment 1).

## Recording

- Behaviour: `mfront/Fcc316LForestRubinSrixGps.mfront` (kept, experimental).
- Passive bridge: `MFrontNativeGeneralisedPlaneStressBatch` in
  `src/fem_inhouse/core/mfront.py` (kept, gated by the factory key
  `mfront-native-generalised-plane-stress`, not a default).
- Script: `scripts/qualify_srix_umat_gps_closure.py` (reproducible; the
  numbers above are from its C1 run and the increment sweep).
- No JSON archive is written for this refuting outcome; the script exits
  non-zero on rejection.
