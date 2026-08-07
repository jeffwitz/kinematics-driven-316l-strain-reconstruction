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

## What was not run

C2b (material-frame closure deviation `delta`) was not measured: the fixed
point already differs from the reference at the first plastic increment, so
the frame measurement is secondary and the preregistration amendment should
decide its fate.

## Recording

- Behaviour: `mfront/Fcc316LForestRubinSrixGps.mfront` (kept, experimental).
- Passive bridge: `MFrontNativeGeneralisedPlaneStressBatch` in
  `src/fem_inhouse/core/mfront.py` (kept, gated by the factory key
  `mfront-native-generalised-plane-stress`, not a default).
- Script: `scripts/qualify_srix_umat_gps_closure.py` (reproducible; the
  numbers above are from its C1 run and the increment sweep).
- No JSON archive is written for this refuting outcome; the script exits
  non-zero on rejection.
