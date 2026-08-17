# FEMU-U identification of the crystal law

**Category: Explanation.** Why the forward identification loop exists, how it
is wired, and what it has and has not measured.

## Why this method, now

The phase-space campaign of 2026-08-17 established, negative by negative, that
the reconstructed effective inelastic field carries no local constitutive
closure: no discovered state (scalar, tensorial, resistance-like, dynamical or
path-memory) predicts the response across held-out increments, and the known
SRIX structure fitted on the experimental plastic power does not transfer
either. The conclusion of that chain is the forward path: the law must live
**inside the equilibrium problem** and be judged by the displacement it
predicts, not by a reconstructed field it should reproduce.

FEMU-U (finite element model updating on displacements) is the minimal form of
that path: run the full finite-element solve with the law, compare the solved
displacement to the measured one, adjust the law's parameters by least squares.
The smoke registered here frees only two parameters, `(tau0, Q)`, to ask the
cheapest possible question first: **does freeing a small parameter freedom and
fitting displacements help at all?**

## The loop

```text
theta -> partition solve (SRIX, measured Dirichlet data)
      -> stitched displacement U
      -> J(theta) = |U - U_meas|^2 / |U_meas|^2
      -> central finite-difference gradient on log(theta)
      -> L-BFGS-B with bounds +/-1.5 around the defaults
```

Each objective evaluation is one `fem-inhouse partition` invocation on a
prepared case; the references are the python J2 baseline and the
default-parameter SRIX. The protocol and the frozen bars live in
`validation/srix_femu_smoke_preregistration.md`.

## The files

| role | path |
|---|---|
| the loop driver | `scripts/srix_femu_smoke.py` |
| prepared case (historical band window) | `data/processed/femu_hist20` |
| cropped raw case source | `data/raw/case_study_hist20` (nodes 1600-1621 x 1050-1071 of the full ROI) |
| preregistration / results | `validation/srix_femu_smoke_preregistration.md`, `validation/srix_femu_smoke_results.md` |
| law and backend | behaviour `fcc_forest_rubin_srix`, backend `mfront-3d-condensed-plane-stress`, library `build/mfront/src/libBehaviour.so` |
| parameter contract | canonical names with units: `tau0_mpa`, `Q_mpa` (also `R_mpa`, `C_mpa`, `b`, `d`) |
| parameter provenance | `src/fem_inhouse/core/srix_parameters.py` |

## What the smoke measured, and what it did not

The loop is fully operational — the optimisation runs, the manifests are
safe, the references computed. The verdict so far is a **wiring finding,
not a law finding**: the partition workflow imposes the measured
displacement as the full nodal field, so every law reproduces it to the
solve tolerance (`~1e-9`) and the misfit carries no constitutive
sensitivity (the fit never moves). The instrument the day's reconstructions
used — measured kinematics on the **boundary only**, interior free — is the
missing piece. Until the boundary-only enforcement is wired, the FEMU-U
reading cannot speak about SRIX.

## Where this sits

This is the forward counterpart of {doc}`parameter_identification` and of
the phase-space analyses recorded under `validation/phase_*_results.md`.
The day's chain — reconstruction, FCC decomposition, LOSO ladders, the
closure check, the power identification, and this loop — is the argument
for why the constitutive structure must be tested inside the equilibrium
problem, and this file is its operational entry point.
