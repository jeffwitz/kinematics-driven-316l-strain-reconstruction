# ADR 0011 — Causal TANN-FCC constitutive identification from DIC

## Status

Accepted.

## Context

The phase-space campaign of 2026-08-17 closed the discovery path: the
effective inelastic field reconstructed from the DIC is a kinematic
inverse (the displacement-to-eigenstrain map has a measured nullspace, and
no discovered local state — scalar, tensorial, resistance-like, dynamical
or path-memory — predicts the response across held-out increments; the
known SRIX structure fitted on the experimental power does not transfer
either). The constitutive law must therefore be tested inside the
equilibrium problem, judged by the displacement it predicts.

The naive response — train a spatial network to reproduce the reconstructed
plastic field — would supervise the law with a non-unique, non-constitutive
target and would let every image own its plastic state. Both are rejected.

## Decision

The next constitutive model is a **causal TANN-FCC**:

```text
q_0 --[TANN / equilibrium]--> q_1, u_1 --> ... --> q_40, u_40
```

* **Causality is structural.** `Y_{n+1} = Integrate(F_theta, Y_n, loading)`.
  The state at `n+1` can only come from the state at `n` through an
  explicit path evolution imposed by the architecture. No per-image free
  state, no interior DIC as an input, no frame index, no coordinates.
* **Crystallography through the twelve FCC systems**, one law shared by all
  of them (the measured invariance: a single shared law reaches 88-90 % of
  the best per-system law), with a permutation-invariant network — the
  system numbering carries no constitutive meaning.
* **Thermodynamics is architectural**, not a loss penalty: a
  GENERIC-type structure with free energy `Psi_el + 1/2 ||z||^2`,
  generalised forces `A = -dPsi/dq` and a mobility
  `M = L L^T` makes `D >= 0` identically, and — deliberately — allows part
  of the work to be stored in the latent variables instead of forcing each
  system to dissipate on its own. The f_0 lesson is registered as a bar,
  not ignored: a model whose improvement comes with the slip channel
  carrying less than 10 % of the generalised work is a nonlinear elastic
  surrogate, not a plastic law.
* **Spatiality is deferred.** T0 has no learned spatial convolution. A
  future spatial operator (intragranular slip-trace kernel, then intergrain
  transport) will only ever provide a *context* to the TANN — the
  temporal transition remains `Y_n -> Y_{n+1}` through the TANN alone.
* **The solver is untouched.** The qualified Dirichlet spectral solver
  provides equilibrium; the TANN provides the constitutive law; the
  comparison is the DIC the model has not seen, under the qualified noise
  model.

The full frozen protocol, bars and prohibitions live in
`validation/tann_fcc_preregistration.md`.

## Consequences

* The reconstruction (Krylov + projection) keeps its role — a kinematic
  diagnostic of what the DIC admits — and is never a training target.
* A qualified T0 that fails its bars is a scientific result: a local causal
  FCC law, at this capacity, does not explain the DIC — the recorded entry
  point for the spatial-context stages.
* The material must satisfy the existing transactional contract
  (evaluate/commit/revert) and expose an exact algorithmic tangent; the
  trajectory adjoint is qualified on a small gate before any long
  training.

## Files

`src/fem_inhouse/constitutive/tann_fcc*.py`,
`src/fem_inhouse/identification/tann_fcc_*.py`,
`scripts/qualify_tann_fcc_material.py`, `scripts/qualify_tann_fcc_adjoint.py`,
`scripts/train_tann_fcc_p43.py`, `validation/tann_fcc_preregistration.md`.
