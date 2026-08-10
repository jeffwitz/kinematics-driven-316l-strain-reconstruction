# ADR 0010 — Preserve the production nonlocal coupling strategy as the robustness reference

## Status

Accepted.

## Context

The nonlocal model couples the local constitutive response to a scalar field
through a Helmholtz problem. The qualified production strategy solves this
coupling inside every mechanical Newton evaluation. A separate experimental
driver briefly used a simplified loop that alternated one mechanical update
and one nonlocal update. That loop was not equivalent in robustness, iteration
hierarchy, transactions or convergence criteria.

The distinction matters independently of the coupling formulation. A new
solver may replace the coupling algorithm, but it must not accidentally
replace the robustness strategy.

## Decision

The canonical partitioned reference is the nested strategy documented in
{doc}`../reference/numerics/nonlocal_fixed_point`:

```text
mechanical Newton trial
    └── fully converged constitutive/nonlocal fixed point
            ├── trial MFront integration
            ├── Helmholtz solve
            ├── bounded relaxation or Aitken acceleration
            └── nonlocal convergence test
```

The shared `evaluate_nonlocal_fixed_point()` implementation is the source of
truth. Production solvers and robustness references must reuse it instead of
copying a local `p -> H^-1 p` loop. An intentionally simplified candidate must
be named as experimental and must not be called the production or reference
staggered method until equivalence is demonstrated.

The following are solver-level robustness contracts, not constitutive-law
features:

- committed-state MFront transactions and exact `revert()` on rejected trials;
- line-search and admissibility checks;
- cutback and restoration after failure;
- separate mechanical and nonlocal convergence criteria;
- relaxation/Aitken policy and fixed-point diagnostics;
- physical parameter units, layout conventions and loading path.

The reference path and a candidate path must use the same physical problem and
must be compared for solution agreement before performance claims are made.
The reference need not be the fastest path; it must be qualified and robust.

This contract applies equally to J2, SRIX, Méric–Cailletaud and future
constitutive behaviours. A constitutive law supplies response and tangent
information; it does not silently create a third global coupling strategy.

## Consequences

The nested production strategy is the robustness reference for the spectral
solver. A monolithic `(u, chi)` Newton method is a candidate algorithm under
qualification, not a replacement reference merely because it uses fewer outer
iterations.

Benchmarks must identify the reference and candidate explicitly. Any change
to transactions, tolerances, cutbacks, field ordering, time increments,
length scales or coupling moduli is a numerical-method change and must be
documented as such.

The P43 M20 nested run is archived as an initial qualification result; its
timing is evidence, not part of this architectural decision.
