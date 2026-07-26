# Current evidence

**Category: Explanation.** What is established now, what is only supported,
and what remains unresolved?

```{include} ../_generated/current_conclusion.inc
```

## Demonstrated numerically

- The local finite-element solver and its constitutive backends are coherent
  within their declared tolerances.
- The local solution is overly concentrated in regions containing coherent
  deformation bands.
- Coupled micromorphic feedback redistributes PEEQ; it does not merely filter
  the final EVM.
- Excessive coupling can reduce global L2 while suppressing relevant band
  morphology.
- Equal-$A_\chi$ pairs do not produce identical reduced-fidelity responses.
  Spatial length therefore has an observable effect beyond amplitude alone.

The full-resolution local/coupled evidence quantifies redistribution without
using plot metadata as its primary source:

```{include} ../_generated/micromorphic_redistribution_metrics.inc
```

```{figure} ../_static/evidence/band_roi_evm_comparison.png
:alt: DIC and raw FEM equivalent total strain on a band-containing calibration region for increasing coupling.
:width: 100%

Raw coupled FEM fields compared with DIC on one common scale. No Helmholtz
post-filter is applied to the FEM EVM.
```

## Supported, but not final

- $H_\chi$ and $\ell$ play distinct roles in the present reduced-fidelity
  observations.
- Some explored regions show interior compromises between amplitude and
  localization objectives.
- Reproducing measured band width requires an effective spatial scale.

These statements are **supported**, not yet independently confirmed at high
fidelity and transferred.

## Not demonstrated

- one unique value of $H_\chi$;
- one unique value of $\ell$;
- a material internal length for 316L;
- unchanged-parameter transfer to an independent band-containing region;
- complete Abaqus parity;
- prediction before the experiment with the present local descriptor maps.

## Current numerical limitation

The homogeneous Newton-25 design currently resolves two interior amplitude
optima while leaving the shortest-length profile censored:

```{include} ../_generated/identifiability_status.inc
```

The short-length, high-coupling corner is censored by mechanical convergence in
the homogeneous F1 design. This is not interpreted as a physical boundary.
Until that part of the parameter domain is either solved robustly or excluded
for a physical reason, the objective surface is incomplete.

The generated {doc}`../reference/claims_matrix` and
{doc}`../reference/evidence_registry` are the source of detailed status and
provenance. Historical campaign narratives are retained outside the public
reading path.

## Conclusion

> Current evidence distinguishes a spatial-length effect from coupling
> strength, but it does not identify a transferable material length.

The final chapter, {doc}`scope_and_prediction`, states what the software can
claim and what extension is required next.
