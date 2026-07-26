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

```{figure} ../_static/evidence/band_roi_evm_comparison.png
:alt: DIC and raw FEM equivalent total strain on a band-containing calibration region for increasing coupling.
:width: 100%

Raw coupled FEM fields compared with DIC on one common scale. No Helmholtz
post-filter is applied to the FEM EVM.
```

## Supported, but not final

- Versioned constant-$A_\chi$ and fixed-alpha comparisons produce distinct
  reduced-fidelity fields and spatial metrics. This supports a length effect
  beyond coupling strength alone.
- $H_\chi$ and $\ell$ therefore play distinct roles in the present
  reduced-fidelity observations.
- Some explored regions show interior compromises between amplitude and
  localization objectives.
- Reproducing measured band width requires an effective spatial scale.

These statements are **supported**, not statistically identified. The
constant-$A_\chi$ differences have not yet been compared with complete mesh,
DIC-resolution and between-region uncertainty envelopes.

## Current discriminating campaign

The table below is generated from the immutable execution attestation:

```{include} ../_generated/current_identification_metrics.inc
```

The short-length failures are numerical censoring, not a physical boundary.
No high-fidelity manifest was generated from this incomplete objective surface.

## Not demonstrated

- one unique value of $H_\chi$;
- one unique value of $\ell$;
- a material internal length for 316L;
- unchanged-parameter transfer to an independent band-containing region;
- complete Abaqus parity;
- prediction before the experiment with the present local descriptor maps.

## Current numerical limitation

The short-length, high-coupling corner is censored by mechanical convergence in
the homogeneous F1 design. Until that part of the parameter domain is either
solved robustly or excluded for a physical reason, the objective surface is
incomplete.

The generated {doc}`../reference/claims_matrix` and
{doc}`../reference/evidence_registry` record detailed status, machine-checked
assertions and provenance. Historical campaign narratives are retained outside
the public reading path.

## Conclusion

> Current evidence supports a spatial-length effect distinct from coupling
> strength, but it does not identify a transferable material length.

The final chapter, {doc}`scope_and_prediction`, states what the software can
claim and what extension is required next.
