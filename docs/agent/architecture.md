# Durable cross-cutting decisions

This page records rules that affect several modules or agent workflows. A
domain chapter may add detail, but should not silently contradict these
contracts without an ADR and a new validation artifact.

## Evidence and status

Current implementation and current committed validation artifacts are the
primary evidence. A result is not a qualification merely because a script
completed or an optimizer returned `success`. Reports must distinguish:

```text
forward convergence
constitutive replay convergence
sensitivity qualification
optimization stationarity
experimental interpretation
```

Historical or superseded campaigns remain useful for diagnosis, but cannot
override a newer artifact or the current code.

## Spatial and batch ordering

The spectral two-state solver stores samples as `(x, y, subcell, component)`
and flattens material batches in C order. Its strain and stress reshape
contract is independent of the EBSD element-order option.

The classical `StructuredMesh` uses its own element numbering and may require
Fortran-order flattening for element fields. The two conventions must be
connected by an explicit mapping; changing the EBSD element order must never
implicitly change the spectral material-batch reshape.

Any change to an orientation or field mapping requires a non-square sentinel
test and, when relevant, an old-versus-corrected forward diagnostic. A better
fit alone is not evidence for a spatial transform.

## FEMU and sensitivity

The global sensitivity must use the same residual, assembly, quadrature,
degrees of freedom and boundary conditions as the reference forward. REGM or
another mechanical surrogate is not a substitute for the FEMU Jacobian.

For shadow sensitivities, fixed-current-strain forcing and history advance are
separate operations. A failed shadow is a diagnostic failure of the
derivative implementation until it is localized; it is not permission to
relax the constitutive law or silently accept an unbalanced forward state.

## Observation and identification

The RAW FEMU objective is a displacement residual in physical units. EVM,
correlation, IoU and similar maps are diagnostics unless explicitly declared
as the optimization objective. A scalar numerical rescaling of the residual
is allowed for optimizer conditioning, but is not a noise model or covariance
model.

Displacement-only identification can contain structural gauge directions.
SVD/TSVD results must report the retained subspace and weak directions; they
must not be rewritten as claims that each individual physical parameter is
identified. The rank and the fixed weak coordinates must be recorded in the
corresponding validation artifact.

## Registration and constitutive interpretation

EBSD-to-mesh ordering, DIC/EBSD axes, crop origin, physical scale and the
sample/crystal frame are separate contracts. Proving one does not prove the
others. Experimental localization claims remain provisional until the relevant
registration provenance is established.

When a mapping is corrected, preserve the historical result as a control and
rerun the forward with fixed parameters before changing constitutive
parameters. This separates registration effects from parameter-fit effects.
