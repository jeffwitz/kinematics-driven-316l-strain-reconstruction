# Full-field Méric--Cailletaud qualification

The registered generic backend was exercised through the complete spectral
stack on the P43 M20 EBSD crop `(1610:1630, 1075:1095)`, eight increments,
four MFront threads and one BLAS/FFTW thread. The independent reference is the
raw Méric 3-D law with external Python condensation.

| quantity | 3-D + condensation | generic StructuralPlaneStress3D |
| --- | ---: | ---: |
| accepted increments | 8 | 8 |
| global Newton iterations | 50 | 50 |
| iterations per increment | `[5,6,6,6,6,6,7,8]` | `[5,6,6,6,6,6,7,8]` |
| final residual | `2.69e-12` | `2.70e-12` |
| elapsed time | `4.38 s` | `1.99 s` |
| material time | `3.66 s` | `1.28 s` |

Relative field differences (generic against condensation) are:

| field | relative L2 error |
| --- | ---: |
| displacement | `9.03e-15` |
| stress | `3.80e-11` |
| reactions | `3.62e-11` |
| accumulated slip | `2.99e-11` |
| signed/equivalent slip arrays | `5.27e-11` |

The raw reports and field arrays are archived as:

```text
validation/_generated/performance/meric_m20_condensed.json
validation/_generated/performance/meric_m20_condensed.fields.npz
validation/_generated/performance/meric_m20_structural.json
validation/_generated/performance/meric_m20_structural.fields.npz
```

No additional host adapter or Méric-specific FEM path was introduced. Both
runs use the same factory, spectral solver, orientation provider, MGIS batch
contract and observables. The result therefore qualifies the statement that
the registered full-field StructuralPlaneStress3D backend has been exercised
with SRIX and Méric--Cailletaud through the same host implementation.

## Generated-shell V1 contract

The mathematical row/Jacobian transformation remains independent of the flow
rule. The current source generator is intentionally narrower and now fails
fast unless the source explicitly provides:

```text
@DSL Implicit
@ModellingHypothesis Tridimensional
@Brick StandardElasticity
FCC g[Nss] state and dg increments
the generated FCC SlipSystems helper
```

This is the current packaging contract needed to reconstruct structural
strain outputs. A source outside this FCC-shell contract receives a clear
error rather than failing in an opaque source replacement. Removing this
packaging restriction remains a separate future MFront/TFEL interface task.
