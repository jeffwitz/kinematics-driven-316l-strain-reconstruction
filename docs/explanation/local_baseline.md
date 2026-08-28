# Establishing the local baseline

**Category: Explanation.** Is the remaining DIC discrepancy a coding
difference, or a limitation of the local constitutive model?

## Historical Abaqus-oriented table versus InHouse/Table

The historical reference represents J2 plasticity with a tabulated
Ludwik-Hollomon hardening curve. The in-house generator reproduces the
**available Abaqus-oriented table definition**: 1000 rows, the documented
plastic-strain grid and the same Ludwik values.

| Assumption | Historical definition | InHouse/Table |
|---|---|---|
| Kinematics | small strain | small strain |
| Mechanical hypothesis | plane stress | plane stress |
| Plasticity | associative isotropic J2 | associative isotropic J2 |
| Hardening | tabulated Ludwik law | same generated table |
| Stress unit | MPa | MPa |
| In-plane shear strain | engineering shear | engineering shear |

This is a definition-level verification, not an Abaqus finite-element
comparison. The original Abaqus input model, ODB and extraction procedure are
not available, so mesh, boundary and output parity cannot be audited end to
end.

:::{admonition} Claim boundary
:class: warning

The available Abaqus-oriented table definition has been reproduced in the
in-house implementation. Abaqus/Table versus InHouse/Table finite-element
parity is **not demonstrated**, because the original model and ODB extraction
are unavailable.
:::

## InHouse/Table versus InHouse/MFront

The analytical MFront law implements the same local scientific model without
the 1000-segment table. The material-point comparison gives:

```{include} ../_generated/local_material_point_metrics.inc
```

The independent DIC-driven finite-element comparison gives:

```{include} ../_generated/local_fem_metrics.inc
```

| Check | Compared quantity | Result |
|---|---|---|
| material histories | stress, plastic strain and PEEQ | within declared thresholds |
| finite-element fields | U, S, E, PE, PEEQ and RF | within declared thresholds |
| constitutive tangent | path-wise and finite-difference diagnostics | consistent with each implementation |
| plane-stress state | in-plane fields and transverse residual | within declared tolerances |

The exact numbers are generated in {doc}`../reference/evidence/evidence_registry`.
The comparison has a deliberate model boundary: below the former tabulation
cap, the table and analytical law represent the same baseline; beyond it, the
analytical law continues while the historical table plateaus.

## Why MFront is nominal

MFront makes the law explicit, removes the memory-heavy table and provides a
consistent tangent through a standard constitutive interface. It is therefore
the nominal implementation. The independent table path remains a regression
oracle.

The switch changes implementation and performance, not the scientific
reference model. MGIS transactions, Kelvin conversion and the alternative
three-dimensional condensation path are technical contracts documented in
{doc}`../reference/numerics/mfront_transaction`,
{doc}`../reference/tensor_conventions` and
{doc}`../reference/numerics/three_dimensional_condensation`.

## Conclusion

> The local baseline is sufficiently verified for the residual discrepancy
> with DIC to be interpreted as a model limitation, rather than a simple
> disagreement between two implementations.

That conclusion is the pivot of the project. Continue with
{doc}`missing_spatial_interaction`.
