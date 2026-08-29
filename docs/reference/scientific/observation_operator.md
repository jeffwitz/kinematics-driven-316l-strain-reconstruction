# Observation-operator contract

**Mode:** reference  
**Domain:** measurement

The observation operator maps a predicted mechanical field to the quantities
represented by the DIC measurement. Conceptually,

$$
y_{DIC}=O_{DIC}(u)+\text{noise}.
$$

For a declared case, $O_{DIC}$ owns:

* component selection and the canonical `[u_x,u_y]` convention;
* physical crop, support and valid mask;
* interpolation or resampling between mechanical and observation grids;
* units and the declared spatial-transfer convention;
* the uncertainty/whitening convention when one is part of the comparison.

The operator must be applied to model predictions and sensitivity columns using
the same support and transformations as the registered measurement. It must
not apply a second transfer to already observed data.

## Separate contracts

Measurement registration and mechanical observation are different contracts:

```text
EBSD/DIC physical registration  ≠  O_DIC mechanical observation
O_DIC                           ≠  crystal rotation Q
O_DIC                           ≠  constitutive inference
```

The EBSD assignment convention $F$, scan geometry and $Q_{global\to material}$
belong to the EBSD input contract. They must not be hidden inside the DIC
operator. Likewise, an EVM or strain-like field obtained after differentiation
is an observed quantity with its own transfer and noise; it is not a direct
measurement of plastic strain.

Every field comparison must record the operator version, mask, axes, crop,
units, interpolation and uncertainty convention in the case manifest.
Definitions of comparison metrics are in
{doc}`../evidence/validation_metrics`; DIC component conventions are in
{doc}`../data/dic_axis_conventions`.
