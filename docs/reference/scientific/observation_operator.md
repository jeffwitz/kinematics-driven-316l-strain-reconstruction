# Observation-operator contract

**Mode:** reference  
**Domain:** measurement

The observation operator maps a mechanical displacement/strain field to the
quantities measured by the DIC chain. It owns component selection, physical
crop, valid mask, interpolation and uncertainty convention. It must not apply
an implicit spatial registration or reorient EBSD data.

Field comparisons must record the operator, mask, axes and units in the case
manifest. Definitions of the resulting comparison metrics are in
{doc}`../evidence/validation_metrics`.
