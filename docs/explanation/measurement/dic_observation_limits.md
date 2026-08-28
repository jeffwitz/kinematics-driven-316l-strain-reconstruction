# What DIC observes, and what it does not

**Mode:** explanation  
**Domain:** measurement

DIC observes image motion, not mechanics. The chain is

```text
images -> optical flow / DIC -> measured displacement -> differentiation
```

Differentiation produces a noisy strain estimate; it does not turn the DIC
field into a measurement of stress, plastic slip or equilibrium.  Those
quantities are inferred by the mechanical model.

For a fair comparison the model follows the opposite path:

```text
mechanical displacement -> observation operator O -> observable prediction
```

The same crop, mask, interpolation, component convention and units must be
used on both sides.  Comparing a raw finite-element field with a field already
filtered and resampled by DISFlow can introduce wavelength-dependent amplitude
and localisation bias.  This is why the operator is part of the calculation
manifest rather than an implicit plotting step.

Repeated-frame differences quantify measurement sensitivity on the valid mask;
they are not confidence intervals for the constitutive parameters.  The
loading path also matters: a displacement observed at one frame cannot stand
in for the unrecorded intermediate history used by a path-dependent material.
No post-filtering of EVM or plastic fields is allowed unless it is declared as
a new observation operator and qualified separately.

The stable contracts are in {doc}`../../reference/scientific/observation_operator`
and {doc}`../../reference/data/dic_axis_conventions`.  Their limits and
available evidence are indexed from the {doc}`../../evidence/index` portal.
