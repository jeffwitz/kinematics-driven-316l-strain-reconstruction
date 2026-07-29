# The missing spatial interaction

**Category: Explanation.** What systematic feature of the DIC field does the
verified local model fail to reproduce?

## The local failure is morphological

The original comparison placed raw FEM EVM beside DIC EVM already transformed
by the image-correlation chain. In regions containing coherent deformation
bands, that comparison showed:

- excessive peaks;
- bands that are too thin;
- excessive contrast between active and inactive regions;
- an overly concentrated plastic support;
- different conclusions from amplitude and localization metrics.

The representative comparison below uses a band-containing calibration region.
Its campaign identifier is retained only in the evidence provenance.

```{figure} ../_static/evidence/local_morphological_defect.png
:alt: DIC equivalent total strain, local FEM equivalent strain, signed error and local PEEQ on a band-containing region.
:width: 100%

This strictly local **raw-field** comparison motivated the spatial
investigation before introducing any coupled solution. PEEQ is an internal
model variable, not an experimental DIC field.
```

Lower global error is not sufficient evidence of a better localization. A
model may reduce peaks everywhere and improve L2 while erasing a physically
relevant band. Amplitude, localization and spatial-scale measures must
therefore remain separate.

The later symmetric image-level replay changes the magnitude of this defect:
DISFlow removes much of the fine raw-FEM structure and halves the local
relative L2 error. A residual localization discrepancy remains — the observed
local field predicts 16.1 % active area above the DIC q90 threshold instead
of 10 % — but the initial apparent peak and width error cannot be assigned
entirely to the constitutive law. The current quantified result is in
{doc}`current_evidence`.

## An output-only Helmholtz diagnostic

Before changing the constitutive model, the saved local FEM field is broadened
with

$$
\bar q-\ell^2\Delta\bar q=q,
\qquad \nabla\bar q\cdot n=0.
$$

This **output-only Helmholtz diagnostic** leaves displacements, equilibrium,
stress and plastic evolution unchanged. It asks only whether a missing spatial
width could explain part of the FEM-DIC discrepancy.

```{figure} ../_static/evidence/helmholtz_diagnostic.png
:alt: Raw local FEM field, Helmholtz-broadened diagnostic fields and DIC comparison.
:width: 90%

A positive diagnostic length improved several field and localization metrics
in the original raw-FEM comparison and transferred unchanged to a held-out
region.
```

The experiment supports the spatial-width hypothesis, but it cannot identify a
constitutive length. It operates after the mechanics and cannot redistribute
plastic evolution or forces. Because the historical diagnostic used the
asymmetric observation objective, its numerical optimum must not be reused as
a constitutive parameter after V3.

## Why post-filtering is not the final model

A scientifically useful spatial interaction must act while plasticity evolves.
It must change where the internal variable grows, preserve equilibrium and
remain part of the constitutive energy. Filtering the final EVM would instead
hide the local result without changing its mechanics.

## Conclusion

> Spatial broadening was a useful model-form diagnostic, but the image
> operator explains a major part of the original apparent defect. The
> remaining discrepancy may motivate a coupled interaction; it does not
> justify filtering the final result or reusing the old fitted length.

The diagnostic therefore motivates the coupled model introduced in
{doc}`micromorphic_model`.
