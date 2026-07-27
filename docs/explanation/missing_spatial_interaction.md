# The missing spatial interaction

**Category: Explanation.** What systematic feature of the DIC field does the
verified local model fail to reproduce?

## The local failure is morphological

In regions containing coherent deformation bands, the local baseline tends to
concentrate plasticity into a support that is too narrow and too intense. The
main symptoms are:

- excessive peaks;
- bands that are too thin;
- excessive contrast between active and inactive regions;
- an overly concentrated plastic support;
- different conclusions from amplitude and localization metrics.

The representative comparison below uses a band-containing calibration region.
Its campaign identifier is retained only in the evidence provenance.

```{figure} ../_static/evidence/band_roi_alpha_summary.png
:alt: DIC equivalent total strain, local and coupled FEM fields, and internal PEEQ on a band-containing region.
:width: 100%

The left comparison establishes the local morphological defect. PEEQ is an
internal model variable, not an experimental DIC field.
```

Lower global error is not sufficient evidence of a better localization. A
model may reduce peaks everywhere and improve L2 while erasing a physically
relevant band. Amplitude, localization and spatial-scale measures must
therefore remain separate.

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

A positive diagnostic length improves several field and localization metrics
on a selection region and transfers unchanged to a held-out region.
```

The experiment supports the spatial-width hypothesis, but it cannot identify a
constitutive length. It operates after the mechanics and cannot redistribute
plastic evolution or forces.

## Why post-filtering is not the final model

A scientifically useful spatial interaction must act while plasticity evolves.
It must change where the internal variable grows, preserve equilibrium and
remain part of the constitutive energy. Filtering the final EVM would instead
hide the local result without changing its mechanics.

## Conclusion

> The improvement obtained by spatial broadening suggests that the local law
> lacks a spatial interaction, but it does not justify filtering the final
> result.

The diagnostic therefore motivates the coupled model introduced in
{doc}`micromorphic_model`.
