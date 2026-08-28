# From measured kinematics to a mechanical field

**Category: Explanation.** How can measured surface displacements become a
mechanically admissible stress and strain field?

## What DIC supplies

Digital image correlation (DIC) supplies in-plane displacements on a regular
measurement grid. Differentiating them produces a kinematic strain estimate,
but it does not enforce equilibrium, a constitutive relation, or consistency
between neighbouring material points. Measurement noise is also amplified by
differentiation.

The DIC field is therefore an observation, not by itself a mechanical
solution.

## The boundary-value problem

The software maps the measurement grid to a structured plane-stress finite
element mesh. DIC displacements are imposed **only on the external boundary**.
Interior displacements remain unknown and are found from equilibrium. Strain
follows from the finite-element kinematics, and stress follows from the local
constitutive model.

```{graphviz}
digraph reconstruction {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fillcolor="#eef5fb", color="#2980b9"];
  dic [label="Measured DIC\nboundary displacement"];
  maps [label="Local reconstruction\ndescriptors"];
  bvp [label="Plane-stress\nboundary-value problem"];
  fields [label="Mechanically admissible\nU, E, S and PEEQ"];
  dic -> bvp;
  maps -> bvp;
  bvp -> fields;
}
```

This is not a spatial filter. The reconstructed interior is constrained
simultaneously by compatibility, the material law and force balance.

## Heterogeneous local descriptors

The fields $\sigma_{y0}(x,y)$ and $K(x,y)$ describe the local response used by
the reconstruction. They preserve spatial information associated with the
experiment, but they are effective descriptors rather than independently
identified grain properties. The current model remains isotropic J2
plasticity at every point.

That distinction limits prediction: the maps are derived from an experiment
that has already occurred. Their role is to reconstruct that experiment, not
yet to predict a new one from microstructure alone.

## What is compared with DIC

The primary comparison uses the same historical equivalent total-strain
operator on DIC and FEM displacements. The DIC field and FEM field therefore
represent the same observable. PEEQ is different: it is a path-dependent
internal variable and has no direct experimental counterpart in this dataset.

Exact axes, units, derivative conventions and tensor completion are specified
in {doc}`../reference/input_contract`,
{doc}`../reference/scientific/observation_operator` and
{doc}`../reference/tensor_conventions`.

## Conclusion

> The software does not smooth a measured strain field. It reconstructs a
> field that satisfies compatibility, the constitutive law and equilibrium.

The next question is whether that local mechanical baseline is trustworthy.
Continue with {doc}`local_baseline`.
