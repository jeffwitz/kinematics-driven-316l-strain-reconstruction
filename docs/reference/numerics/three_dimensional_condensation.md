# External three-dimensional condensation reference

**Category: Reference.**

This page specifies the independent numerical reference backend
`mfront-3d-condensed-plane-stress`. The complete scientific comparison with
the specialised GPS and generic `StructuralPlaneStress3D` routes is in
{doc}`mfront_structural_plane_stress`.

## Problem and local closure

The finite-element model supplies the structural in-plane strain

$$
\varepsilon_a=(\varepsilon_{xx},\varepsilon_{yy},\gamma_{xy}).
$$

The missing local constitutive components are

$$
\varepsilon_b=(\varepsilon_{zz},\gamma_{xz},\gamma_{yz}),
$$

and the plane-stress approximation imposes the traction-free condition

$$
\sigma_b=(\sigma_{zz},\sigma_{xz},\sigma_{yz})=0.
$$

The three transverse strains are local material-point unknowns, not additional
FEM degrees of freedom. For an arbitrarily oriented anisotropic or crystal
material, imposing only \(\sigma_{zz}=0\) is insufficient.

## Algorithm

The raw MFront behaviour remains a complete `Tridimensional` behaviour. For
each trial transverse strain, the bridge:

1. restores the same committed MGIS state;
2. rotates the structural gradient into the material convention;
3. integrates the 3D behaviour;
4. rotates stress and tangent back to the structural frame;
5. solves the local residual \(\sigma_b=0\).

After convergence, partition the structural tangent as

$$
C=
\begin{bmatrix}C_{aa}&C_{ab}\\C_{ba}&C_{bb}\end{bmatrix}
$$

and return

$$
\boxed{C^{PS}=C_{aa}-C_{ab}C_{bb}^{-1}C_{ba}.}
$$

The inverse is never formed. The implementation solves the small \(C_{bb}\)
systems and monitors their conditioning. A committed-state or tangent-based
transverse predictor can be selected. Complete transactional snapshots are
required because a rejected closure trial must not modify the constitutive
state.

## Scope and role

This route accepts any compatible three-dimensional MFront behaviour without
adding a GPS-specific equation to that behaviour. It is therefore the
independent oracle for a new constitutive model and for the generic structural
closure. Its cost is the repeated MGIS integration and host-side local Newton.

For SRIX + EBSD, the recommended production choices and the generic route are
documented in {doc}`../../how-to/crystal-plasticity/choose_mfront_backend`. Do not infer the
performance of the specialised or generic GPS routes from this page; they use
different one-step constitutive implementations and share only the physical
closure.

## Configuration

```yaml
solver:
  constitutive_backend: mfront-3d-condensed-plane-stress
  mfront_behaviour_id: fcc_forest_rubin_srix
  mfront_library: build/mfront/src/libBehaviour.so
  mfront_threads: 4
  constitutive_options:
    local_transverse_predictor: tangent
    local_condition_check_mode: on_failure
    crystal_orientation:
      mode: ebsd
```

Important controls are the absolute and relative closure tolerances, maximum
local iterations, transverse predictor, conditioning-check policy and MGIS
thread count. The conditioning policy may be `always`, `on_failure` or a
diagnostic sample. These options control the nonlinear closure and do not alter
the constitutive parameters.

## Qualification role

Use this backend as the independent constitutive reference when introducing a
new three-dimensional behaviour or checking a structural closure. A comparison
is meaningful only when the two routes use the same committed state,
orientation, parameter set, load path, and tolerances. Numerical values belong
to the case-specific qualification artefact rather than to this stable
algorithmic contract.
