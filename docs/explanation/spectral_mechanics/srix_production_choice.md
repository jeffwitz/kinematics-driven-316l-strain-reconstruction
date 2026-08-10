# Why SRIX is the production law for the registered P43 reconstruction

The registered P43 reconstruction uses the Forest–Rubin SRIX law because the
available DIC sequence defines an ordered displacement path but does not
provide a qualified physical time scale. SRIX is rate-independent: its
response depends on the strain path and its increments, not on an arbitrary
assignment of elapsed seconds between images.

Méric–Cailletaud is a rate-dependent comparison law. Its response requires a
physical or pseudo-time history and parameters identified at that rate. It is
useful for sensitivity studies, but it is not interchangeable with SRIX under
the current DIC data contract.

## Current production configuration

For the two-dimensional EBSD workflow, use:

```yaml
solver:
  constitutive_backend: mfront-structural-plane-stress
  mfront_behaviour_id: fcc_forest_rubin_srix
  mfront_threads: 4
  constitutive_options:
    gps_composite_fd_tangent: true
    gps_composite_fd_step: 1.0e-6
    crystal_orientation:
      mode: ebsd
```

The independent reference is `mfront-3d-condensed-plane-stress` with the same
behaviour, orientation field, parameter set, and load path.

## Interpretation limits

Rate independence does not remove dependence on load-path discretisation. The
number and ordering of increments can affect activation and reversal of slip
systems. Nor does the production-law choice identify the SRIX parameters from
DIC; the parameter provenance and any transposition from a rate-dependent
reference remain part of the case record.

The principal SRIX observables are signed system slips, accumulated absolute
slip, stress, and the relaxed transverse strains. A scalar J2 equivalent
plastic strain is not a native SRIX state variable.

See {doc}`../../how-to/use_srix_crystal_law` for the operational setup,
{doc}`../../reference/numerics/mfront_structural_plane_stress` for the
constitutive formulation, and {doc}`../forest_rubin_srix` for the law itself.
