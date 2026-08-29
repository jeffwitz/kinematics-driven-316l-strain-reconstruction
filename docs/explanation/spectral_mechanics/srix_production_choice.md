# Why SRIX is the production law for the registered P43 reconstruction

The registered P43 reconstruction uses the Forest–Rubin SRIX law because the
available DIC sequence defines an ordered displacement path but does not
provide a qualified physical time scale. SRIX is path-dependent but
rate-independent: its physical response depends on the ordered strain path,
not on an arbitrary assignment of elapsed seconds between images. A different
interpolation between measured states is a different physical path. Refining
the numerical subdivision of the same prescribed path should converge to the
same solution; residual dependence on that subdivision is an integration or
solver issue, not a constitutive rate effect.

Méric–Cailletaud is a rate-dependent comparison law. Its response requires a
time/rate history whose scale has physical meaning relative to the calibrated
``K,n`` parameters. A numerical time parameterisation may be used by the
solver, but changing its scale changes the constitutive problem. Méric is
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

Rate independence does not remove dependence on the physical load path: the
ordering and interpolation of measured states can affect activation and
reversal of slip systems. That is distinct from refining the numerical
subdivision of one fixed path, which should converge. Nor does the
production-law choice identify the SRIX parameters from DIC; the parameter
provenance and any transposition from a rate-dependent reference remain part
of the case record.

The principal SRIX observables are signed system slips, accumulated absolute
slip, stress, and the relaxed transverse strains. A scalar J2 equivalent
plastic strain is not a native SRIX state variable.

See {doc}`../../how-to/crystal-plasticity/use_srix_crystal_law` for the operational setup,
{doc}`../../reference/numerics/mfront_structural_plane_stress` for the
constitutive formulation, and {doc}`../constitutive/forest_rubin_srix` for the law itself.
