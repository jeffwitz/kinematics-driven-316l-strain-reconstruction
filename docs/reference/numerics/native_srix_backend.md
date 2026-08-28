# Native SRIX backend contract

**Mode:** reference  
**Domain:** crystal-plasticity

The native backend implements the qualified SRIX equations using the same
central parameter objects as MFront. It exposes transactional
`evaluate`, `complete_trial`, `commit` and `revert` operations through the
`PlaneStressMaterialBatch` contract.

## Options

The backend and closure are selected independently:

```yaml
constitutive_backend: numpy-srix
constitutive_options:
  plane_stress_solver: nested   # or coupled
  coupled_block_solver: auto   # numpy, numba-fused, numba-fused-state, auto
  fused_state_threshold: 12000
  response_level: residual      # residual, tangent, complete
```

`nested` is the conservative native reference and is also the strategy
available to generic MFront. `coupled` is available only for native SRIX and
solves the twelve slip equations together with the three transverse traction
equations. `auto` chooses the block kernel from the active batch size; the
12,000-point threshold is conservative and machine-dependent.

The six Kelvin components are `[xx, yy, zz, xy, xz, yz]`; the in-plane
components are `[xx, yy, xy]` and the relaxed components are `[zz, xz, yz]`.
The required residual is
`sigma_zz = sigma_xz = sigma_yz = 0`. Orientations use
`Q_global_to_material` and EBSD assignment follows mapping convention F.

## Transaction and responses

An evaluation creates a trial without modifying committed state. `commit()`
promotes it; `revert()` discards it. `response_level=residual` avoids enriched
field construction, `tangent` returns the condensed in-plane tangent, and
`complete` includes the diagnostic state required for archival.

See {doc}`plane_stress`, {doc}`srix_semismooth_jacobian` and
{doc}`../../reference/api` for the surrounding interfaces.
