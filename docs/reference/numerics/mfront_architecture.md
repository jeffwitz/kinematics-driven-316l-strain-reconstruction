# MFront backend architecture

The MFront backend is split into layers with unchanged public compatibility
through fem_inhouse.core.mfront.

~~~text
             ┌─────────────┐
             │ MGIS runtime│
             └──────┬──────┘
                    │
          ┌─────────┴─────────┐
          │                   │
      raw 3D bridge       GPS adapter
          │                   │
    condensation          substepping
                              │
                       composite tangent
~~~

The responsibilities are deliberately separate:

- mfront_runtime.py loads MGIS behaviours, applies parameters, inspects
  variable layouts, and provides Kelvin conversion helpers.
- mfront_state.py owns generic immutable snapshots and public timing records.
- mfront_native.py contains the native 2D bridge.
- mfront_3d.py contains the raw 3D bridge and explicit global/crystal
  rotations.
- mfront_condensation.py contains external plane-stress closure, Schur
  condensation, predictors, and block batching.
- mfront_gps/adapter.py adapts the GPS behaviour to the PlaneStressMaterialBatch
  contract.
- mfront_gps/substepping.py owns the GPS substep policy and failure cache.
- mfront_gps/composite_tangent.py differentiates the composed substepped
  application when required.
- mfront_gps/diagnostics.py contains the non-production shadow tangent.

The constitutive behaviour, the substepping strategy, and the derivative of
the composed integration algorithm are three distinct layers. In particular,
when the driver composes several constitutive maps,

$$
D\Phi_{\mathrm{last}}
\neq
D(\Phi_n\circ\cdots\circ\Phi_1),
$$

so the composite tangent belongs to the GPS integration layer rather than to
the MFront constitutive law.

The raw 3D bridge stores MGIS gradients in the material convention while
retaining committed_global_strain separately. The GPS bridge passes global
gradients to a behaviour that owns its crystal rotation. These conventions
are intentionally not hidden behind a boolean rotation flag.

mfront.py is now a compatibility façade. Existing imports remain valid,
including the diagnostic private helpers used by validation scripts.

## Qualification checkpoint

The extraction was replayed on the registered P43 M100 EBSD case with the
qualified runtime settings (four MFront threads, one FFTW thread, one Krylov
BLAS thread). GPS with composite FD retained 58 Newton iterations, the
increment sequence [6, 6, 7, 7, 7, 8, 8, 9], 192 FD points, 1152 trajectories,
and a final residual of 5.34e-9. The measured elapsed time was 44.98 s.

The corresponding artifacts are
validation/_generated/performance/mfront_refactor_m100_gps_fd.json and
validation/_generated/performance/mfront_refactor_m100_gps_fd.fields.npz.

## Known coupling, recorded rather than fixed

`GPSSubsteppingMixin` and `CompositeTangentMixin` are extracted from the
adapter but not decoupled from it. Both reach into adapter attributes --
`_manager`, `_failing_cache`, `_maximum_substeps`, `_last_substep_mask` and
others -- and both carry a `# mypy: ignore-errors` header for that reason. A
reader cannot take either file on its own and learn its contract: the contract
is the adapter's internals.

That is a large improvement on three thousand lines in one module, and it is
not the end state. The end state is an explicit interface -- a
`GPSIntegrationContext` protocol, or an integrator object handed the operations
it needs -- so that the mixins depend on a named surface instead of on whatever
the adapter happens to expose.

It is **not** being done now: the regression risk of rewiring the
sub-stepping and the composite tangent outweighs the readability gain while the
qualification numbers are fresh. It is written here so the split is understood
as a stage rather than as finished work.
