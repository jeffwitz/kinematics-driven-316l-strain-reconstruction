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
