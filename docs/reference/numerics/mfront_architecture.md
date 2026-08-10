# MFront backend architecture

This page defines the stable implementation boundaries of the MFront/MGIS
material layer. Application code should select a registered backend through
the configuration and factory layers; the modules below are maintainer
interfaces.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `mfront_runtime` | Load MGIS behaviours, inspect variables and properties, apply parameters, and manage native material workers. |
| `mfront_state` | Represent committed snapshots and provide transaction primitives. |
| `mfront_native` | Native two-dimensional and plane-stress bridges. |
| `mfront_3d` | Three-dimensional material bridge, material/global rotations, and global committed strain. |
| `mfront_condensation` | External transverse closure, Schur complement, predictors, and block batching. |
| `mfront_gps.adapter` | Native structural plane-stress adapter for MFront behaviours. |
| `mfront_gps.substepping` | Selective constitutive substepping and trajectory integration. |
| `mfront_gps.composite_tangent` | Derivative of a composed substepped integration map. |
| `mfront_gps.diagnostics` | Optional shadow and validation diagnostics. |
| `mfront.py` | Compatibility facade preserving historical imports. |

## Frame contracts

The raw three-dimensional bridge receives the imposed gradient in the
structural/global frame, rotates it into the material frame for MGIS, and
rotates stresses and tangent operators back. Its committed global strain is a
separate state value and must not be inferred from the material-frame MGIS
gradient.

The structural plane-stress bridge also receives a global/structural gradient.
The selected MFront behaviour performs its own structural closure. The host
must not rotate the gradient a second time.

All tensor rotations use the repository Kelvin conventions. A tangent changes
frame on both its input and output sides; it is therefore transformed as a
fourth-order operator, not as a stress vector.

## Transaction contract

Material evaluation is trial-only. A successful evaluation updates the trial
state and may be followed by `commit`; it never commits implicitly. A rejected
trial is discarded with `revert` or restored from a complete snapshot.

A committed snapshot contains every quantity needed to reconstruct the physical
state, including MGIS state arrays, global committed strain, and nonlocal values
when present. Snapshots are immutable records from the caller's perspective.
Restoration must not mutate `s0` behind the transaction layer.

The required invariants are:

- evaluating a trial does not alter the committed state;
- reverting twice is idempotent;
- committing without a valid trial is rejected;
- taking a snapshot while a trial is active is rejected;
- raw 3D and structural plane-stress bridges retain their distinct frame
  contracts.

## Constitutive and host layers

The MFront behaviour supplies the one-step constitutive response and tangent.
The host may optionally replace one increment by a sequence of constitutive
substeps. If that happens, the tangent required by the global Newton solve is
the tangent of the composed map. Substepping and its composite tangent belong
to the host adapter; they are not part of the constitutive law.

This separation lets the same GPS adapter serve the specialised SRIX
behaviour and the registered generic `StructuralPlaneStress3D` behaviours.
