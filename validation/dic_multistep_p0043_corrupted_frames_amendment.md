# P43 multi-step DIC corrupted-frame amendment

**Status:** pre-registered before producing a repaired displacement history or
re-running mechanics.

## Trigger

The first measured-history mechanical run failed during the imposed boundary
path.  A post-failure audit of the already archived direct-reference history
found transient, spatially localised strain increments at measured states 31
and 32:

- state 31: maximum incremental EVM `5.4593e-2`;
- state 32: maximum incremental EVM `4.0608e-2`;
- state 33 returns to the normal spatial pattern, creating a further large
  reverse increment.

The source-of-provenance script
`references/legacy_dic/yield_stress_hardening.py` independently declares
frames 31 and 32 corrupted through `CORRUPTED_FRAMES`.  It repairs only EVM,
which is sufficient for map identification but cannot provide a mechanically
consistent displacement boundary history.

## Pre-registered repair

The immutable archived raw and endpoint-anchored histories remain unchanged.
A new derived history will replace displacement states 31 and 32 by linear
interpolation between the unaffected displacement states 30 and 33:

```text
u_31 = (2 u_30 + u_33) / 3
u_32 = (u_30 + 2 u_33) / 3
```

Indices here include the zero-displacement reference as state 0.  Thus states
31 and 32 correspond to images `000325.tif` and `000326.tif`.

This rule is fixed from temporal adjacency and the independent legacy
corruption declaration.  It is not selected from mechanical convergence or
FEM-DIC agreement.  No other state may be changed.  The prepared final
endpoint must remain bitwise identical after runtime restoration of storage
round-off.

## Required diagnostics

Before mechanics:

- report EVM maxima and robust quantiles for every state and increment;
- show states and increments 29--34 with common colour limits;
- show the repaired-minus-original displacement magnitude;
- verify finite values, unchanged states outside 31--32, and unchanged final
  endpoint.

The repaired history is a sensitivity to a documented measurement failure,
not a new experimental measurement.  Results must retain both the raw and
repaired provenance.

## Decision rule

The repaired measured-history run may proceed only if the transient EVM peaks
are removed without creating a new outlier.  If mechanics still fails, no
additional temporal repair, smoothing, or solver relaxation is authorised by
this amendment.
