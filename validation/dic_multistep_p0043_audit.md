# P43 measured-boundary history audit

Date: 2026-07-29

## Available experimental sequence

The raw directory contains 42 images, `000294.tif` through `000335.tif`, all
with shape \(5400\times4400\). The mapping supported by the supplied legacy
scripts and current inventory is:

- `000294.tif`: reference;
- `000295.tif`--`000334.tif`: monotonic steps 1--40;
- `000335.tif`: candidate repeated final state.

The acquisition log and a synchronised force history are not available.
Consequently the step index can be used as an ordered pseudo-time, but cannot
be converted into physical load or elapsed time.

## Available displacement histories

The maintained prepared case contains only final nodal fields:

```text
displacement_x_mm.npy
displacement_y_mm.npy
```

No intermediate experimental displacement array is archived in the
repository or under `9_numerical`.

`fem_partition_E200_allframes_dataset.h5` contains 18 nodal Abaqus
displacement states and corresponding stress/strain fields. Its metadata
explicitly identifies a 5x5 hard-cut Abaqus reconstruction. These fields are
not experimental DIC history and must not be substituted for the missing
intermediate DIC displacements.

## Reproducible reconstruction route

The supplied historical source computes every displacement field directly
from the same reference image, rather than accumulating consecutive flows.
The maintained equivalent route is therefore:

```text
000294 -> each of 000295..000334
legacy_script_2021 DISFlow profile
canonical crop rows 400:4000, columns 1211:4311
canonical image-flow to (ux, uy) conversion
extract P43 solve-support nodes from manifest bounds
```

The absent historical `mask.png` does not block this route. The maintained
campaign declares an all-valid mask and does not claim bitwise historical
reproduction.

## Solver capability gap

The current nonlinear solver accepts one final displacement field and applies
it proportionally through pseudo-time. It does not accept a sequence of
measured boundary targets. Supporting measured history requires:

1. a nodal target at every history knot;
2. interpolation only during cutback between two adjacent measured knots;
3. an elastic predictor recomputed from each actual boundary increment;
4. exact restoration of displacement and constitutive state after failure;
5. provenance distinguishing measured-knot history from proportional loading.

No constitutive change is required.

## Scope decision

The first implementation will:

- reconstruct and archive the 40 direct-reference DIC displacement fields on
  the P43 solve support;
- add a generic piecewise-linear boundary-history path to the solver;
- compare the local P43 result obtained with measured boundaries against the
  existing proportional local baseline;
- report EVM at 25, 50, 75 and 100 % of the ordered image sequence.

It will not identify material parameters, rerun a micromorphic sweep, infer a
load-cell curve or call this a held-out prediction.
