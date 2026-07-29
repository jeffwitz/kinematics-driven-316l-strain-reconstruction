# P43 measured-boundary history preregistration

Date: 2026-07-29

## Question

How much does replacing the assumed proportional ramp by the measured
40-step boundary-displacement history change the local P43 reconstruction?

## Fixed data and profile

- reference image: `000294.tif`;
- measured ordered states: `000295.tif`--`000334.tif`;
- profile: `legacy_script_2021`;
- canonical recorded crop: rows `400:4000`, columns `1211:4311`;
- partition: P43 with solve and core bounds read from the existing manifest;
- absent legacy mask: declared all-valid;
- pixel size: 0.00184 mm;
- local material maps and analytical MFront plane-stress law unchanged.

Each state is correlated directly against the reference. No incremental flow
accumulation is used.

## Solver path

The measured targets define 41 knots including the zero reference state.
Between adjacent knots, interpolation is piecewise linear and is used only
when Newton cutback requires a substep. Every measured knot must be reached
exactly.

The proportional control uses the same final displacement, 40 nominal
increments and all other numerical settings unchanged.

## Outputs

Archive:

- the 40 P43 solve-support displacement fields and their hashes;
- queried OpenCV parameters and image hashes;
- final `U`, `S`, `E`, `PE`, `PEEQ`, `RF`;
- snapshots at ordered fractions 0.25, 0.50, 0.75 and 1.00;
- Newton iterations, cutbacks and residuals;
- raw and V3-observed EVM comparison where available.

## Registered comparisons

Measured-history versus proportional control:

- relative field difference for `U`, `S`, `E`, `PE`, `PEEQ`, `RF`;
- final DIC/FEM RMSE, relative L2, Pearson correlation, top-10 % IoU and
  absolute-q90 IoU;
- snapshot EVM metrics at steps 10, 20, 30 and 40;
- maximum boundary-target mismatch at every measured knot;
- converged increments, Newton iterations and cutbacks.

## Interpretation

No pass/fail threshold is imposed for this first diagnostic. A significant
difference demonstrates sensitivity to the assumed path; a small difference
supports the proportional approximation for this monotonic sequence.

This is not held-out prediction because the material maps were inferred from
the same experiment. No `H_chi`, `alpha` or `ell` value will be fitted or
changed.
