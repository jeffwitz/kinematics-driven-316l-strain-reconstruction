# Validation strategy and evidence

## Four questions that must remain separate

### 1. Is the implementation numerically correct?

Patch tests and analytical reduced cases check:

- CPS4 interpolation and strain-displacement matrices;
- plane-stress elasticity;
- engineering-shear conventions;
- sparse assembly;
- reactions and equilibrium;
- partition ownership and stitching.

These are software and discretization checks.

### 2. Is the constitutive coupling correct?

Material-point paths and full Newton tests compare Python and MFront:

- stress histories;
- PEEQ;
- consistent tangents;
- trial/commit/revert behaviour;
- serial/parallel identity;
- final FE fields.

These checks validate the adapter and nonlinear state management. They do not
validate the material model against an experiment.

### 3. Does the reconstruction agree with DIC?

DIC and FE displacement are differentiated with the same grid operator.
Comparison reports include:

- RMSE and MAE;
- signed bias;
- relative L2 error;
- spatial correlation;
- localization overlap where applicable;
- interface-gradient diagnostics.

This asks whether equilibrium reconstruction preserves measured spatial
organization.

### 4. Does it reproduce the historical Abaqus calculation?

That requires the original `.inp`, section thickness, exact material table,
ODB extraction, locations, and component conventions. Those artifacts are not
available. Abaqus parity is therefore explicitly deferred.

## Why no single metric is sufficient

RMSE measures amplitude error but ignores topology. A field may have a low RMSE
because most pixels are quiet while missing narrow localization bands.

Spatial correlation detects co-variation but is sensitive to shifts and weak
when the field variance is small. Localization overlap detects hot-zone
placement but depends on a declared threshold. Interface metrics identify grid
artefacts but do not measure global amplitude.

The report keeps these metrics separate instead of collapsing them into one
score.

## Equivalent strain comparison

Both displacement fields use:

$$
\epsilon_{xx}=\partial_xu_x,\quad
\epsilon_{yy}=\partial_yu_y,\quad
\gamma_{xy}=\partial_yu_x+\partial_xu_y.
$$

Engineering shear is converted to tensorial shear before the three-dimensional
deviatoric invariant is evaluated under plane stress.

This common reconstruction avoids mixing:

- DIC gradients from NumPy;
- element strain from one Gauss-point/extrapolation convention;
- visualization smoothing from another solver.

Direct FE element strain remains a valid solver output, but it answers a
different comparison question.

## Four stress–strain curves

The project distinguishes:

1. measured macroscopic stress–strain;
2. DIC strain-based reconstructed equivalent stress;
3. FE strain-based reconstructed equivalent stress;
4. direct FE equivalent stress.

Curves 2 and 3 apply the scalar Ludwik law to spatially averaged equivalent
strain. They are consistency reconstructions.

Curve 4 first averages $S_{11}$, $S_{22}$, and $S_{12}$, then evaluates
plane-stress von Mises stress. It may differ after yielding because stress
redistribution and nonlinear averaging do not commute. Replacing it with curve
3 to obtain a more favourable plot would hide a real modelling distinction.

## Current strongest result

The saved `510 × 460` corner partition:

- converges 20/20 increments without cutback;
- satisfies DIC boundary displacement to machine precision;
- balances global reactions to approximately $4\times10^{-14}$ relative;
- preserves all six fields and fingerprints;
- provides DIC/FE derived maps and a comparison with the historical table.

Its full-region equivalent-strain RMSE is `0.254` percentage points and spatial
correlation is `0.016`. The error amplitude is informative, but the weak
correlation and single-partition scope prevent a reproduction claim.

## Evidence needed for the next scientific claim

Before claiming complete reproduction:

1. solve and validate all 100 partitions;
2. stitch every field with the declared ownership rule;
3. apply the exact ROI mask and metric conventions;
4. run the padding sensitivity study;
5. compare the four macroscopic curves;
6. obtain the original Abaqus input and ODB extraction for external parity.

The repository’s provenance and resumption mechanisms are designed so those
calculations can extend the evidence without invalidating saved work.

