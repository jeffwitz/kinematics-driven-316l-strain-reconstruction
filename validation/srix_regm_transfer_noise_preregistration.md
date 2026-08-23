# SRIX-REGM transfer/noise twin preregistration

Date frozen: 2026-08-23  
Gate: 4, before any P43 identification

## Inputs

- exact clean M8 twin: `reference_data/srix_regm_twin_v1`;
- observation transfer: affine-preserving application of
  `dic_measurement_chain_v4/sinusoidal_transfer.csv`;
- measured repeat-frame noise:
  `dic_uncertainty_propagation_p0043_v1/centred_repeat_flow_pixels.npy`;
- pixel size: `0.00184 mm`;
- deterministic sampling seeds: transfer/noise `20260823`, whitener `42`;
- the same eight scored macro endpoints and every accepted causal warm-up step
  as the exact twin.

## Frozen levels

1. `T0 exact`: existing unobserved displacement, identity observation and
   identity whitener.
2. `T1 transfer`: every synthetic displacement is passed once through the
   affine-preserving measured transfer; REGM corrections pass through the same
   transfer; no noise and identity whitener.
3. `T2 transfer_noise`: T1 plus independent deterministic `9 x 9` patches from
   the measured repeat-frame displacement noise at every non-initial state;
   the initial state stays exactly zero. Corrections are whitened with the
   stationary displacement whitener fitted from 256 independent patches.

No displacement is observed twice. The measured-noise realisation is added in
physical millimetres after the image-flow-to-canonical axis conversion.

## Read-outs

- REGM residual at truth and at the frozen initial theta4;
- central-FD Jacobian at log step `3e-3`;
- singular values, normalized values, rank and right singular vectors;
- deterministic least-squares recovery from the same frozen initial point;
- timing split and parameter error projected on every retained subspace.

This gate is descriptive: loss of rank under measured transfer/noise is a valid
negative result and must not be repaired by changing thresholds. P43 remains
blocked until the independent REGM/FEMU ranking gate passes.

