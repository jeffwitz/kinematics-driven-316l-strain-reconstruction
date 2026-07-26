# Run a coupled nonlocal reconstruction

**Category: How-to.**

## Prerequisites

- a completed local campaign on the same region;
- padding appropriate for the selected $\ell$;
- $H_{\mathrm{ref}}$ and the intended $\alpha$ recorded before execution;
- micromorphic MFront behaviours compiled.

## Derive the reference modulus

```bash
fem-inhouse estimate-nonlocal-reference \
  --help
```

Run the command with the local campaign and core arguments required by the
installed revision. Preserve its JSON output; identification reads
$H_{\mathrm{ref}}$ from metadata rather than a hard-coded value.

## Run one coupled candidate

```bash
fem-inhouse --verbose partition \
  --input data/processed/case-study \
  --output results/coupled-candidate \
  --parts-x PARTS_X --parts-y PARTS_Y \
  --padding PADDING_PIXELS \
  --partition-id PARTITION_ID \
  --increments 20 \
  --max-newton-iterations 25 \
  --constitutive-backend mfront-native-plane-stress \
  --nonlocal-plasticity \
  --nonlocal-length-um LENGTH_UM \
  --nonlocal-coupling-modulus-mpa HCHI_MPA \
  --nonlocal-relaxation 0.5 \
  --nonlocal-tolerance 1e-6 \
  --nonlocal-max-iterations 15 \
  --mfront-threads 8
```

The primary comparison uses raw converged FEM displacement. Do not apply a
Helmholtz filter to final EVM.

## Check completion

Inspect `PEEQ_NONLOCAL.npy`, `PEEQ_MISMATCH.npy`,
`NONLOCAL_HARDENING_MPA.npy`, `YIELD_SURFACE_RADIUS_MPA.npy` and
`NONLOCAL_RESIDUAL.npy`, together with Newton, fixed-point and cutback
diagnostics.

Typical failures are insufficient padding, an incompatible local reference,
fixed-point exhaustion, mechanical Newton exhaustion, non-finite state or a
non-positive yield radius. Treat a convergence failure as numerical censoring
unless a physical admissibility check fails.

See {doc}`../explanation/micromorphic_model` for the physics and
{doc}`../reference/numerics/nonlocal_fixed_point` for the algorithm.
