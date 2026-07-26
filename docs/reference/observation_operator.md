# DIC observation operator

**Category: Reference.**

The comparison operator $\mathcal M_{\mathrm{DIC}}$ maps a FEM result to the
support and convention of the experimental observation.

## Required stages

1. select the manifest-defined retained core;
2. reconstruct the configured displacement support;
3. apply the same displacement-to-strain operator to DIC and FEM;
4. compute the same equivalent total-strain measure;
5. apply spatial averaging or resampling only when explicitly configured;
6. apply the valid-data and boundary masks;
7. compute metrics on identical finite samples.

The current historical EVM chain is implemented through the shared
`reconstruct_historical_evm` path. Plotting and validation code must reuse it
rather than reimplementing the formula.

## Prohibited substitutions

- Do not compare PEEQ amplitude directly with DIC EVM.
- Do not filter the primary coupled FEM EVM after convergence.
- Do not infer core bounds from array shape.
- Do not transpose or flip fields independently for plotting.
- Do not tune a measurement filter separately for every candidate.

## Recorded metadata

The operator configuration records grid spacing, interpolation, averaging,
mask, strain convention, missing-value policy and source hashes. Cache keys
include this configuration.
