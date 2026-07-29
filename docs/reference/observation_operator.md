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

## Declared DISFlow reproduction

The optional measurement-chain implementation adds an image-level operator:

```text
reference speckle image
→ known or FEM displacement warp
→ OpenCV DISFlow
→ reconstructed displacement
→ historical EVM
```

Its current declared configuration is OpenCV 4.14, medium preset, finest
scale 1, patch size 8, patch stride 3, mean normalisation and spatial
propagation enabled, 30 gradient-descent iterations, and variational
refinement with \(\alpha=100\), \(\delta=1\), \(\gamma=0\),
\(\epsilon=0.002\), 30 iterations.

Every run queries these values back from the OpenCV object and stores them in
its manifest. This is a **reproduction implementation**, not a bitwise copy of
the historical executable whose OpenCV version and remaining settings were
not archived.

The pre-registered measurement-chain evidence is summarised in
{doc}`../explanation/current_evidence`; the runnable procedure is
{doc}`../how-to/characterise_dic_measurement_chain`.

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
