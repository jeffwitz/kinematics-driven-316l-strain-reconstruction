# Reproduce the registered EBI-TET falsification

**Mode:** how-to  
**Domain:** evidence

## Prerequisites

Build the registered MFront behaviour and run from the repository root. Record
the source commit and the MFront library SHA next to the generated JSON.

## Reproduce

Use the registered source commit and MFront library, then run the actual
qualification entry point:

```bash
MFRONT_BEHAVIOUR_LIBRARY=build/mfront/src/libBehaviour.so \
PYTHONPATH=src python scripts/qualify_ebi_state_sharing.py \
  --mesh 12 --increments 8 --tolerance 1e-8 \
  --output validation/_generated/ebi_tet/state_sharing_m12_reproduced.json
```

## Expected output and verification

The JSON output contains CPS4, TET2, EBI and independent verification results.
For the registered checks, the expected thresholds are a TET2 verification
residual below `1e-13` and a directional tangent error below `1e-6` (evidence
`E-EBI-001`). The state-sharing comparison is expected to report approximately
`5.39%` EBI-versus-TET2 accumulated-slip error versus `0.72%` TET2-versus-CPS4
(`E-EBI-002`). Archive the JSON with the commit SHA and MFront library SHA,
then record whether these values are reproduced.

The result is a documented falsification for that registered SRIX case; do not
generalise it to every possible EBI formulation. Record the exact configuration
and artifact identifiers in the evidence registry.
