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
Archive it with the commit SHA and MFront library SHA, then compare the
reported residuals and field metrics with the registered thresholds.

The result is a documented falsification for that registered SRIX case; do not
generalise it to every possible EBI formulation. Record the exact configuration
and artifact identifiers in the evidence registry.
