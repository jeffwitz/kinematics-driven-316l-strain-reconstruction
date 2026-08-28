# Reproduce the TET2 discretisation qualification

**Mode:** how-to  
**Domain:** spectral

This procedure isolates the two-state TET2 discretisation from the later
EBI-TET state-sharing test.

```bash
MFRONT_BEHAVIOUR_LIBRARY=build/mfront/src/libBehaviour.so \
PYTHONPATH=src python scripts/qualify_ebi_state_sharing.py \
  --mesh 12 --increments 8 --tolerance 1e-8 \
  --output validation/_generated/ebi_tet/tet2_reference_m12.json
```

Read the CPS4/TET2 rows in the JSON and record the grid, weights, number of
material histories, residual and accumulated-slip difference. TET2 is a
supported two-history discretisation; it is not the EBI-TET negative result.
Archive the JSON with the source and MFront library SHAs. The operator
definition is in {doc}`../../reference/numerics/tet2_operators` and the
Newton contract in {doc}`../../reference/numerics/newton_gmres_contract`.
