# Run the registered TET2/EBI SRIX comparison

Use the qualification entry point that actually runs CPS4, two-state TET2 and
one-state EBI Newton-GMRES:

```bash
python scripts/qualify_ebi_state_sharing.py --help
```

The complete 12x12 command is:

```bash
MFRONT_BEHAVIOUR_LIBRARY=build/mfront/src/libBehaviour.so \
python scripts/qualify_ebi_state_sharing.py \
  --mesh 12 --increments 8 --tolerance 1e-8 \
  --output validation/_generated/ebi_tet/state_sharing_m12_reproduced.json
```

The JSON summary contains the CPS4, TET2, EBI and independent verification
results. Archive it with the commit SHA and the MFront library SHA.

Record the commit, grid, tolerance, reference parameters, thread count and
the JSON/NPZ output paths. The TET2/CPS4 comparison and the EBI/TET2
state-sharing comparison use the same registered kinematics.

The older `qualify_spectral2d_against_newton.py` entry point is retained only as
a **Historical fixed-point and Anderson diagnostic**. It does not reproduce
the Newton-GMRES TET2/EBI evidence.
