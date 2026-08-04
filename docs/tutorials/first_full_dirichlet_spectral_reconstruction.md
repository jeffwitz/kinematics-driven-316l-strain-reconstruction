# First full-Dirichlet spectral reconstruction

This tutorial follows the smallest reproducible path. It is intentionally
split into an elastic operator check and the registered SRIX campaign.

1. construct a rectangular nodal displacement field;
2. split it into harmonic applied and zero-boundary fluctuation parts;
3. verify the DST-I round trip;
4. evaluate the two TET2 strain samples;
5. apply the B0-preconditioned Newton-GMRES solve;
6. inspect the verified residual and reactions.

## 1. Elastic operator check

Run the executable tests that do not require an MFront library:

```bash
python -m pytest -q \
  tests/unit/spectral2d/test_grid_transforms_boundary.py \
  tests/unit/spectral2d/test_kinematics.py \
  tests/unit/spectral2d/test_newton_two_state.py
```

The expected result is a zero exit status. The tests cover DST-I round trips,
zero boundary reconstruction, affine strains, adjoint kinematics and the
small-grid Newton transaction. The principal arrays are:

```text
nodal displacement  (nx + 1, ny + 1, 2)
sample strain       (nx, ny, 2, 3)
sample stress       (nx, ny, 2, 3)
```

## 2. Registered SRIX run

With the compiled behaviour library available, run the registered causal
comparison. This command executes CPS4, two-state TET2, one-state EBI and the
independent final verification:

```bash
MFRONT_BEHAVIOUR_LIBRARY=build/mfront/src/libBehaviour.so \
python scripts/qualify_ebi_state_sharing.py \
  --mesh 12 --increments 8 --tolerance 1e-8 \
  --output validation/_generated/ebi_tet/state_sharing_m12_reproduced.json
```

The run writes the JSON summary under
`validation/_generated/ebi_tet/`. Inspect the exact result with:

```bash
jq '{errors, verification_residual, iterations, timings}' \
  validation/_generated/ebi_tet/state_sharing_m12_reproduced.json
```

The expected EBI verification residual at 12x12 is approximately
$1.12\times10^{-12}$ for the archived case. This qualification entry point
does not write field NPZ files or Newton/GMRES JSONL traces; use the archived
JSON reports under `validation/_generated/ebi_tet/` for the reproducible
comparison. The run is diagnostic, not a permission to generalize the
registered negative EBI verdict.
