# Plane-stress tensor reconstruction: DIC 10×10 reference

This immutable campaign validates the complete 3D tensor reconstruction on the
versioned DIC-driven 10×10 case. Both constitutive backends use the unchanged
2D plane-stress FEM solve. Only converged results are completed with the
out-of-plane components.

## Reproduction

From the repository root:

```bash
source /home/jeff/.local/share/tfel/env/env.sh
export PYTHONPATH="/home/jeff/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
.venv/bin/python scripts/compare_fem_backends.py \
  --input data/processed/case-study-10x10 \
  --output validation/reference_data/plane_stress_tensor_reconstruction_dic_10x10_v1 \
  --library build/mfront/src/libBehaviour.so \
  --threads 2 \
  --historical-reference validation/reference_data/mfront_newton_dic_10x10_v1
```

The script refuses to overwrite a non-empty campaign. Use another output
directory for a fresh run.

## Result

All field-comparison, invariant, and historical-regression checks pass.

| Quantity | Python | MFront |
|---|---:|---:|
| reconstruction source | analytical | native `AxialStrain` |
| maximum `abs(S33)` (MPa) | 0 | 1.046e-14 |
| maximum `abs(trace(PE))` | 0 | 1.406e-19 |
| maximum `abs(E - EE - PE)` | 8.132e-20 | 1.355e-19 |

The largest historical-field difference from the pre-feature reference is
zero for Python and 4.263e-14 MPa for MFront stress. This is floating-point
round-off; the 2D mechanical outputs are unchanged.

The complete numerical report, configuration, input hashes, library hash,
thresholds, and artifact hashes are stored in `report.json`. Each backend NPZ
contains the historical fields, the four reconstructed tensors, the native
plane-stress residual, and the explicitly separated `EVM_HISTORICAL` and
`EVM_RECONSTRUCTED_3D` measures.
