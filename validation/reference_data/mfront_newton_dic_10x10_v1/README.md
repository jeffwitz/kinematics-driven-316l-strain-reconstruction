# MFront/Newton DIC comparison v1

This directory preserves the first complete finite-element comparison of the
Python and MFront constitutive backends inside the Newton loop.

The input is the real central `10×10` DIC crop prepared from the versioned raw
case-study arrays. Both runs use 20 increments, the analytical Ludwik law, and
PyPardiso. They converge without cutback and every pre-declared relative-L∞
threshold passes.

An exploratory run preceded threshold fixation, so these values are
regression/acceptance thresholds rather than an independent blind validation.

Artifacts:

- `python_fields.npz`: `U/S/E/PE/PEEQ/RF` from the Python backend;
- `mfront_fields.npz`: the same six fields from the MFront backend;
- `report.json`: input and library hashes, full configuration, convergence
  diagnostics, metrics, thresholds, artifact hashes, and the decision.

Regeneration:

```bash
source /home/jeff/.local/share/tfel/env/env.sh
bash scripts/build_mfront_behaviour.sh
.venv/bin/python scripts/compare_fem_backends.py \
  --input data/processed/case-study-10x10 \
  --output validation/reference_data/mfront_newton_dic_10x10_v2 \
  --library build/mfront/src/libBehaviour.so \
  --threads 2
```

This small crop validates the coupling and state transaction. It is not a
production-size performance measurement and does not justify switching the
default backend by itself.
