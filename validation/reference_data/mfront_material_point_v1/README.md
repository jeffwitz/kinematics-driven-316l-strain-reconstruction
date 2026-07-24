# MFront material-point comparison v1

This directory preserves the first reproducible comparison between the
production Python constitutive update and the MFront/MGIS implementation.

The calculation uses 200 increments for each of three plane-stress paths:
in-plane uniaxial strain, equibiaxial strain, and simple shear. The material
parameters are the nominal case-study values recorded in `report.json`.

Artifacts:

- `material_point_histories.npz`: every imposed strain, stress, plastic strain,
  PEEQ, yield radius, and consistent tangent for both implementations;
- `comparison.png`: equivalent-stress and PEEQ histories;
- `report.json`: hashes, parameters, pre-declared thresholds, metrics, and the
  global decision.

The comparison passes the initial stress and PEEQ thresholds on all three
paths. It is a material-point result only: it does not yet validate the MFront
backend inside the finite-element Newton loop.

Regeneration:

```bash
source /home/jeff/.local/share/tfel/env/env.sh
bash scripts/build_mfront_behaviour.sh
.venv/bin/python scripts/compare_constitutive_backends.py \
  --output validation/reference_data/mfront_material_point_v2 \
  --steps 200
```
