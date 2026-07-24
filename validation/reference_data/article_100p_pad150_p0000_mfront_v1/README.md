# Article partition 0 — analytical MFront v1

This campaign repeats the previously preserved `510×460`-element corner
partition with the same DIC inputs, 100-partition layout, 150-pixel overlap,
20 increments, PyPardiso settings, and convergence tolerance.

The only intended constitutive change is the default MFront
`PixelLudwikJ2Plasticity` behaviour:

- analytical Ludwik hardening after the regularised first `1e-6` PEEQ segment;
- no upper PEEQ cap;
- eight-point MGIS thread pool;
- trial state committed only after global Newton convergence.

`run-request.json` fixes the code, input, MFront source, compiled library,
layout, solver, and thread configuration before execution. The completed
campaign must retain the six fields, workflow manifest/status, full log,
`/usr/bin/time -v` resource measurement, and a post-run validation report.

## Result

The solve completed successfully on 2026-07-24:

- 20/20 increments converged without cutback;
- 112 Newton iterations, at most 6 per increment;
- final relative residual: `2.207e-8`;
- solver time: `648.402 s`;
- complete process wall time: `650.08 s` (`10 min 50.08 s`);
- peak process RSS: `4,163,308 KiB` (3.97 GiB);
- no swap;
- prescribed DIC boundary error: `4.163e-17 mm` maximum;
- reaction balance ratio: `3.961e-14`.

The maximum PEEQ is `0.06496`; no integration point reaches the historical
`0.2` cap on this particular partition. The cap is nevertheless absent from
the nominal model, so a future loading path is not silently flattened.

## Comparison with the preserved tabulated Python run

The historical campaign under `../article_100p_pad150_p0000` used the same
geometry, DIC data, increments, tolerance and PyPardiso solve, but the Python
1000-point table capped at PEEQ `0.2`.

| Measure | Python table | MFront analytical | Change |
|---|---:|---:|---:|
| Process wall time | `1089.80 s` | `650.08 s` | `-40.35%` |
| Solver time | `1088.126 s` | `648.402 s` | `1.678×` faster |
| Constitutive time | `575.906 s` | `83.409 s` | `6.905×` faster |
| Newton iterations | 113 | 112 | -1 |
| Peak RSS | `3,768,132 KiB` | `4,163,308 KiB` | `+10.49%` |

The memory result is intentionally reported without attributing the whole
process peak to the hardening representation. The MFront path never constructs
the Python 1000-point table, but MGIS state and tangent storage plus the sparse
finite-element working set lead to a higher measured peak on this run.

The final displacement changes by `1.575e-5` in relative L2 norm. Relative-L2
differences are `0.721%` for total strain, `0.910%` for plastic strain,
`0.868%` for PEEQ, and `0.759%` for stress. These are expected model
differences between the analytical law and its historical interpolation, not a
bitwise-parity claim.

## Preserved artifacts

- `run-request.json`: immutable pre-run intent and hashes;
- `manifest.json` and `preflight.log`: partition contract checked before solve;
- `run.log` and `resource-usage.txt`: solver log and independent process timing;
- `partitions/0000/status.json`: convergence and output hashes;
- `partitions/0000/{U,S,E,PE,PEEQ,RF}.npy`: all final raw fields;
- `derived/*.npy` and `preview.png`: DIC/FEM comparison maps;
- `validation-report.json`: integrity, mechanics, performance and comparison
  metrics;
- `validation.log`: complete machine-readable output from the validation
  script.

Rebuild the derived products and report without rerunning the solve:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_saved_article_partition.py \
  --campaign validation/reference_data/article_100p_pad150_p0000_mfront_v1 \
  --input data/processed/case_study \
  --comparison-campaign validation/reference_data/article_100p_pad150_p0000
```
