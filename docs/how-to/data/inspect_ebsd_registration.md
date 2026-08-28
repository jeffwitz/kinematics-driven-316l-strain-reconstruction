# Inspect EBSD registration

**Mode:** how-to  
**Domain:** crystal-plasticity

## Run the data-only audit

From the repository root, run:

```bash
PYTHONPATH=src python scripts/run_p0043_fullfield_schmid_registration.py
```

The expected artifacts are
`validation/reference_data/p0043_fullfield_schmid_registration_v1/final_report.json`,
`provenance_report.json` and `schmid_metrics.csv`. Check the report fields
`ebsd_global_geometry_known`, `ebsd_axis_metadata_found` and
`registration_proven`; the registered campaign currently keeps the last one
false unless independent registration metadata are supplied.

Before a forward run, also check dimensions, pixel sizes, origin, row/column
order, crop history and the declared `Q_global_to_material` convention. Keep
spatial mapping F separate from sample-frame axis transformations.

See {doc}`../../reference/scientific/ebsd_orientation_contract`.
