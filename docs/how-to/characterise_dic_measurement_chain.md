# Characterise the DIC measurement chain

**Category: How-to.** Run the pre-registered DISFlow null and synthetic
transfer diagnostics on a raw image sequence.

## Prerequisites

Install the measurement dependency:

```bash
python -m pip install -e '.[measurement]'
```

You need:

- the raw TIFF sequence containing `000294.tif`, `000334.tif` and
  `000335.tif`;
- the prepared canonical case containing `displacement_x_mm.npy` and
  `displacement_y_mm.npy`;
- an empty report directory and an empty figure directory.

Read the experiment-specific frame mapping in
{doc}`../reference/experimental_data_inventory` before treating the final pair
as a repeated state.

## Run both diagnostics

```bash
fem-inhouse characterise-dic-measurement-chain \
  --images /path/to/DIC_images \
  --prepared-case data/processed/case_study \
  --output validation/reference_data/my_dic_chain \
  --figure-output validation/figures/my_dic_chain
```

Use `--null-only` to skip the synthetic sinusoidal and band correlations.
Existing non-empty output directories are rejected unless `--overwrite` is
given.

## Outputs

The report directory contains:

- `manifest.json`, including source hashes, OpenCV version and queried DIS
  settings;
- `null_test_report.json` and `null_autocorrelation.csv`;
- `transfer_report.json`, `sinusoidal_transfer.csv` and
  `band_width_fidelity.csv`.

The figure directory contains the null-test map, transfer curve and recovered
band-width plot.

The main EVM is never Helmholtz-filtered in this workflow. The synthetic tests
characterise the algorithmic observation operator; they do not reproduce
out-of-plane motion or load-dependent illumination changes.

## Common failures

`DISFlow support requires the 'measurement' optional dependency`
: Install `.[measurement]` in the active environment.

`unexpected DIC image shape`
: The public workflow currently implements the pre-registered
  \(5400\times4400\) source and fixed crop. Do not silently resize another
  experiment.

`refusing to overwrite non-empty directory`
: Select a new campaign identifier, or use `--overwrite` only when
  intentionally regenerating the same declared campaign.

For interpretation, see {doc}`../explanation/current_evidence`. Exact field
and measurement conventions are in {doc}`../reference/observation_operator`.
