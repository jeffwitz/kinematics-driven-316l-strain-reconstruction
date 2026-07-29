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

The production metrology must reach the native image grid:
`finest_scale=0`. Stopping at scale 1 discards the last full-resolution
refinement and is not acceptable for measuring the response to 4--32 px
bands. The command records both the requested and queried OpenCV settings in
the manifest; verify this value before interpreting the results.

## Run both diagnostics

```bash
fem-inhouse characterise-dic-measurement-chain \
  --images /path/to/DIC_images \
  --prepared-case data/processed/case_study \
  --output validation/reference_data/my_dic_chain \
  --figure-output validation/figures/my_dic_chain \
  --profile legacy_script_2021 \
  --warp-mode iterative_forward_inverse
```

Use `--profile declared_medium_v4` for the fully explicit V4 sensitivity. Do
not select a profile from its final FEM/DIC score. Use
`--warp-mode legacy_approximate_inverse` only to reproduce the pre-correction
synthetic artefacts.

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

Before using a report, check:

```bash
python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path(
    "validation/reference_data/my_dic_chain/manifest.json"
).read_text())
assert manifest["disflow_queried"]["finest_scale"] == 0
PY
```

## Replay an archived FEM campaign symmetrically

The image-level V3 operator requires an archived displacement field and its
unchanged manifest/status files:

```bash
fem-inhouse replay-dic-observation \
  --campaign results/constitutive-local-p0043-pad150 \
  --prepared-case data/processed/case_study \
  --reference-image /path/to/DIC_images/000294.tif \
  --partition-id 43 \
  --profile legacy_script_2021 \
  --output validation/reference_data/my_symmetric_replay
```

The command verifies the source `U.npy` hash, obtains solve/core bounds from
the campaign manifest, uses the corrected iterative inverse and writes DIC,
raw-FEM and DISFlow-observed EVM separately. It does not rerun mechanics,
modify the source campaign or post-filter EVM.

Repeat with `--profile declared_medium_v4` into another empty directory for
the required profile sensitivity.

## Diagnose local photometric quality

Once the primary legacy-profile V3 replays exist, compare their local EVM
errors with the direct image residual:

```bash
fem-inhouse diagnose-dic-photometric-quality \
  --reference-image /path/to/DIC_images/000294.tif \
  --final-image /path/to/DIC_images/000334.tif \
  --prepared-case data/processed/case_study \
  --replay local 0 validation/reference_data/V3/local_legacy_script_2021 \
  --replay alpha1 1 validation/reference_data/V3/a100_legacy_script_2021 \
  --replay alpha2 2 validation/reference_data/V3/a200_legacy_script_2021 \
  --replay alpha4 4 validation/reference_data/V3/a400_legacy_script_2021 \
  --output validation/reference_data/my_photometric_quality \
  --figure-output validation/figures/my_photometric_quality
```

The command verifies the immutable replay hashes. It writes the element-scale
photometric residual, geometric validity mask, fixed-decile CSV, figures and
report. It does not rerun mechanics. The unmasked metrics remain primary;
the q90 exclusion is only a declared sensitivity.

## Common failures

`DISFlow support requires the 'measurement' optional dependency`
: Install `.[measurement]` in the active environment.

`unexpected DIC image shape`
: The public workflow currently implements the pre-registered
  \(5400\times4400\) source and fixed crop. Do not silently resize another
  experiment.

`archived FEM displacement hash does not match status`
: Restore the immutable campaign artefact or select the correct campaign.
  Never replay an untracked replacement field.

`refusing to overwrite non-empty directory`
: Select a new campaign identifier, or use `--overwrite` only when
  intentionally regenerating the same declared campaign.

For interpretation, see {doc}`../explanation/current_evidence`. Exact field
and measurement conventions are in {doc}`../reference/observation_operator`.
