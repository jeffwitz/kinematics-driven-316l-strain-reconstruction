# Historical record: Helmholtz diagnostic

:::{admonition} Superseded
:class: warning
Historical record. Superseded for current use.
:::

This guide filters an already converged padded partition. It does not rerun or
modify the mechanical calculation.

## Prerequisites

You need:

- a prepared canonical input directory;
- a partition campaign whose `manifest.json`, partition metadata, `U.npy`, and
  `PEEQ.npy` are intact;
- a saved partition that corresponds to the requested ID;
- SciPy and Matplotlib from the locked environment.

The workflow reads the core and solved bounds from the campaign metadata. It
does not infer them from field dimensions.

## Run an exploratory sweep

```bash
.venv/bin/fem-inhouse diagnose-nonlocality \
  --input data/processed/case_study \
  --campaign validation/reference_data/article_100p_pad150_p0000_mfront_v1 \
  --partition-id 0 \
  --output results/nonlocality-diagnostic-p0000 \
  --lengths-um 0 1.84 3.68 7.36 14.72 29.44 58.88 \
  --include-peeq \
  --mode exploratory
```

Specify exactly one of `--lengths-mm`, `--lengths-um`, or
`--lengths-pixels`. Values must be finite and non-negative. The command adds
zero if needed, removes duplicates, sorts the sweep, and stores every value in
millimetres, micrometres, and physical pixels.

The pixel conversion uses the spacing recorded by the prepared case. It is
not a generic image pixel with an assumed size.

## Configure metrics and persistence

The default localization fractions are 5%, 10%, and 20%, and the default DIC
absolute-threshold quantiles are 80%, 90%, and 95%. Override them explicitly:

```bash
  --top-fractions 0.05 0.10 0.20 \
  --dic-quantiles 0.80 0.90 0.95
```

`--minimum-padding-length-ratio 4` controls the numerical boundary warning.
It does not define a material property.

Choose saved arrays with:

- `--save-fields all`: raw fields and every filtered length;
- `--save-fields best`: raw fields plus candidates selected by the report;
- `--save-fields none`: reports and figures only.

The default is `all`. Without `--overwrite`, the command refuses to replace a
non-empty output directory. Use a new directory for an immutable campaign;
reserve `--overwrite` for deliberate local iteration.

## Run a confirmatory campaign

Choose the decision criteria before the calculation:

```yaml
decision_thresholds:
  minimum_correlation_gain: 0.05
  minimum_relative_l2_reduction: 0.05
  minimum_iou_gain: 0.02
  maximum_relative_mean_drift: 1.0e-10
```

Then run:

```bash
.venv/bin/fem-inhouse diagnose-nonlocality \
  --input data/processed/case_study \
  --campaign results/reconstruction-100 \
  --partition-id 17 \
  --output results/nonlocality-confirmation-p0017 \
  --lengths-um 0 14.72 \
  --mode confirmatory \
  --decision-thresholds decision-thresholds.yaml
```

A serious confirmation uses a length selected on another partition. Reusing
the selection partition is still exploratory, regardless of the CLI mode.

## Inspect the outputs

The output contains:

```text
manifest.json
report.json
metrics.csv
fields/
figures/
```

Check, in this order:

1. `manifest.json` for input and output hashes, Git state, spacings, core and
   solved bounds, padding, versions, and exact options;
2. `metrics.csv` for one row per field and length;
3. `boundary_status` and `padding_to_length_ratio`;
4. `report.json` for separate numerical facts, diagnostic selection, and
   physical interpretation;
5. figures for peak attenuation and loss of texture.

`metric_curves.svg` summarizes field-error and localization metrics.
`diffusivity_curves.svg` shows gradient, total variation, peak, and standard
deviation changes. Every EVM comparison uses one common colour scale for DIC,
raw FEM, and all filtered FEM panels; difference scales are symmetric around
zero.

Do not interpret:

- a low PEEQ-to-DIC-EVM amplitude error—the workflow deliberately does not
  compute one;
- a `boundary_contaminated` candidate as primary evidence;
- an exploratory optimum at the sweep boundary as a bracketed optimum;
- a selected diagnostic length as an identified material internal length.

For the scientific rationale and the preserved partition-0 result, see
{doc}`../explanation/nonlocality_diagnostic`.
