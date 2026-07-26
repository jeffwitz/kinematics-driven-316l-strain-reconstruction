# Run joint nonlocal identification

**Category: How-to.**

## Prerequisites

- a versioned identification YAML file;
- completed local and existing coupled campaigns listed by that file;
- a validated DIC observation operator;
- no unresolved mismatch in numerical policy between reused campaigns.

## Inspect and screen

```bash
fem-inhouse identify-nonlocal inspect \
  --config configs/joint_nonlocal_identification.yaml \
  --dry-run

fem-inhouse identify-nonlocal screen-frozen \
  --config configs/joint_nonlocal_identification.yaml
```

F0 writes dense frozen-field diagnostics. It does not run mechanics.

## Validate and run F1

```bash
fem-inhouse identify-nonlocal run-low-fidelity \
  --config configs/joint_nonlocal_identification.yaml

fem-inhouse identify-nonlocal run-low-fidelity \
  --config configs/joint_nonlocal_identification.yaml \
  --identifiability-design
```

Use `--point` to resume or select one candidate and `--workers` to cap
concurrency. Do not use F1 for selection unless its ranking criteria against
existing F2 cases pass.

## Collect and select

```bash
fem-inhouse identify-nonlocal collect-results \
  --config configs/joint_nonlocal_identification.yaml
fem-inhouse identify-nonlocal profile-h \
  --config configs/joint_nonlocal_identification.yaml
fem-inhouse identify-nonlocal select-candidates \
  --config configs/joint_nonlocal_identification.yaml
fem-inhouse identify-nonlocal report \
  --config configs/joint_nonlocal_identification.yaml
```

The report keeps amplitude, localization and spatial-scale metrics separate.

## Generate, but do not launch, F2

```bash
fem-inhouse identify-nonlocal generate-high-fidelity-manifest \
  --config configs/joint_nonlocal_identification.yaml \
  --dry-run
```

Review cost, parameters, duplicate detection and the scientific role of every
candidate. High-fidelity execution requires a separate explicit action.

See {doc}`../explanation/parameter_identification` for the design logic and
{doc}`../reference/nonlocal_parameters` for units.
