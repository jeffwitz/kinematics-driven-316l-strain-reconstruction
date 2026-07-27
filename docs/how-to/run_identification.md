# Run joint nonlocal identification

**Category: How-to.**

## Prerequisites

- a versioned identification YAML file;
- completed local and existing coupled campaigns listed by that file;
- a validated DIC observation operator;
- no unresolved mismatch in numerical policy between reused campaigns.

The versioned Newton-25 configuration is the current executable example:

```bash
cp configs/joint_nonlocal_identifiability_p0043_newton25.yaml \
  configs/my_identification.yaml
```

Edit the copied paths and ROI metadata before running it. Campaign-specific
identifiers belong in configuration, not in the workflow implementation.

## Inspect and screen

```bash
fem-inhouse identify-nonlocal inspect \
  --config configs/my_identification.yaml \
  --dry-run

fem-inhouse identify-nonlocal screen-frozen \
  --config configs/my_identification.yaml
```

F0 writes dense frozen-field diagnostics. It does not run mechanics.

## Validate and run F1

```bash
fem-inhouse identify-nonlocal run-low-fidelity \
  --config configs/my_identification.yaml

fem-inhouse identify-nonlocal run-low-fidelity \
  --config configs/my_identification.yaml \
  --identifiability-design
```

The first call runs the configured F1 validation points and checks their
ranking against existing F2 reference cases. Only after all validation gates
pass does the second call run the homogeneous saturation, constant-$A_\chi$
and fixed-$\alpha$ discriminating design. Use `--point` to resume or select
one candidate and `--workers` to cap concurrency.

## Collect and select

```bash
fem-inhouse identify-nonlocal collect-results \
  --config configs/my_identification.yaml
fem-inhouse identify-nonlocal profile-h \
  --config configs/my_identification.yaml
fem-inhouse identify-nonlocal select-candidates \
  --config configs/my_identification.yaml
fem-inhouse identify-nonlocal report \
  --config configs/my_identification.yaml
```

The report keeps amplitude, localization and spatial-scale metrics separate.

## Generate, but do not launch, F2

```bash
fem-inhouse identify-nonlocal generate-high-fidelity-manifest \
  --config configs/my_identification.yaml \
  --dry-run
```

Review cost, parameters, duplicate detection and the scientific role of every
candidate. High-fidelity execution requires a separate explicit action.

See {doc}`../explanation/parameter_identification` for the design logic and
{doc}`../reference/nonlocal_parameters` for units.
