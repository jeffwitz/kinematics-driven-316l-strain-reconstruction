# Run joint nonlocal identification

**Category: How-to.**

## Prerequisites

- a versioned identification YAML file;
- completed local and existing coupled campaigns listed by that file;
- a validated DIC observation operator;
- no unresolved mismatch in numerical policy between reused campaigns.

The repository contains one fully versioned example configuration:

```bash
CONFIG=configs/joint_nonlocal_identifiability_p0043_newton25.yaml
```

That file is tied to the preserved case-study campaigns. Copy it to a new name
before changing paths, observation regions or numerical policy. Never edit a
configuration after a campaign using its hash has started.

## Inspect and screen

```bash
fem-inhouse identify-nonlocal inspect \
  --config "$CONFIG" \
  --dry-run

fem-inhouse identify-nonlocal screen-frozen \
  --config "$CONFIG"
```

F0 writes dense frozen-field diagnostics. It does not run mechanics.

## Validate the reduced model

```bash
fem-inhouse identify-nonlocal run-low-fidelity \
  --config "$CONFIG"
```

This first F1 action replays the reduced counterparts of existing
high-fidelity cases. Do not use F1 for candidate selection unless the
pre-declared ranking and metric-error gates pass.

## Run the discriminating F1 design

```bash
fem-inhouse identify-nonlocal run-low-fidelity \
  --config "$CONFIG" \
  --identifiability-design
```

This second action runs the homogeneous saturation, constant-$A_\chi$ and
fixed-alpha experiments declared in the YAML. Use `--point` to resume or select
one candidate and `--workers` to cap concurrency.

## Collect and select

```bash
fem-inhouse identify-nonlocal collect-results \
  --config "$CONFIG"
fem-inhouse identify-nonlocal profile-h \
  --config "$CONFIG"
fem-inhouse identify-nonlocal select-candidates \
  --config "$CONFIG"
fem-inhouse identify-nonlocal report \
  --config "$CONFIG"
```

The report keeps amplitude, localization and spatial-scale metrics separate.

## Generate, but do not launch, F2

```bash
fem-inhouse identify-nonlocal generate-high-fidelity-manifest \
  --config "$CONFIG" \
  --dry-run
```

Review cost, parameters, duplicate detection and the scientific role of every
candidate. High-fidelity execution requires a separate explicit action. The
command must refuse generation when the discriminating design is incomplete,
numerically censored or still reaches an unbounded coupling boundary.

See {doc}`../explanation/parameter_identification` for the design logic and
{doc}`../reference/nonlocal_parameters` for units.
