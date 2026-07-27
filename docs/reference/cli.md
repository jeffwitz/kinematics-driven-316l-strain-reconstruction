# Command-line reference

**Category: Reference.**

The installed entry point is `fem-inhouse`. `fem-inhouse COMMAND --help` is the
authoritative option list for the installed revision.

| Command | Purpose |
|---|---|
| `backend` | report sparse and constitutive backend availability |
| `prepare-case` | create canonical DIC-driven inputs |
| `prepare-material-map-control` | derive homogeneous or jointly translated material-map controls |
| `partition` | create, solve, resume and stitch partition campaigns |
| `diagnose-section-equilibrium` | evaluate generalized section balance including lateral shear flux |
| `diagnose-nonlocality` | run an output-only Helmholtz diagnostic |
| `estimate-nonlocal-reference` | derive $H_{\mathrm{ref}}$ from a local campaign |
| `validate-coupled-nonlocal` | compare raw coupled fields with DIC |
| `plot-coupled-alpha-fields` | generate common-scale EVM and PEEQ figures |
| `select-dic-partition` | rank candidate observation regions |
| `validate-material-map-controls` | compare mapped, homogeneous and translated-map local campaigns |
| `identify-nonlocal` | run F0/F1 collection, selection, report and transfer preparation |

## Common option contracts

| Group | Options | Contract and defaults |
|---|---|---|
| outputs | `--output`, `--report`, `--difference` | preserved unless `--overwrite` is explicit |
| partition geometry | `--count` or `--parts-x` + `--parts-y`, `--padding` | legacy counts 25/100 remain supported; default padding 150 |
| constitutive backend | `--constitutive-backend`, `--mfront-library`, `--mfront-threads` | MFront is nominal; library and threads enter provenance |
| mechanical solve | `--increments`, `--max-newton-iterations`, `--residual-tolerance`, `--minimum-step-divisor` | 20, 15, `1e-6`, 1024 |
| nonlocal solve | `--nonlocal-*` | disabled unless requested; defaults: 58.88 um, fixed relaxation 0.5, tolerance `1e-6`, 15 iterations |

For `partition`, exactly one action is required: `--list-pending`,
`--partition-id`, `--solve-pending` or `--stitch`. For
`diagnose-nonlocality`, exactly one length unit is required. Confirmatory mode
also requires a pre-declared decision-threshold file.

## Identification subcommands

`identify-nonlocal` provides `inspect`, `screen-frozen`,
`run-low-fidelity`, `profile-h`, `select-candidates`,
`generate-high-fidelity-manifest`, `collect-results`, `report`, and
`prepare-transfer-validation`.

High-fidelity execution is deliberately absent from implicit selection:
generation of an F2 manifest and execution are separate user actions.

| Low-fidelity switch | Meaning |
|---|---|
| none | validate F1 rankings against existing F2 references |
| `--design` | run the configured sparse ranking design |
| `--identifiability-design` | run saturation, constant-$A_\chi$ and fixed-$\alpha$ experiments |

`--point` is repeatable for resumable candidate selection. `--workers`
defaults to one and must be positive. `--dry-run` reports intended work
without launching it.

## Exit and overwrite contract

Commands return non-zero on invalid metadata, missing required fields,
incompatible cache keys, non-finite data or solver failure. Existing
non-empty outputs are not overwritten unless the relevant command explicitly
accepts `--overwrite`.

Field comparison and coupled-validation threshold failures return non-zero
while retaining their reports. A low-fidelity collection with failed points
returns exit code 2 and preserves per-point status. Cache reuse requires
matching mesh, DIC, material, configuration, loading history, observation
operator, constitutive variant, fidelity and nonlocal parameters.

Operational examples are in {doc}`../how-to/index`.
