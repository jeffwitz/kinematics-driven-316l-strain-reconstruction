# Command-line reference

**Category: Reference.**

The installed entry point is `fem-inhouse`. `fem-inhouse COMMAND --help` is the
authoritative option list for the installed revision.

| Command | Purpose |
|---|---|
| `backend` | report sparse and constitutive backend availability |
| `prepare-case` | create canonical DIC-driven inputs |
| `partition` | create, solve, resume and stitch partition campaigns |
| `diagnose-nonlocality` | run an output-only Helmholtz diagnostic |
| `estimate-nonlocal-reference` | derive $H_{\mathrm{ref}}$ from a local campaign |
| `validate-coupled-nonlocal` | compare raw coupled fields with DIC |
| `plot-coupled-alpha-fields` | generate common-scale EVM and PEEQ figures |
| `select-dic-partition` | rank candidate observation regions |
| `identify-nonlocal` | run F0/F1 collection, selection, report and transfer preparation |

## Identification subcommands

`identify-nonlocal` provides `inspect`, `screen-frozen`,
`run-low-fidelity`, `profile-h`, `select-candidates`,
`generate-high-fidelity-manifest`, `collect-results`, `report`, and
`prepare-transfer-validation`.

High-fidelity execution is deliberately absent from implicit selection:
generation of an F2 manifest and execution are separate user actions.

## Exit and overwrite contract

Commands return non-zero on invalid metadata, missing required fields,
incompatible cache keys, non-finite data or solver failure. Existing
non-empty outputs are not overwritten unless the relevant command explicitly
accepts `--overwrite`.

Operational examples are in {doc}`../how-to/index`.
