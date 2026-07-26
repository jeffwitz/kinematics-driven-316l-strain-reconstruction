# P43 joint nonlocal identification record

This directory is the versioned compact record of the staged P43
identification campaign generated on 2026-07-26.

It contains:

- the complete F0 frozen-field table and per-length diagnostics;
- the F0 trend comparison with existing F2 calculations;
- the F1-versus-F2 validation report;
- the immutable consolidated table containing 13 F1 and four F2 rows;
- the bounded \(H_\chi\) profiles;
- the amplitude-localization Pareto selection;
- the log-coordinate identifiability diagnostic;
- the four-point high-fidelity proposal;
- the no-recalibration transfer manifest.

The source configuration is:

```text
configs/joint_nonlocal_identification_p0043.yaml
```

The collection key is:

```text
e8addb3214115762b15bd8f697b000cd267a57b30980ecb8638ec41c00847ff2
```

The generated F2 proposal key is:

```text
8599ffb746112951e44bf3ae67f2ff0965da526f5e01f26233f7af8c075ac90e
```

No proposed F2 calculation was launched. Every candidate in
`f2_proposal_manifest.json` has status `proposed_not_run`; explicit human
approval remains required.

The method, figures, timing interpretation and temporary scientific
conclusions are documented in:

```text
docs/explanation/joint_nonlocal_identification.md
```

The operational commands are documented in:

```text
docs/how-to/run_joint_nonlocal_identification.md
```

## SHA-256

```text
c3ccd2e227a80f8dfb033b87a7579ad291979da7ecf45a2f5c7ae7c99054a7cc  f0_frozen_screen.csv
786818f3fb2511f868105c7b9ff8441e879ea4e98e18796a27bc116f79cee0a4  f0_length_diagnostics.json
ba65703d8d3f685a32c70f4f912c5cccf1449719931c973feebbbcef6c1feec0  f0_proxy_validation.json
615b648024c4f8e4080a00d44642f8bbb01d2cbc6c6399e9bae7cbf88ef4b605  f1_validation.json
88175f769ad59d4b8c0c4cbcefa62460261775c871c29b9c9556928755a644f5  f2_proposal_manifest.json
874c0a0f97cc766c7bb6c36e346f333e1f447742b6db0ab6dc5fba63eec331de  h_profile.json
a88068bcb8c22c7d4c13ae7dcee9568debb0ecb668a2cf678775a20d984fca77  identifiability.json
82c087206d8cb5cfcf3a5fed5f2d1af00f8f64cd95277da81bf3e0beb3a3fa50  pareto_selection.json
a8e7c342caa78b020081d804c03049501094f06626ad35db8c1eca92cc0117fd  results.csv
6329b84f991053cf960fcf06639d69bdb2912c585b38d8925db0dd39ae462db8  results.json
c63e43c1e47be1318355f27259accb3b3cfc40b6eb3b51ad5eb59f56901e86a4  transfer_validation_manifest.json
```
