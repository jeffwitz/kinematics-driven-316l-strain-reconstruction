# Run the staged joint nonlocal identification

This guide operates the P43 workflow described in
{doc}`../explanation/joint_nonlocal_identification`. It never launches a
full-resolution F2 calculation implicitly.

## Activate the environment

```bash
source /home/jeff/.local/share/tfel/env/env.sh
source .venv/bin/activate
```

All commands below use the versioned configuration:

```text
configs/joint_nonlocal_identification_p0043.yaml
```

## Inspect before writing

```bash
fem-inhouse identify-nonlocal inspect \
  --config configs/joint_nonlocal_identification_p0043.yaml
```

The inspection verifies the local campaign, P43 metadata, `HREF.json`,
prepared DIC data, existing F2 campaigns and planned fidelity levels.

Use `--dry-run` on every mutating action when checking a new configuration:

```bash
fem-inhouse identify-nonlocal run-low-fidelity \
  --config configs/joint_nonlocal_identification_p0043.yaml \
  --design \
  --dry-run
```

## Run or reuse the F0 DCT screen

```bash
fem-inhouse identify-nonlocal screen-frozen \
  --config configs/joint_nonlocal_identification_p0043.yaml
```

The output is under:

```text
results/joint-nonlocal-identification-p0043/f0/
```

It contains the cache manifest, 463 parameter records, one diagnostic per
length and the comparison with existing F2 trends. Re-running an identical
command reuses the result. An incompatible non-empty cache is rejected.

## Validate F1 before using it

The validation points replay the local case and the existing positive F2
couplings at 58.88 µm:

```bash
fem-inhouse identify-nonlocal run-low-fidelity \
  --config configs/joint_nonlocal_identification_p0043.yaml
```

The workflow checks ranking and numerical differences against all declared F2
reports. Candidate selection remains disabled if the validation gates fail.

## Run the sparse F1 design

Preview:

```bash
fem-inhouse identify-nonlocal run-low-fidelity \
  --config configs/joint_nonlocal_identification_p0043.yaml \
  --design \
  --dry-run
```

Explicit execution:

```bash
fem-inhouse identify-nonlocal run-low-fidelity \
  --config configs/joint_nonlocal_identification_p0043.yaml \
  --design
```

Each point has an independent status and immutable cache. The action
continues after a point failure and returns status 2 if at least one point
failed. It does not reuse a partial constitutive state.

Run targeted adaptive points with selectors of the form `alpha:ell_um`:

```bash
fem-inhouse identify-nonlocal run-low-fidelity \
  --config configs/joint_nonlocal_identification_p0043.yaml \
  --point 2:20 \
  --point 2.5:20
```

Do not change solver settings merely to make a difficult DOE point converge.
A clean failure and a new bracket are preferable to an unregistered numerical
change.

## Collect, profile and build the Pareto front

```bash
fem-inhouse identify-nonlocal collect-results \
  --config configs/joint_nonlocal_identification_p0043.yaml

fem-inhouse identify-nonlocal profile-h \
  --config configs/joint_nonlocal_identification_p0043.yaml

fem-inhouse identify-nonlocal select-candidates \
  --config configs/joint_nonlocal_identification_p0043.yaml
```

Collections are immutable and keyed by all source hashes plus the Git
revision. The profile uses PCHIP only within converged samples and records
whether the optimum lies on a sampled boundary.

## Generate the F2 proposal without running it

Preview:

```bash
fem-inhouse identify-nonlocal generate-high-fidelity-manifest \
  --config configs/joint_nonlocal_identification_p0043.yaml \
  --dry-run
```

Write the immutable proposal:

```bash
fem-inhouse identify-nonlocal generate-high-fidelity-manifest \
  --config configs/joint_nonlocal_identification_p0043.yaml
```

The manifest is under:

```text
results/joint-nonlocal-identification-p0043/f2-proposals/
```

It includes parameters, units, purpose, estimated cost, destination and the
complete command for each candidate. The action cannot execute those
commands. Review and explicit human approval are mandatory.

## Generate the report and documentation figures

```bash
fem-inhouse identify-nonlocal report \
  --config configs/joint_nonlocal_identification_p0043.yaml
```

The command writes:

- an immutable Markdown/JSON report under the campaign;
- SVG, PNG and PDF figures under the report directory;
- the same documentation assets under
  `docs/_static/joint_identification/`.

No mechanical calculation is performed.

## Prepare transfer validation

```bash
fem-inhouse identify-nonlocal prepare-transfer-validation \
  --config configs/joint_nonlocal_identification_p0043.yaml
```

The produced manifest contains at most three frozen candidates, forbids
recalibration and remains in `awaiting_validation_roi` until another
band-containing ROI is declared. It launches nothing.

## Resume after interruption

Run the same action again. Completed compatible F0/F1 points and immutable
collections are reused. A missing or failed point can be selected explicitly
with `--point`. Never delete or overwrite an old campaign to change one
parameter; use a new cache key and output directory.

## Verify the documentation

```bash
sphinx-build -W --keep-going -b html docs docs/_build/html
make -C docs latexpdf SPHINXOPTS="-W --keep-going"
```

Open:

```text
docs/_build/html/explanation/joint_nonlocal_identification.html
```

The generated PDF is:

```text
docs/_build/latex/kinematics-driven316lstrainreconstruction.pdf
```
