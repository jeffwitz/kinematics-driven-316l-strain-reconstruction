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
| `characterise-dic-measurement-chain` | measure DISFlow null response and synthetic spatial transfer |
| `replay-dic-observation` | pass archived FEM displacement through the image-level DISFlow operator |
| `diagnose-dic-photometric-quality` | relate direct image residuals to archived V3 FEM/DIC field errors |
| `diagnose-dic-boundary-history` | audit early DIC boundary states at recorded nonlinear-failure locations |
| `diagnose-dic-boundary-loading-subspace` | measure boundary temporal noise and the loading subspace |
| `filter-dic-boundary-history` | truncate a measured boundary history to its resolved modes |
| `compare-path-dependence` | compare final-state fields reached by two loading paths |
| `export-run-as-campaign` | present a multistep run in the layout the DISFlow replay expects |
| `diagnose-nonlocality` | run an output-only Helmholtz diagnostic |
| `estimate-nonlocal-reference` | derive $H_{\mathrm{ref}}$ from a local campaign |
| `validate-coupled-nonlocal` | compare raw coupled fields with DIC |
| `plot-coupled-alpha-fields` | generate common-scale EVM and PEEQ figures |
| `select-dic-partition` | rank candidate observation regions |
| `validate-material-map-controls` | compare mapped, homogeneous and translated-map local campaigns |
| `measure-ebsd-structural-length` | measure the preregistered EBSD/Schmid structural correlation scale |
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

## Independent EBSD/Schmid measurement

`measure-ebsd-structural-length` requires an HDF5 input containing
`/schmid/max_schmid_factor` and the three Euler datasets under
`/orientation`. It writes `report.json`, radial and directional CSV profiles,
and a common correlation figure. A non-empty output directory is rejected
unless `--overwrite` is supplied.

The command applies the estimator declared in
`validation/ell_ebsd_definition_preregistration.md`; it does not run FEM,
fit a micromorphic parameter or launch a coupled campaign.

## DIC measurement-chain characterisation

`characterise-dic-measurement-chain` requires `--images`, `--prepared-case`,
`--output` and `--figure-output`. By default it executes the full-crop
candidate-repeat test and the fixed-window synthetic transfer campaign.
`--profile` accepts `legacy_script_2021` or `declared_medium_v4`.
`--warp-mode` accepts `iterative_forward_inverse` or the regression-only
`legacy_approximate_inverse`. `--null-only` disables the synthetic stage. The
command requires the `measurement` optional dependency and rejects non-empty
outputs unless `--overwrite` is explicit.

Its manifest records the image hashes, crop, OpenCV and NumPy versions,
requested settings, values queried back from the OpenCV object, and figure
hashes. The command does not run FEM or modify a saved campaign.

`replay-dic-observation` requires `--campaign`, `--prepared-case`,
`--reference-image`, `--partition-id`, `--profile` and `--output`. It verifies
the archived displacement hash, resolves support bounds from the campaign
manifest, applies the corrected image warp and writes DIC, raw-FEM and
DISFlow-observed EVM separately. The default profile is
`legacy_script_2021`; `declared_medium_v4` is the declared sensitivity.

`diagnose-dic-photometric-quality` requires reference and final images, the
prepared case, output and figure directories, plus one or more
`--replay LABEL ALPHA PATH` triples. Every replay must be a completed
`legacy_script_2021` V3 artefact with matching core bounds and immutable field
hashes. The command performs no mechanics and does not replace primary
unmasked metrics by its q90-residual sensitivity.

`diagnose-dic-boundary-history` requires an immutable history array, its
report, a measured-history mechanical failure report, the raw-image directory,
and separate data and figure outputs. `--maximum-state` defaults to 6. It
separates affine and non-affine boundary motion, evaluates exact CPS4 strains
at the rejected elements, and computes direct photometric residuals. It does
not change the history, rerun mechanics, delete a frame or interpolate an
internal variable.

`propagate-dic-uncertainty` requires final and repeated-final images, the
prepared case, output and figure directories, plus one or more
`--replay LABEL ALPHA PATH` triples. It propagates contiguous windows of the
centred measured repeat-flow residual through the common EVM operator while
keeping every FEM field fixed. Defaults are `--samples 256` and
`--seed 20260729`. The reported intervals are surrogate sensitivities, not
confidence intervals; PEEQ is not propagated without a mechanical rerun.

`diagnose-dic-boundary-loading-subspace` requires an immutable history and its
report, plus data and figure outputs. It estimates per-state measurement noise
from temporal second differences, which is valid because the states are
independent direct correlations of one reference image, and separates loading
signal from noise by the temporal roughness of each SVD mode. States repaired
by interpolation are excluded from the noise estimate, read from the immutable
repair report rather than from a magnitude cutoff. It runs no mechanics.

`filter-dic-boundary-history` requires an immutable history and its report, and
writes a filtered copy whose boundary ring is a rank `--rank` reconstruction of
its deviation from the straight endpoint ramp. Origin, endpoint and the whole
interior stay bit-identical, and Dirichlet enforcement stays exact: only the
imposed data changes. The report states the RMS amplitude removed, in pixels,
against the measured noise floor. A filtered history is a drop-in input to
`run-dic-multistep-mechanics`.

`compare-path-dependence` requires a measured run directory, a proportional run
directory with the **same increment count**, and one archived field as
discretisation control. It evaluates the core only, padding excluded, and
withdraws its verdict if the control is not at least three times smaller than
the path effect. It also reports a band structure ratio separating a diffuse
noise-like excess from one concentrated in the localisation bands.

`export-run-as-campaign` copies a multistep run's `U.npy` into the
`manifest.json` plus `partitions/<id>/` layout that `replay-dic-observation`
expects, recomputing the hash from the copied array rather than carrying it
over. It exists so a measured-history run can pass through the archived
observation operator without duplicating that workflow.

`run-dic-multistep-mechanics` accepts `--record-newton-trace`, which writes an
observational per-iteration `newton_trace.csv` **including after a failed
solve**. It records the correction norm, residual, tangent diagonal statistics,
the constitutive-to-elastic tangent ratio and the degree of freedom carrying
the largest correction. The trace never alters the numerical path; that
invariance is asserted by a test requiring bitwise identical fields with and
without it.

## Boundary enforcement

`run_case_study` accepts `boundary_enforcement`, which defaults to
`"elimination"`. Elimination is the production formulation: prescribed degrees
of freedom are removed from the solved system, the constraint holds exactly and
conditioning is untouched.

`"penalty"` keeps them in the system with a finite spring of
`boundary_penalty_stiffness` on their diagonal, so the boundary becomes a
weighted data term rather than a hard constraint. It returns
`boundary_misfit_mm`, the nodal gap between the measurement and the value the
solver actually imposes, and the reaction becomes the spring force conjugate to
that gap. The consistency error scales as one over the spring stiffness, and
conditioning degrades once the spring far exceeds the tangent diagonal, so a
finite measurement-derived weight is intended rather than a large one. This
mode is diagnostic; no archived campaign uses it.
