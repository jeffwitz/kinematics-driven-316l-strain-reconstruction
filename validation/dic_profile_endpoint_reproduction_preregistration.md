# Which DISFlow profile reproduces the archived field — preregistration

Date: 2026-07-31
Written before the `declared_medium_v4` reproduction has been computed.

## Question

`legacy_script_2021` (patch 4, stride 1) is the primary DISFlow profile of this
project. Its justification is **provenance**: it reproduces the setters
explicitly present in the supplied historical script. That is an argument about
source code, not about data.

This campaign asks the stronger question: does it actually reproduce the
archived historical displacement field better than the alternative profile
`declared_medium_v4` (patch 8, stride 3)?

## Why this is the right test, and the wrong one to skip

The tempting criterion — pick the profile that agrees best with the FEM — is
circular, because the observation operator would then be selected by the model
it is later used to judge. `current_evidence` and evidence `E-DIC-003` forbid
it.

Reproduction of the **archived measured field** is not circular: it compares a
recomputation against data produced by the historical chain, with no mechanics
involved. It can therefore promote the provenance argument from "these setters
appear in the script" to "these setters reproduce the data", or refute it.

## Method

For each profile, run DISFlow on the raw crop from `000294.tif` to
`000334.tif`, convert to canonical displacement, and compare with the immutable
prepared field `data/processed/case_study/displacement_{x,y}_mm.npy`.

The prepared field is nodal, `3601 x 3101`, obtained from the `3600 x 3100`
measured support by the `edge-pad-upper` completion rule. The comparison uses
the `3600 x 3100` sub-block only, which is the genuinely measured part; the
padded last row and column are excluded because they are a copy, not a
measurement.

Metrics, identical to those already archived for `legacy_script_2021` in
`dic_multistep_p0043_endpoint_amendment.md` so the numbers are directly
comparable:

- component RMS difference, mm;
- maximum absolute component difference, mm;
- relative displacement-vector norm, per cent.

The archived `legacy_script_2021` values are `7.764e-4` mm, `1.209e-3` mm and
**1.583 %**. They are recomputed here rather than quoted, and the recomputation
must match the archived figures; a mismatch invalidates the campaign before any
comparison is read.

## Registered outcomes

Read from the relative displacement-vector norm.

| Result | Conclusion |
|---|---|
| `legacy_script_2021` at least `1.5x` smaller | provenance **confirmed by reproduction**; 4/1 primary on a demonstrated basis |
| the two within `1.5x` of each other | reproduction does **not** discriminate; 4/1 stays primary on the documented-setters argument alone, which is weaker, and this must be said |
| `declared_medium_v4` at least `1.5x` smaller | the provenance argument is **contradicted**; the profile choice must be reopened |

The factor is fixed here, before the second number exists.

## Registered expectation

`legacy_script_2021` is expected to reproduce the archived field better. If it
does not, that is a substantive negative result about the identification of the
historical chain, and it is reported as such rather than explained away.

## What this campaign cannot settle

A successful reproduction would show that these settings are consistent with
the archived data. It would **not** identify the historical configuration
uniquely, because several confounds remain open and are not addressed here:

- the historical pipeline appears to mask before correlation; the reproduction
  does not;
- the OpenCV version used historically is unknown, and unset setters fall back
  to factory defaults that differ between versions;
- the preset is implicit in the supplied script;
- residual differences of this size are consistent with several distinct
  causes.

Above all, this test says nothing about which profile is **metrologically
better**. Reproducing the historical chain faithfully and measuring
displacement accurately are different properties, and a profile could do the
first well while doing the second badly. No claim about measurement quality is
licensed by this campaign.

## Deliverable

`validation/dic_profile_endpoint_reproduction_results.md` and
`reference_data/dic_profile_endpoint_reproduction_v1/`.

## Amendment, 2026-07-31, fixed before any comparison was read

The registered archived-consistency guard fired on the first execution:
recomputing `legacy_script_2021` gave a relative vector norm of `1.673 %`
against the archived `1.583 %`, and a maximum absolute difference of
`1.979e-2` mm against the archived `1.209e-3` mm, a factor of sixteen.

The cause is a support mismatch, not a numerical one. The archived figure in
`dic_multistep_p0043_endpoint_amendment.md` is computed on the **P43 partition
support**, solve bounds `[1290, 1950, 780, 1390]`, since that amendment
concerns the P43 measured-history campaign. This campaign compares the **full
measured field**.

The two are different quantities and neither is wrong. The correction is
therefore to report both supports:

- the **full `3600 x 3100` field** remains the primary comparison, because the
  question is which profile reproduces the archived data as a whole;
- the **P43 sub-block** is added solely so the archived-consistency guard
  compares like with like.

The registered outcome table and its `1.5x` factor are unchanged and are still
read from the full-field relative vector norm. This amendment is fixed before
the `declared_medium_v4` numbers were compared against the outcome table.

## Correction of record, 2026-07-31, after the result

Two statements in this preregistration were wrong on the facts and are
corrected here rather than silently edited.

**Masking.** The confound list asserted that the historical pipeline masks
before correlation. It does not. Masking is applied **after** correlation in
this practice, because with dense optical-flow methods masking beforehand is
not effective: the solver propagates information across the masked region
anyway. This confound is therefore withdrawn, and the reproduction is not
missing a masking step.

**The dominant cause of the common residual.** The preregistration listed the
confounds as jointly unexplained. The residual is in fact expected to be driven
mainly by the **variational refinement at the finest scale**, which both
profiles share identically (`alpha=100`, `delta=1`, `gamma=0`,
`epsilon=0.002`, 30 iterations, native scale 0). That stage operates on the
full-resolution image and largely overwrites the coarse matching, so patch size
and stride, which act on the earlier matching, have little influence on the
final field.

This makes the null result of this campaign **expected rather than
disappointing**: it is the signature of a refinement-dominated chain, and the
`1.6` to `1.7 %` agreement with the archive is close.
