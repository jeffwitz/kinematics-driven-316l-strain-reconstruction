# Which DISFlow profile reproduces the archived field — results

Date: 2026-07-31
Preregistration: `dic_profile_endpoint_reproduction_preregistration.md`,
including its amendment on the comparison support.
Primary machine-readable result:
`reference_data/dic_profile_endpoint_reproduction_v1/report.json`.

## Short answer

**Reproduction does not discriminate.** The two profiles reproduce the archived
displacement field almost identically: `1.673 %` for `legacy_script_2021` (4/1)
against `1.738 %` for `declared_medium_v4` (8/3), a ratio of `1.04` against a
registered discrimination factor of `1.5`.

The registered expectation, that 4/1 would reproduce the archived field better,
is **not met** in any meaningful sense, and on the P43 sub-block 8/3 is
marginally ahead. 4/1 therefore remains primary on the documented-setters
argument alone, which is weaker than a demonstrated reproduction.

## Numbers

`000294 -> 000334`, OpenCV 4.14, against the immutable prepared field.

| Profile | patch / stride | support | RMS (mm) | max (mm) | relative norm |
|---|---|---|---:|---:|---:|
| `legacy_script_2021` | 4 / 1 | full `3600x3100` | `8.516e-4` | `1.979e-2` | **1.6726 %** |
| `declared_medium_v4` | 8 / 3 | full `3600x3100` | `8.847e-4` | `2.381e-2` | **1.7376 %** |
| `legacy_script_2021` | 4 / 1 | P43 | `7.764e-4` | `1.209e-3` | 1.5831 % |
| `declared_medium_v4` | 8 / 3 | P43 | `7.750e-4` | `1.242e-3` | 1.5802 % |

4/1 is ahead by `3.9 %` on the full field and behind by `0.19 %` on P43. Both
are far inside the registered `1.5x` factor.

### The archived guard fired, and was informative

The registered consistency check failed on first execution: the recomputed
`legacy_script_2021` relative norm was `1.673 %` against the archived
`1.583 %`, with a maximum difference sixteen times larger. The cause was a
support mismatch — the archived figure in
`dic_multistep_p0043_endpoint_amendment.md` is computed on the P43 partition,
not the full field. After adding the P43 support, the recomputation reproduces
the archive **exactly**: `1.5831 %` against `1.583 %`, and `1.209e-3` mm
against `1.209e-3` mm.

This is worth recording as a method point: quoting an archived number without
recomputing it would have silently compared two different quantities.

## What this actually shows

**Patch and stride are not what separates the reproduction from the archive.**
Both profiles leave a residual near `1.6` to `1.7 %`, and changing patch size by
a factor of two and stride by a factor of three moves that residual by `4 %` of
itself. The common residual is therefore dominated by something else: the
unknown historical OpenCV version and its factory defaults for unset setters,
the masking applied before correlation in the historical pipeline but not here,
and the implicit preset. Those confounds were listed in the preregistration and
remain open; this campaign now shows they dominate.

**And that makes the score-based criterion worse, not better.** The two profiles
are nearly indistinguishable on their ability to reproduce the measured data,
yet they differ by `1.8` noise margins on relative L2 in the FEM/DIC comparison,
which is more than the loading path or the boundary filter. A choice with almost
no grounding in the data moves the agreement score more than two campaigns of
mechanics.

That is the circularity the project's rule exists to prevent, now quantified: a
profile selected on agreement would be selected on essentially nothing
measurable in the data itself.

## Consequence for the profile decision

`legacy_script_2021` stays primary. Nothing here contradicts it, and it is the
profile whose setters are documented in the supplied historical script.

But its justification must be stated accurately:

- **it is not** the profile demonstrated to reproduce the archive best, because
  no profile is;
- **it is** the profile whose settings are traceable to the supplied source;
- the honest statement is *primary by documented provenance, with reproduction
  unable to discriminate*.

Identifying the historical chain would require closing the confounds above,
starting with masking before correlation, which is the one difference known to
exist and not yet reproduced.

## Claim boundary

This compares reproduction fidelity only. It says nothing about which profile
measures displacement more accurately; a profile can reproduce a historical
chain faithfully while being metrologically poor, and the two properties are
not tested against each other anywhere in this project.

## Reproduction

```bash
fem-inhouse compare-profile-reproduction \
  --images <DIC_images> \
  --prepared-case data/processed/case_study \
  --output validation/reference_data/dic_profile_endpoint_reproduction_v1
```
