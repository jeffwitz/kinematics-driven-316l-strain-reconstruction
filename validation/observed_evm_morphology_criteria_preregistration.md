# Observed-EVM comparison, criteria set v2 with morphology — preregistration

Date: 2026-07-31
Supersedes the Pareto criteria of
`observed_evm_candidate_comparison_preregistration.md`. Everything else in that
document — candidates, observation operator, Otsu segmentation, band geometry,
bootstrap scheme, decision vocabulary — is carried over unchanged.

**Not to be run until validated.**

## Why a second criteria set

The first run fired registered failure condition 2: five of six candidates were
non-dominated, including the translated-map negative control, whose whole
purpose is to be rejected. Its mass error was statistically indistinguishable
from `alpha = 1` and `alpha = 4`.

The morphology block separated it immediately — eccentricity `0.645` against
`0.79`, minor axis `269` against `200 px` — but morphology was not in the
criteria set. The set was section-based and therefore blind to exactly the
defect the control was built to expose.

## What this document cannot claim

**This criteria set is not blind, and the honest statement is stronger than
last time.** I have already seen the morphology of all six fields, and I know
that morphology is what separates the negative control. Choosing morphology
criteria now is choosing them *because* they gave the answer I wanted on a
control.

That is a real weakness. It is not repaired by wording. It is bounded by three
devices, each of which operates on something never computed:

1. **bounds and senses come from the DIC and the measurement chain**, never
   from a candidate value;
2. **two gates must pass before any candidate is read**, both on quantities
   never produced: a falsification bench and a self-consistency check;
3. **a blind confirmation set**: the `declared_medium_v4` profile has never been
   through this pipeline, and the verdict must survive on it.

If the reader trusts nothing else here, gate G1 and the blind confirmation are
the parts that carry evidential weight.

## Prerequisite P0 — done before registering, and why it was needed

Checking the v1 archive before writing this document turned up a defect in the
first campaign. Its `report.json` recorded:

```
homogeneous -> .../scratchpad/obs_homogeneous/fem_observed_evm.npy
translated  -> .../scratchpad/obs_translated/fem_observed_evm.npy
```

**Both controls were read from a session scratchpad, not from
`validation/reference_data/`.** The four models were archived with SHA-256; the
two controls were not, and that directory does not persist. The v1 result was
therefore not reproducible on the very field the acceptance test below rests
on.

Repaired immediately rather than registered as future work, because the
scratchpad could have been wiped at any moment and the control would then have
had to be regenerated. Both fields are now in
`reference_data/observed_evm_controls_p0043_v1/` under Git LFS, with a
`SHA256SUMS` manifest, and **both `fem_observed_evm.npy` reproduce the v1
hashes exactly** — `ca4044fe…` and `1cd91dbb…`. These are the fields v1 used.
The `fem_displacement_image_grid.npy` are archived with them so the controls
can be re-observed through the second DISFlow profile without rerunning
mechanics.

This changes no number in v1; it makes v1 checkable. The v1 results document is
annotated accordingly.

## Gate G1 — the criteria must rank known defects correctly

The falsification bench of lot 2 exists and **has never been run**. Before any
candidate is read, each criterion is applied to the registered defect ladder
built from the DIC itself, using the generators of
`fem_inhouse.validation.falsification_cases`: translation of 1, 4 and 16 px;
amplitude at 0.9 and 1.5; width at 0.8 and 1.2; one band removed; one band
interrupted; one spurious band added.

Registered severity order, most severe first: a missing band, a spurious band,
a shift comparable with the band width, a `20 %` width error, a `10 %`
amplitude error.

**A criterion whose ranking contradicts that order is removed from the decision
before any candidate is scored.** Which criteria survive is not known now.

## Gate G2 — the reference must score perfectly against itself

The DIC compared with itself must give, on every criterion, the value that
means "identical": zero object-count error, zero eccentricity error, unit
minor-axis ratio, zero centreline and mass error, minimum skilful scale of
`1 px`, and detection in `100 %` of valid sections.

This gate is not ceremonial. In the first run it is what exposed the detection
defect: the DIC scored `0.46` against its own first band because section
validity had been conflated with band detection. **Any criterion failing G2 is
defective and is fixed or removed before the campaign proceeds.**

## Criteria set v2

Elimination, unchanged in spirit and deliberately minimal:

| # | Criterion | Bound | Origin |
|---|---|---|---|
| E1 | converged, plane-stress residual valid | as archived | solver contract |
| E2 | each DIC band carried in at least half its **valid** sections | `0.50` | below half, "reproduced" is not defensible |
| E3 | at least one object of at least `256 px` | `1` | a field with no band cannot be compared |

E2 and E3 are the rules that eliminated the homogeneous control correctly. No
elimination bound is taken from a candidate value.

Pareto criteria, reduced and non-redundant, one per family. Morphology
contributes three, not one: a candidate can get the object count right and the
width wrong, or both right and the shape wrong, so they fail independently and
collapsing them would hide exactly what v1 missed.

| Family | Criterion | Sense |
|---|---|---|
| morphology, topology | object-count error against the DIC | lower |
| morphology, width | `abs(log(minor axis ratio))`, worst paired object | lower |
| morphology, shape | eccentricity error, worst paired object | lower |
| position | median centreline error, worst band | lower |
| amplitude | median integrated-mass error, worst band | lower |
| multiscale | minimum scale reaching FSS `0.7` at q90 | lower |
| residual | corridor fraction of residual energy | lower |

Width error per section is **dropped**: it measures the same thing as the
minor-axis ratio and the set must stay non-redundant. The minor-axis ratio is
taken in log so that twice too wide and half too wide are penalised equally.

Every criterion is on the **worst** band or the worst paired object, as a
vector, never summed.

## The acceptance test

**The criteria set is accepted only if the translated control is dominated or
eliminated.** This is registered as a pass condition of the *criteria*, not of
the candidates, and it is checked before any ranking is looked at.

The reasoning is the one v1 established: that control has the right
microstructure in the wrong place. A criteria set that cannot separate it from
a coupled model is not measuring placement, whatever else it measures. If it
survives again, no ranking is published and the conclusion is about the
observable, not about the models.

## Registered expectations, recorded before running

- **object-count error is expected to be non-zero for every candidate**, since
  all four merged the two bands in the first run. Stating it now prevents it
  being presented later as a discovery of this campaign;
- **the translated control is expected to be dominated** once morphology enters.
  If it is not, the criteria set fails again and that is the result;
- the front may still be degenerate. Failure condition 2 stands.

## Blind confirmation on the second DISFlow profile

Everything above is repeated on `declared_medium_v4`, patch 8 stride 3, against
`legacy_script_2021`, patch 4 stride 1, which stays primary by provenance.
**This is the only genuinely blind part of the campaign** and it is where the
non-blindness of the criteria set is paid for.

What exists and what does not:

- the four models are archived under `declared_medium_v4` in
  `dic_symmetric_observation_p0043_v1`, and **none has been through this
  pipeline**;
- the DIC is archived under that profile too;
- **neither control has a `declared_medium_v4` observation.** They must be
  produced by replaying the archived control displacement grids through the
  second profile. No mechanics is rerun — only the observation operator, on a
  displacement field that already exists.

That second point matters: the acceptance test needs the translated control, so
the confirmation is incomplete without it. If the control replay cannot be
produced, the confirmation is reported as **not performed**, and the campaign
rests on the primary profile alone with the non-blindness unpaid.

Otsu is **recomputed from the DIC of the second profile** and frozen there. It
is not carried over from `legacy_script_2021`: a different spatial transfer
gives a different EVM distribution, and reusing the first threshold would
compare fields against a boundary derived from other data. Two profiles, two
thresholds, each frozen from its own DIC.

If the profiles disagree on which candidates are dominated, the result is
reported as **profile-dependent** and no ranking is published.

## Failure conditions

1. a criterion fails G1 or G2 — it is removed or fixed, and that is reported;
2. no candidate passes E1 to E3 — reported, campaign stops;
3. the front holds more than half the survivors — the criteria still do not
   discriminate, reported, no ranking;
4. **the translated control remains non-dominated** — the criteria set has
   failed a second time, and the conclusion is that this observable and this
   descriptor family cannot separate placement from coupling on this ROI;
5. the two profiles disagree — profile-dependent, no ranking;
6. the control replay on the second profile cannot be produced — the blind
   confirmation is reported as not performed, and every conclusion is labelled
   as resting on a non-blind criteria set.

Outcome 4 is the one to watch. It would be a substantive negative result about
the comparison itself and must not be softened.

## What no outcome licenses

No micromorphic parameter is selected by this campaign, whatever the front
shows. Selection needs the separately registered identification campaign, and
that campaign is itself gated on the observable being able to discriminate —
which is exactly what is in doubt here.

One ROI, one loading path, an observed EVM field. Nothing about internal
stresses; PEEQ is not a DIC observable. The FEM fields are smoother than the
DIC because the replay adds no speckle-decorrelation noise, so texture is never
counted as model error.

## Deliverable

`validation/observed_evm_morphology_criteria_results.md` and
`reference_data/observed_evm_morphology_criteria_p0043_v1/`, in a commit
separate from this one. The control archive of P0 is already in place.

Both profiles are reported side by side in one table. The gate results are
reported whatever they say, including any criterion the falsification bench
throws out.
