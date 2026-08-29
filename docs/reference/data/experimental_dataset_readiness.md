# Experimental dataset readiness

This contract records whether an experimental dataset is sufficiently
specified for a particular scientific use.  Readiness is cumulative: a case
can be useful for one method while remaining unsuitable for a stronger claim.
The levels describe the dataset contract, not the success of a constitutive
model.

## Readiness levels

### R0 — Archived / inspectable

The case can be inspected and traced.  It has a specimen identifier, source or
raw files, hashes where available, units, dimensions, acquisition provenance,
preparation scripts and versions, and coordinate conventions.

### R1 — Ready for mechanical comparison

R0 plus a usable DIC mechanical contract:

* DIC provenance, physical scale and the canonical $u_x/u_y$ convention;
* support, crop and valid mask;
* ordered loading states and an explicit boundary extraction rule;
* specimen geometry and thickness, when required by the forward model;
* retained interior DIC as an observation rather than an imposed field;
* a declared observation operator and an explicit missing/outlier policy.

Reaction force is useful but not mandatory for a full-Dirichlet forward.  Its
absence must remain recorded rather than silently reconstructed.

### R2 — Ready for heterogeneous / EBSD-driven mechanics

R1 plus an EBSD and registration contract:

* EBSD source, native step, scan origin and scan axes;
* orientation convention, phase information and crystal/sample frames;
* an independently supported physical DIC--EBSD transformation;
* common landmarks or fiducials, transformation parameters and registration
  uncertainty or residual;
* a declared interpolation or material-point assignment rule.

Array correspondence is not physical co-registration.  A better mechanical
fit is not registration proof.

### R3 — Ready for inverse material identification

R2 plus a reproducible forward and inverse contract:

* constitutive implementation/version and parameter convention;
* declared observation used for fitting;
* a meaningful plastic loading range;
* sensitivity and identifiability analysis;
* physically motivated bounds or priors;
* preferably independent reaction/force information and/or multiple
  experiments.

R3 does not mean that all parameters are identifiable.  It means that the
experiment is specified well enough for identifiability to be assessed
honestly.

## Status vocabulary

Every item should use exactly one of these statuses and cite its source:

```text
KNOWN          directly documented or independently checked
DERIVED        produced by a declared preparation transform
ASSUMED        working hypothesis, not independent proof
MISSING        required but unavailable or not recorded
NOT_APPLICABLE not required for the declared use
```

## Reusable readiness template

| Item | Status | Source | Units / frame | Required level | Notes |
|---|---|---|---|---|---|
| raw DIC images | KNOWN | source manifest / external delivery | image pixels | R0 | received externally; retain source and hash when available |
| DIC displacement fields | DERIVED | preparation report | mm, canonical $[u_x,u_y]$ | R1 | record transform and support |
| DIC scale | DERIVED | calibration/preparation manifest | mm/pixel | R1 | do not reuse as EBSD step |
| axes / components | KNOWN | DIC axis contract | specimen/image frame | R1 | include row/column mapping |
| loading-state order | ASSUMED | case inventory | ordered states | R1 | timestamps may still be missing |
| acquisition timestamps | MISSING | — | physical time | R1/R3 | distinguish order from synchronized time |
| reaction force | MISSING | — | force units | R1 optional, R3 preferred | absence does not block full-Dirichlet mechanics |
| geometry / thickness | MISSING | specimen record | length units | R1 | replace with KNOWN only when the source is documented; state exactly what the forward uses |
| EBSD source | KNOWN | HDF5/export manifest | crystal/sample frame | R2 | received externally; payload must be versioned |
| EBSD native step | MISSING | — | length/pixel | R2 | never infer from DIC dimensions |
| EBSD axes / origin | MISSING | — | acquisition frame | R2 | needed for physical registration |
| orientation convention | KNOWN | EBSD orientation contract | $Q_{global\to material}$ | R2 | include phase and handedness |
| DIC--EBSD transform | ASSUMED | declared $F$ mapping | specimen frame | R2 | numerical assignment is not proof |
| registration residual / uncertainty | MISSING | — | declared metric | R2 | independent support required |
| repeated-state / noise data | DERIVED | repeated-frame record | DIC units | R1 | full temporal covariance may remain missing |

The template is intentionally a contract, not a checklist to complete by
inference.  If an item is unavailable, retain `MISSING` and state which claims
are consequently out of scope.

## Go / no-go rules

### Mechanical model comparison

**GO** when geometry, DIC coordinates and units, and an ordered boundary
history are reliable.  The interior DIC field remains the observed quantity.

### EBSD-driven crystal-plasticity mechanics

**GO** only when the mechanical comparison is R1-ready and EBSD metadata plus
an independently supported physical co-registration are R2-ready.

### FEMU or material identification

**GO** only when the EBSD/mechanical dataset is R2-ready, the model shows
plausible relevance, the fitted observation is declared, and
sensitivity/identifiability have been considered.  Otherwise the result is
**methodological use only**.

## What each level enables

The same thresholds apply to the methods developed in this repository:

| Level | Defensible uses |
|---|---|
| R0 | archive inspection, provenance and preparation audits |
| R1 | full-Dirichlet spectral mechanics, DIC observation comparison, the qualified $A/A^T$ operator, and DIC-driven dissipative plastic reconstruction |
| R2 | EBSD-driven heterogeneous mechanics, FCC-coordinate interpretation and orientation-dependent constitutive comparisons |
| R3 | FEMU parameter studies, experiment-level identifiability assessment and multi-case constitutive-law discovery |

In particular, DIC-driven dissipative reconstruction needs R1 to study

```text
DIC defect -> A^T -> mechanically relevant plastic correction
```

but interpreting that correction in FCC coordinates needs R2.  Any attempt to
infer a transferable constitutive law from several experiments requires R3.

## Current P43 status

P43 is inspectable but does not yet satisfy every item needed for the higher
levels:

| Level | Current status | Reason |
|---|---|---|
| R0 | **partial** | 42 external TIFF frames and prepared fields are inventoried, but the raw acquisition/software provenance is not fully versioned in the repository |
| R1 | **working / partial** | canonical DIC scale, axes, crop and a working ordered history exist; synchronized timestamps, force history and some geometry metadata are unavailable |
| R2 | **no-go** | EBSD native step, scan axes/origin and independent physical DIC--EBSD registration are not proven (`registration_proven=false`) |
| R3 | **no-go** | R2 is not met, experimental material calibration is open, and the available observation/sensitivity contracts do not yet justify a generalized identification claim |

These statuses do not invalidate the numerical methods.  They delimit what a
P43 calculation may be used to claim.  The detailed P43 source boundary is in
{doc}`experimental_data_inventory`, with the axis, observation and EBSD
contracts in {doc}`dic_axis_conventions`,
{doc}`../scientific/observation_operator` and
{doc}`../scientific/ebsd_orientation_contract`.
