# Experimental dataset readiness

Readiness is defined relative to a scientific claim and to the inputs used by
the declared model. It is not a universal ladder that every dataset must
climb before any inverse problem is allowed. R0 and R1 are cumulative; R2 is a
modality-specific extension for EBSD-resolved mechanics; R3 adds the
requirements of the particular inverse problem.

## Readiness levels

### R0 — Archived / inspectable

The case can be inspected and traced. It has a specimen identifier, source or
raw files, hashes where available, units, dimensions, acquisition provenance,
preparation scripts and versions, and coordinate conventions. R0 may still be
partial when external source material or acquisition metadata is not fully
versioned.

### R1 — Ready for mechanical comparison

R1 requires R0 plus the information needed by the chosen mechanical
comparison:

* DIC provenance, physical scale and the canonical $u_x/u_y$ convention;
* support, crop, valid mask and an explicit boundary extraction rule;
* an ordered loading-state path;
* specimen geometry and thickness when required by the forward model;
* retained interior DIC as an observation rather than an imposed field;
* a declared observation operator and missing/outlier policy.

An ordered path is required for a path-dependent model. Absolute physical
timestamps are not universally required: they are a model-dependent
requirement for a rate-dependent law. Reaction force is an independent and
valuable observable, but its absence does not make a full-Dirichlet
displacement comparison undefined.

### R2 — Ready for heterogeneous / EBSD-driven mechanics

R2 is a modality-specific extension of R1. It is required only when the
declared mechanical or inverse model uses spatially registered EBSD or other
heterogeneous orientation information. It adds:

* EBSD source, native step, scan origin and scan axes;
* orientation convention, phase information and crystal/sample frames;
* an independently supported physical DIC--EBSD transformation;
* common landmarks or fiducials, transformation parameters and registration
  uncertainty or residual;
* a declared interpolation or material-point assignment rule.

Array correspondence is not physical co-registration. A better mechanical fit
is not registration proof. A homogeneous J2 or homogeneous-orientation SRIX
study therefore has no R2 requirement.

### R3 — Ready for inverse material identification

R3 requires R1, a reproducible inverse-problem contract, and all
model-specific experimental inputs. It requires R2 only when the declared
constitutive or inverse model uses EBSD-resolved heterogeneous information.
The contract should include:

* constitutive implementation/version and parameter convention;
* declared observation used for fitting;
* a meaningful plastic loading range;
* sensitivity and identifiability assessment;
* physically motivated bounds or priors;
* preferably independent reaction/force information and/or multiple
  experiments.

R3 does not mean that all parameters are identifiable. It means that the
experiment is specified well enough for identifiability to be assessed
honestly for the declared model.

## Status vocabulary

Every data item should use exactly one status and cite its source:

```text
KNOWN          directly documented or independently checked
DERIVED        produced by a declared preparation transform
ASSUMED        working hypothesis, not independent proof
MISSING        required but unavailable or not recorded
NOT_APPLICABLE not required for the declared use
```

Requirement classes are separate from data status:

```text
REQUIRED         needed for the declared use
MODEL_DEPENDENT  needed only for a particular law or inverse model
RECOMMENDED      improves validation but does not block the calculation
```

## Reusable blank template

Copy this table for a new specimen and fill the status and source from its
manifest. The `Required for` column is intentionally usage-specific.

| Item | Status | Source | Units / frame | Required for | Notes |
|---|---|---|---|---|---|
| source DIC data |  |  | image pixels | R0 (REQUIRED) |  |
| displacement fields |  |  | canonical components | R1 (REQUIRED) |  |
| DIC scale |  |  | length/pixel | R1 (REQUIRED) |  |
| axes / components |  |  | image/specimen frame | R1 (REQUIRED) |  |
| loading-state order |  |  | ordered states | R1 (REQUIRED) |  |
| physical timestamps / rate |  |  | physical time | MODEL_DEPENDENT | required for rate-dependent laws |
| reaction force |  |  | force units | R3 (RECOMMENDED) | independent mechanical observable |
| geometry / thickness |  |  | length units | MODEL_DEPENDENT | required when used by the forward |
| EBSD source |  |  | crystal/sample frame | R2 (REQUIRED) | only for EBSD-driven use |
| EBSD native step |  |  | length/pixel | R2 (REQUIRED) | never infer from DIC dimensions |
| EBSD axes / origin |  |  | acquisition frame | R2 (REQUIRED) |  |
| orientation convention |  |  | $Q_{global\to material}$ | R2 (REQUIRED) | include phase and handedness |
| DIC--EBSD transform |  |  | specimen frame | R2 (REQUIRED) | numerical assignment is not proof |
| registration residual / uncertainty |  |  | declared metric | R2 (REQUIRED) | independent support required |
| measurement uncertainty / repeat state |  |  | DIC units | R3 (RECOMMENDED) | required for calibrated likelihood claims |

The template is a contract, not a checklist to complete by inference. If an
item is unavailable, retain `MISSING` and state which claims are out of scope.

## P43 worked example

The current P43 inventory records the following statuses:

| Item | Status | Source / known fact | Limitation |
|---|---|---|---|
| raw DIC images | KNOWN | 42 external grayscale TIFF frames | acquisition log and production version absent |
| prepared DIC fields | DERIVED | `U_40`, `V_40`, canonical `3600 x 3100` support | historical processing chain incomplete |
| DIC scale | DERIVED | `0.00184 mm/pixel` | not an EBSD step size |
| DIC axes / components | KNOWN | canonical component contract | row/column convention is declared |
| loading-state order | ASSUMED | reference, monotone sequence and repeated final state | synchronized load timestamps absent |
| physical timestamps | MISSING | manuscript indicates displacement-controlled tension | no synchronized physical time scale |
| reaction force | MISSING | — | no synchronized force history |
| EBSD export | KNOWN | external HDF5 orientation/Schmid arrays | acquisition geometry and native step absent |
| EBSD axes / origin | MISSING | — | global specimen-frame metadata absent |
| DIC--EBSD transform | ASSUMED | declared working $F$ assignment | physical co-registration not independently proven |
| registration residual / uncertainty | MISSING | `registration_proven=false` | correlation tests are not independent proof |
| repeated-state noise | DERIVED | registered repeated-frame source exists | full temporal covariance is not established |

### P43 readiness by use

| Declared use | Status | Interpretation |
|---|---|---|
| R0 archive / inspection | **PARTIAL** | prepared fields are traceable, but raw acquisition/software provenance is not fully versioned |
| R1 rate-independent full-Dirichlet mechanics | **WORKING GO WITH PROVENANCE CAVEATS** | DIC fields, units, axes, support and an ordered working path are available; force and synchronized timestamps are not required for this use |
| R1 rate-dependent physical claim (for example Méric) | **NO-GO FOR PHYSICAL RATE CLAIM** | a physical time/rate scale is missing; numerical replay remains possible as a study |
| R2 EBSD-driven mechanics | **NO-GO FOR EXPERIMENTAL CLAIM** | native EBSD step, axes/origin and independent physical DIC--EBSD registration are absent |
| R3 SRIX/EBSD experimental identification | **NO-GO** | R2 is required for this model and is not satisfied |
| R3 non-EBSD inverse problem | **MODEL-DEPENDENT** | decide from R1 and the declared inverse contract; R2 is not an automatic blocker |

These statuses delimit P43 claims; they do not invalidate the numerical
methods. Registered-case calculations remain methodological uses, not proof of
physical DIC--EBSD co-registration or material calibration.

## Method requirements

| Method / use | R1 | R2 | Time | Force | Uncertainty |
|---|---|---|---|---|---|
| elastic/J2 full-Dirichlet comparison | yes | no | order only | optional | recommended |
| SRIX homogeneous-orientation test | yes | no | order only | optional | recommended |
| SRIX + EBSD | yes | yes | order only | optional | recommended |
| Méric + EBSD | yes | yes | physical time/rate | optional / valuable | recommended |
| DIC-driven plastic reconstruction | yes | no | ordered path | optional | recommended |
| FCC interpretation of reconstructed field | yes | yes | ordered path | optional | recommended |
| statistical parameter-confidence claim | yes | model-dependent | model-dependent | valuable | calibrated uncertainty required |

The table separates mathematical requirements from scientifically valuable
additional evidence. For example, force is not required to solve a
full-Dirichlet displacement problem, while calibrated uncertainty is required
before making a statistical likelihood or confidence claim.

## Go / no-go rules

### Mechanical model comparison

**GO** when the information required by the chosen mechanics is available:
geometry where needed, DIC coordinates and units, and an ordered boundary
history. The interior DIC field remains the observed quantity.

### EBSD-driven mechanics

**GO** only when the mechanical comparison is R1-ready and the R2 metadata and
independent physical co-registration are available.

### Inverse identification

**GO** when R1, a declared inverse contract, all model-specific modalities,
and sensitivity/identifiability assessment are available. Add R2 when the
model uses EBSD-resolved heterogeneity. Otherwise the result is
**methodological use only**.

## Method-specific interpretation

The DIC-driven dissipative reconstruction needs R1 to study

```text
DIC defect -> A^T -> mechanically relevant plastic correction
```

but interpreting that correction in FCC coordinates needs R2. Inferring a
transferable constitutive law from several experiments requires R3. In the
same way, a homogeneous J2 FEMU can be assessed without EBSD, while an
EBSD-driven SRIX claim cannot bypass the registration contract.

The detailed P43 source boundary remains in
{doc}`experimental_data_inventory`; axis, observation and EBSD conventions are
defined in {doc}`dic_axis_conventions`,
{doc}`../scientific/observation_operator` and
{doc}`../scientific/ebsd_orientation_contract`.
