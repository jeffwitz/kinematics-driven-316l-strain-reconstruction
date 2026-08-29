# Global documentation cold review

## Scope and method

This is a cold-reader audit of the documentation as a project in numerical and
experimental mechanics, rather than an audit of the identification story alone.
The first pass followed the public entry points in `docs/index.rst`, then the
Explanation, Tutorials, How-to, Reference and Evidence portals.  Representative
canonical pages were read in each domain, including the spectral index and its
solver pipeline, the DIC/EBSD contracts, the constitutive pages, the native SRIX
contract and the current user guides.  This is a comprehension audit, not a
request to make every page exhaustive.

Scores use the following scale:

* **0** — absent or incoherent;
* **1** — fragmentary;
* **2** — usable, with significant gaps;
* **3** — solid for the intended scope.

## Domain scores

### A. Experimental / measurement

| Dimension | Score |
|---|---:|
| Scientific narrative | 2 |
| Technical completeness | 2 |
| Evidence / provenance | 2 |
| Usability / navigation | 2 |

Remarks:

1. `dic_observation_limits` now gives the right physical chain (displacement,
   texture, DIC, differentiation), including transfer attenuation, noise and
   the MTF-scale example.  It correctly prevents a DIC EVM map from being read
   as a plasticity or stress measurement.
2. The observation, axis, EBSD orientation and data-inventory contracts are
   clear and identify important missing experimental metadata.  EBSD
   registration is nevertheless a provenance-limited qualification, not a
   demonstrated experimental fact.
3. The preparation and registration How-to guides are executable, but the
   measurement-to-mechanics route still requires several short pages to be
   read together.  There is no first-contact DIC/EBSD tutorial.

### B. Reconstruction mechanics

| Dimension | Score |
|---|---:|
| Scientific narrative | 2 |
| Technical completeness | 2 |
| Evidence / provenance | 2 |
| Usability / navigation | 2 |

Remarks:

1. The boundary/interior distinction is stated clearly: measured boundary
   kinematics drive a mechanically admissible interior; interior fields remain
   predictions.  The local baseline and historical spatial-interaction path
   provide the needed context.
2. Full-Dirichlet lifting and the spectral equilibrium route are substantially
   documented, but the local and coupled reconstruction How-tos are less
   self-contained than the spectral tutorial.
3. The first reconstruction tutorial is a useful executable entry point, yet
   the J2 and historical nonlocal branches are easier to reach than a complete
   current constitutive workflow.

### C. Constitutive modelling

| Dimension | Score |
|---|---:|
| Scientific narrative | 2 |
| Technical completeness | 2 |
| Evidence / provenance | 2 |
| Usability / navigation | 2 |

Remarks:

1. SRIX, structural plane stress and the MFront/native relationship now have a
   coherent current story.  J2/Ludwik is correctly presented as a baseline,
   and Méric as a comparison branch rather than an interchangeable production
   law.
2. The constitutive pages are scientifically useful but uneven in depth:
   Méric parameter/state details and the standalone execution path are thinner
   than the SRIX explanation and the plane-stress reference.
3. The current routes are easy to find from Explanation and Reference, but a
   reader wanting to use crystal plasticity without the identification context
   still has to assemble the workflow from How-to, contract and qualification
   pages.

### D. Spectral solver / numerical mechanics

| Dimension | Score |
|---|---:|
| Scientific narrative | 3 |
| Technical completeness | 3 |
| Evidence / provenance | 2 |
| Usability / navigation | 2 |

Remarks:

1. This is the strongest non-identification domain.  The spectral index and
   solver pipeline explain full-Dirichlet lifting, true residual/Jacobian,
   Newton--GMRES and the scope of the registered problem.
2. The documentation explicitly separates the DST-I/B0 preconditioner from
   the nonlinear residual, and distinguishes one-point, TET2 and EBI-TET.  TET2
   is not conflated with the registered negative EBI state-sharing result.
3. The scientific material is hidden behind one `spectral_mechanics/index`
   entry in the Explanation menu, while the spectral How-to pages remain more
   concise and less operational than the underlying contracts.

### E. Native SRIX / performance architecture

| Dimension | Score |
|---|---:|
| Scientific narrative | 2 |
| Technical completeness | 2 |
| Evidence / provenance | 2 |
| Usability / navigation | 2 |

Remarks:

1. The optimization ladder (MFront oracle, native NumPy/Numba, coupled closure,
   fused/adaptive paths) explains why the backend exists independently of
   FEMU.  Machine dependence and the future GPU path are appropriately bounded.
2. The native contract is concrete about state, transaction, response level and
   plane-stress options; stepwise qualification evidence is still less visible
   than the architecture narrative.
3. A user can reach the backend pages, but the current run/qualification guide
   is primarily a protocol description rather than a fully reproducible first
   native-SRIX experience.

### F. Identification

| Dimension | Score |
|---|---:|
| Scientific narrative | 3 |
| Technical completeness | 3 |
| Evidence / provenance | 3 |
| Usability / navigation | 2 |

Remarks:

1. This is the most thoroughly reviewed narrative: field reconstruction,
   field observability, parametric sensitivity, SVD, FEMU limits and the
   distinction between demonstrated, negative and open claims are explicit.
2. The evidence routes preserve the important boundaries: free-field inverse
   non-uniqueness, compact synthetic recovery, unqualified TANN, REGM transfer
   failure and the blocked production boundary-only FEMU.
3. The domain is easy to find, but it occupies many visible Explanation links;
   its polished presentation can make it appear to be the project's only
   mature scientific story.

### G. Software / user documentation

| Dimension | Score |
|---|---:|
| Scientific narrative | 2 |
| Technical completeness | 2 |
| Evidence / provenance | 2 |
| Usability / navigation | 1 |

Remarks:

1. Installation, data preparation, reconstruction, reproduction and extension
   categories are present and the canonical routes are now coherent.
2. The three Tutorials are useful but cover only first reconstruction, a local
   versus coupled comparison, and the full-Dirichlet spectral path.  They do
   not provide a first DIC/EBSD or first SRIX/plane-stress experience.
3. Several How-to pages contain exact commands (for example case preparation,
   local reconstruction, EBSD inspection and EBI reproduction), while others
   remain procedural summaries.  API/CLI/configuration Reference pages define
   contracts but give few concrete examples.  This is the largest usability
   weakness outside the scientific narrative.

## Is the documentation FEMU-biased?

The answer is **yes in presentation, but not in the underlying scientific
coverage**.

### Content bias

Identification has the most polished cross-page argument and the most explicit
claim boundaries.  That is a real content asymmetry.  It does not mean that the
repository is scientifically a FEMU project: the spectral mechanics pipeline,
constitutive contracts, DIC measurement model and native SRIX architecture are
substantive bodies of work.  They have simply received less global narrative
review.

### Navigation bias

The Explanation index exposes many identification pages individually, whereas
the entire spectral mechanics story is presented through one visible index
entry.  This makes the FEMU area look larger than it is and makes the spectral
work look smaller than it is.  The home page itself is balanced and names the
full current stack before identification.

### Historical bias

Recent observability/SRIX pages have precise current-status language.  Older
reconstruction, constitutive and user-workflow material is more often a concise
canonical facade over a richer historical report.  This is a maturity gradient,
not evidence that the older scientific work is unimportant.

No navigation correction is applied by this audit.

## What the existing `partial` statuses mean

The matrix is correctly conservative, but `partial` has two distinct causes:

| Subjects | Interpretation |
|---|---|
| DIC, J2, plane stress, TET2 | **Not fully audited as complete cross-quadrant documentation**; the core scientific material is present, but the user/evidence routes need the same review standard as FEMU. |
| EBSD | **Actual provenance limitation plus incomplete user onboarding**: registration is documented, but independently verified registration metadata are not established. |
| SRIX, native SRIX | **Actual evidence/actionability gap**: the current model and contract are strong, while stepwise qualification and an end-to-end user route remain thinner. |
| Méric | **Actual content gap** in the standalone reference and operational workflow, especially around parameters, state and reproducible comparison. |
| Spectral solver | **Mostly review and usability gap**, with the remaining scientific boundary being FFTW performance rather than the solver formulation itself. |
| REGM, reduced integration | **Actual How-to/evidence-depth gap**; the interpretation is present, but the operational and quantitative routes are not as autonomous as the explanation. |
| Nonlocal / micromorphic | **Intentional historical status**, not a missing current production claim; it remains important as a scientific branch and baseline. |

## Entry-path check

### Scientific path

`docs/index.rst` reaches Explanation, whose reconstruction, measurement,
constitutive and identification routes are coherent.  The spectral path is
scientifically complete at the index level, but its depth is one click further
away than the many identification pages.  Evidence routes correctly lead to
registry, interpretation and reproduction without replacing the scientific
narrative with commands.

### User path

`docs/index.rst` → Tutorials → How-to → Reference is stable and free of the
previous legacy-routing problem.  The first reconstruction and spectral
tutorials are genuinely actionable.  The gap is breadth and consistency: there
is no first DIC/EBSD or SRIX/plane-stress tutorial, and some How-to pages still
state a procedure without a complete command, artifact and verification block.

### Reproducibility path

Evidence → registry → reproduce is discoverable, and the EBI/TET2 route states
commands, thresholds and claim boundaries.  Other qualification routes are
more uneven, especially native SRIX, Méric and some historical/negative
campaigns.  This is a usability/provenance gap rather than a broken portal.

## Tutorial assessment

The three existing tutorials form a reasonable minimal set for reconstruction,
coupled comparison and spectral mechanics.  They do represent real project
capabilities, but they do not yet represent the full user-facing stack.  A
future tutorial for DIC/EBSD preparation or for the SRIX/plane-stress path may
be justified; this audit does not create one or rank it above the operational
How-to gaps.

## Short domain summary

| Domain | Status | Priority |
|---|---|---|
| Measurement | usable but uneven | medium |
| Reconstruction | coherent, operational depth uneven | medium |
| Constitutive modelling | scientifically coherent, needs standalone depth | high |
| Spectral mechanics | strongest technical narrative, under-exposed in navigation | high |
| Native backend | coherent architecture, qualification route thin | medium |
| Identification | most mature narrative, currently over-visible | low for documentation architecture |
| Software / user docs | structurally sound, least actionable | high |

## Global verdict

**B — structurally sound but scientifically uneven.**

The documentation now represents the major scientific components and keeps the
Diátaxis routes separate.  It is not yet uniformly re-read at the same depth:
the identification story is mature, while measurement onboarding,
standalone constitutive use, spectral visibility and executable user/reproduce
guides remain less even.

## Top five priorities

1. Make the core non-identification How-to/reproduce pages consistently
   actionable, with exact commands, inputs, artifacts, verification and claim
   boundaries.
2. Give the spectral mechanics path comparable first-class visibility to the
   identification path, while keeping its existing scientific index intact.
3. Complete a standalone constitutive review for SRIX, Méric, J2 and structural
   plane stress, including use outside FEMU.
4. Strengthen the measurement/data route (DIC, EBSD registration and
   provenance) and decide whether a first-contact DIC/EBSD tutorial is needed.
5. Expose the native SRIX/MFront qualification and performance path as a
   reproducible user route, with explicit evidence artefacts and current GPU
   boundaries.

This file is an audit record only.  No canonical page, manifest status,
navigation tree or scientific claim was changed as part of this review.
