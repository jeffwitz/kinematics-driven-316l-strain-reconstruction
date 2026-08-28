# Semantic documentation review

The Diátaxis infrastructure now distinguishes three dimensions that must not
be collapsed into one label:

* `routing_status`: can a reader reach the applicable page through the
  canonical menu?
* `content_status`: has the content been reviewed, or is it partial/blocked?
* `scientific_status`: what does the evidence permit us to claim?

`complete` is retained as a routing status for compatibility with the coverage
checker. It no longer means that every scientific or operational quadrant is
automatically complete. An inapplicable quadrant must say why; it must not be
filled with a facade page.

| Subject | Routing | Content | Scientific status | Remaining issue |
|---|---|---|---|---|
| DIC and observation | complete | reviewed | supported | continue migrating historical campaign details when needed |
| EBSD and registration | complete | reviewed | supported | registration evidence is provenance-limited |
| J2/Ludwik baseline | complete | reviewed | verified | baseline, not a production crystal law |
| SRIX | complete | partial | supported | expand evidence paths and parameter claim boundary |
| Méric--Cailletaud | complete | partial | supported | distinguish rate physics from integration robustness |
| Structural plane stress | complete | partial | verified | keep standard 2-D MFront distinct from the three-traction contract |
| Spectral solver and FFTW | complete | partial | supported | FFTW performance remains a separate pending claim |
| TET2 / EBI-TET | complete | reviewed | negative | falsification is registered-case specific |
| Native SRIX / Numba | complete | partial | supported | retain explicit evidence IDs for each optimization step |
| FEMU and SVD | incomplete | blocked | open | no registered production boundary-only FEMU driver |
| REGM | complete | partial | negative | observed-DIC transfer remains a no-go |
| Reduced integration | complete | reviewed | negative | CPS4R is not qualified for production plastic campaigns |
| Nonlocal / micromorphic | complete | partial | historical | preserve historical status and limitations |

The authoritative machine-readable fields are in
`docs/_audit/scientific_coverage.yml`. In particular, FEMU deliberately has
`how_to.applicable: false`: `scripts/srix_femu_smoke.py` is documented as a
full-field-Dirichlet negative control, not as a boundary-only identification
workflow.

## Review rules

For a subject to become content-reviewed, a maintainer must read the applicable
Explanation, Reference, How-to and Evidence pages and check that they contain
the actual equations, commands/procedure, provenance and claim boundary. The
automated checker verifies routing, mode, status, manifest consistency and
current-to-legacy links; it does not pretend to perform that semantic review.
