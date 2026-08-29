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
| DIC and observation | complete | partial | supported | actionable workflow added; semantic review still required |
| EBSD and registration | complete | partial | supported | actionable workflow added; registration evidence is provenance-limited |
| J2/Ludwik baseline | complete | partial | verified | actionable workflow added; baseline, not a production crystal law |
| SRIX | complete | partial | supported | expand evidence paths and parameter claim boundary |
| Méric--Cailletaud | complete | partial | supported | distinguish rate physics from integration robustness |
| Structural plane stress | complete | partial | verified | keep standard 2-D MFront distinct from the three-traction contract |
| Spectral solver and FFTW | complete | partial | supported | FFTW performance remains a separate pending claim |
| TET2 discretisation | complete | partial | supported | two-history discretisation is distinct from EBI-TET |
| EBI-TET state sharing | complete | reviewed | negative | falsification is registered-case specific |
| Native SRIX / Numba | complete | partial | supported | retain explicit evidence IDs for each optimization step |
| FEMU and SVD | incomplete | blocked | open | no registered production boundary-only FEMU driver; archived parametric SVD is documented, while full DIC-weighted parametric SVD remains blocked by the unavailable noise payload |
| REGM | complete | partial | negative | observed-DIC transfer remains a no-go |
| Reduced integration | complete | partial | negative | actionable workflow added; CPS4R is not qualified for production plastic campaigns |
| Nonlocal / micromorphic | complete | partial | historical | preserve historical status and limitations |

The authoritative machine-readable fields are in
`docs/_audit/scientific_coverage.yml`; each claim is bound to evidence and a
claim boundary in `docs/_audit/claim_provenance.yml`. In particular, FEMU deliberately has
`how_to.applicable: false`: `scripts/srix_femu_smoke.py` is documented as a
full-field-Dirichlet negative control, not as a boundary-only identification
workflow.

Scientific status is claim-level, not a blanket verdict for a subject. The
`claim_statuses` mapping records, for example, that J2 table equivalence can be
verified while experimental morphology adequacy is negative, or that FFTW
functional equivalence is supported while its full-solver performance remains
open. A reviewed content status also requires an actionable How-to; a route or
an index page alone is not evidence of semantic completeness.

## Review rules

For a subject to become content-reviewed, a maintainer must read the applicable
Explanation, Reference, How-to and Evidence pages and check that they contain
the actual equations, commands/procedure, provenance and claim boundary. The
automated checker verifies routing, mode, status, manifest consistency and
current-to-legacy links; it does not pretend to perform that semantic review.

The historical inversion campaigns are now synthesised canonically in
`explanation/identification/observable_fit_vs_latent_identifiability.md`.
Their source records remain `tensor_local_inverse_results.md`,
`local_coefficient_inverse_results.md`, `tann_fcc_primary_run_results.md` and
`tann_fcc_recovery_strategy.md`; they are not deleted or reclassified as new
production evidence.

| Historical DIC source | Concept retained | Canonical destination |
|---|---|---|
| `docs/explanation/dic_synthetic_measurement_tests.md` | real-texture warp, sinusoidal transfer, localised-band amplitude/width bias | `explanation/measurement/dic_observation_limits.md` |
| `validation/dic_uncertainty_propagation_p0043_results.md` | repeated-frame noise scale and sensitivity interpretation | `explanation/measurement/dic_observation_limits.md` |
| `validation/dic_epsilon_band32_v2/epsilon_band32_metrics.csv` | profile/Charbonnier trade-off | `explanation/measurement/dic_observation_limits.md` |
| `validation/dic_photometric_quality_p0043_results.md` | brightness-residual proxy does not explain the structured mismatch | `explanation/measurement/dic_observation_limits.md` and `explanation/evidence/negative_results.md` |
