# Cold-reader review of the scientific big picture

This is an internal audit, not a new scientific page. It starts from
`docs/index.rst` and
`docs/explanation/identification/identification_strategy_big_picture.md`, then
follows only the links needed to answer the reader's questions. The criterion
is causal comprehension, not exhaustive coverage of secondary thresholds or
implementation details.

## Reader questions

| Question | Answer available | Where the answer is found | Problem, if any |
|---|---|---|---|
| 1. What experimental data do we have? | yes | `docs/index.rst`; big picture §1; `explanation/measurement/dic_observation_limits.md`; `explanation/measurement/ebsd_registration_and_orientation.md` | The inventory/provenance details correctly remain outside the overview. |
| 2. What is imposed and what is predicted? | yes | big picture §2; `explanation/reconstruction/from_dic_to_mechanics.md` | The lifting/predicted-interior distinction is explicit. |
| 3. Why is J2/Ludwik insufficient as the final model? | partial | big picture §2; `explanation/reconstruction/local_baseline.md`; `explanation/constitutive/ludwik_j2.md` | The overview states the limitation, but the concrete localisation mismatch requires one follow-up click. |
| 4. Why is SRIX the principal model? | yes | big picture §2; `explanation/constitutive/forest_rubin_srix.md` | Physical rationale and production status are clear. |
| 5. What are the roles of EBSD, plane stress, MFront and native SRIX? | yes | big picture §2; `explanation/measurement/ebsd_registration_and_orientation.md`; `explanation/constitutive/structural_plane_stress.md`; `reference/numerics/native_srix_backend.md` | The roles are separated rather than presented as competing laws. |
| 6. Why is the full temporal sequence necessary? | yes | `explanation/reconstruction/temporal_loading_path.md` | Same endpoint versus same internal state is stated and supported by P43 results. |
| 7. Why must predictions pass through the DIC operator? | yes | big picture §1/§4; `explanation/measurement/dic_observation_limits.md` | Transfer, amplitude bias and noise are explained at the right level. |
| 8. What is the difference between field reconstruction, field observability and parameter identification? | yes | big picture §3; `explanation/identification/dic_weighted_tensor_observability.md`; `explanation/identification/observable_fit_vs_latent_identifiability.md` | The three questions are explicitly separated. |
| 9. Why does an excellent kinematic fit not suffice? | yes | big picture §3; `explanation/identification/observable_fit_vs_latent_identifiability.md` | The free-tensor nullspace and constrained `q=1` counterexample make the point. |
| 10. Why use SVD? | yes | big picture §4; `explanation/identification/srix_parametric_observability.md` | `V`, weak modes and supported subspaces are explained without overclaiming. |
| 11. Which parameter combinations appear observable? | yes | big picture §4; `reference/evidence/srix_parametric_observability.md` | The registered `tau0/R`, `Q+b` and weak `Q-b` interpretation is bounded to the tested case. |
| 12. What is demonstrated, negative and open? | yes | big picture §7; `explanation/evidence/negative_results.md`; `evidence/index.md` | The categories are visible and do not turn negative results into competing projects. |
| 13. What next experiment/calculation would add information? | yes | big picture §7; `reference/evidence/srix_parametric_observability.md`; `explanation/reconstruction/temporal_loading_path.md` | The required next ingredients are listed without launching a new campaign. |

## Main comprehension breaks (maximum five)

1. **Concrete J2 motivation requires a follow-up click.** The overview says
   that J2 is a baseline and has limits, but does not give the one-sentence
   localisation mismatch that motivated crystal plasticity. This is acceptable
   for a short overview, but the link to `local_baseline` is the essential next
   click.
2. **The performance rationale is qualitatively stated.** Native SRIX and the
   matrix-free solver are said to make repeated forwards practical, but the
   overview does not give even an order-of-magnitude indication of the cost
   reduction. Adding benchmark numbers would risk turning the compass into a
   performance report; keep the detail in the optimisation pages unless a
   reader test shows this gap is decisive.
3. **The immediate next action is intentionally a two-part open item.** The
   page names both the qualified DIC whitener and a boundary-only FEMU workflow.
   It does not rank them, because the former is an offline data-availability
   blocker and the latter is a missing production workflow. This is a real
   open-status nuance, not a documentation error.

No other rupture of the causal story was found. In particular, the roles of
SRIX versus MFront/native, the temporal-path argument, the DIC observation
operator, the three inverse questions, the SVD interpretation and the negative
results are understandable without opening the legacy documentation.

## Deliberately not promoted into the overview

The following remain correctly detailed elsewhere: exact DISFlow settings,
full FSS/variogram protocols, individual evidence IDs, backend implementation
options, solver cutback diagnostics, and secondary thresholds. They are useful
for reproduction or qualification, but do not repair a break in the scientific
line of reasoning.

## Result of this pass

Overall comprehension is **good**. The big picture is sufficient to explain:

```text
DIC + EBSD + history
        ↓
boundary-driven mechanical forward
        ↓
SRIX crystal plasticity
        ↓
predicted displacement
        ↓
DIC observation operator
        ↓
comparison and sensitivities
        ↓
SVD and supported parameter combinations
```

This pass made no further canonical-page edits and introduced no new
scientific claim, evidence record or calculation. The three items above are a
normal, bounded follow-up backlog rather than a reason for another
documentation restructuring phase.

