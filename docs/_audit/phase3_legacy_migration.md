# Phase 3 legacy migration audit

This audit records the legacy routes removed from the scientific coverage
matrix during the canonical-reference migration.  `routed` is retained when
the subject still needs a dedicated tutorial, how-to, explanation or evidence
review; it is not upgraded merely because a path exists.

| Subject | Status before | Legacy blocker | Canonical replacement | Status after |
|---|---|---|---|---|
| DIC and observation | routed | `reference/observation_operator.md`, `reference/evidence_registry.md` | `reference/scientific/observation_operator.md`, `reference/evidence/evidence_registry.md` | routed |
| EBSD and registration | routed | `reference/ebsd_orientation_contract.md`, `reference/evidence_registry.md` | `reference/scientific/ebsd_orientation_contract.md`, `reference/evidence/evidence_registry.md` | routed |
| J2/Ludwik baseline | routed | `reference/constitutive_models.md` | `reference/scientific/constitutive_models.md` | routed |
| SRIX | routed | `reference/evidence_registry.md` | `reference/evidence/evidence_registry.md` | routed |
| Méric--Cailletaud | routed | `reference/constitutive_models.md` | `reference/scientific/constitutive_models.md` | routed |
| Structural plane stress | routed | `reference/evidence_registry.md` | `reference/evidence/evidence_registry.md` | routed |
| Spectral solver / FFTW | routed | `reference/evidence_registry.md` | `reference/evidence/evidence_registry.md` | routed |
| Native SRIX / Numba | routed | `reference/evidence_registry.md` | `reference/evidence/evidence_registry.md` | routed |
| FEMU and SVD | routed | `reference/selection_indicators.md` | `reference/evidence/selection_indicators.md` | routed |
| REGM | routed | `reference/claims_matrix.md` | `reference/evidence/claims_matrix.md` | routed |
| Reduced integration | routed | `reference/constitutive_models.md` | `reference/scientific/constitutive_models.md` | routed |

The two subjects already marked `complete` were not downgraded: their routes
were already canonical and reachable.  The strict checker is the authority
for future status changes and rejects historical, legacy or unreachable pages
for `complete` entries.
