# Agent knowledge index

This directory is agent-agnostic. `CLAUDE.md`, `.claude/rules/`, Codex
instructions and other agent entry points should reference these documents
instead of duplicating technical history.

The index is intentionally semantic: follow the row matching the task, then
read only the linked chapter and its current validation artifact. Do not load
the complete documentation tree by default.

Read [`architecture.md`](architecture.md) for durable cross-cutting contracts
before changing code that touches multiple domains.

## Routing table

| If the task mentions | Read first | Evidence to inspect |
|---|---|---|
| FEMU, identification, raw displacement objective | [`docs/explanation/identification/femu_identification.md`](../explanation/identification/femu_identification.md) and [`docs/reference/numerics/femu_sensitivity_and_svd.md`](../reference/numerics/femu_sensitivity_and_svd.md) | relevant `validation/` campaign artifact |
| direct sensitivity, shadow, Jacobian | [`docs/explanation/identification/identifiability.md`](../explanation/identification/identifiability.md) and [`docs/reference/numerics/srix_semismooth_jacobian.md`](../reference/numerics/srix_semismooth_jacobian.md) | current sensitivity qualification artifact |
| SVD, observability, identifiable modes | [`docs/explanation/identification/identifiability.md`](../explanation/identification/identifiability.md) and [`docs/reference/evidence/selection_indicators.md`](../reference/evidence/selection_indicators.md) | current SVD/observability report |
| EBSD, DIC, mapping, registration, axes | [`docs/reference/scientific/ebsd_orientation_contract.md`](../reference/scientific/ebsd_orientation_contract.md), [`docs/reference/data/dic_axis_conventions.md`](../reference/data/dic_axis_conventions.md) and [`docs/adr/0004-dic-input-preparation.md`](../adr/0004-dic-input-preparation.md) | current registration audit |
| SRIX, Méric, slip systems, crystal law | [`docs/explanation/constitutive/forest_rubin_srix.md`](../explanation/constitutive/forest_rubin_srix.md), [`docs/how-to/crystal-plasticity/use_srix_crystal_law.md`](../how-to/crystal-plasticity/use_srix_crystal_law.md) | current constitutive qualification |
| plane stress, sigma33, condensation | [`docs/explanation/constitutive/structural_plane_stress.md`](../explanation/constitutive/structural_plane_stress.md), [`docs/reference/numerics/plane_stress.md`](../reference/numerics/plane_stress.md) | plane-stress validation artifact |
| spectral solver, FFT, DTT, Gélébart, kinematics | [`docs/explanation/spectral_mechanics/index.md`](../explanation/spectral_mechanics/index.md) and [`docs/reference/numerics/spectral_result_contract.md`](../reference/numerics/spectral_result_contract.md) | spectral convergence/performance artifact |
| DIC preparation, units, noise, boundaries | [`docs/how-to/data/prepare_dic_case.md`](../how-to/data/prepare_dic_case.md), [`docs/how-to/data/characterise_dic_measurement_chain.md`](../how-to/data/characterise_dic_measurement_chain.md) | DIC provenance and measurement-chain artifacts |
| nonlocality, micromorphic or spatial interaction | [`docs/explanation/reconstruction/nonlocal_micromorphic.md`](../explanation/reconstruction/nonlocal_micromorphic.md), [`docs/reference/scientific/nonlocal_parameters.md`](../reference/scientific/nonlocal_parameters.md) | current local/nonlocal comparison |
| performance or scaling | [`docs/explanation/native-srix/optimization_strategy.md`](../explanation/native-srix/optimization_strategy.md), [`docs/reference/evidence/native_srix_qualification.md`](../reference/evidence/native_srix_qualification.md) | benchmark artifact with environment fingerprint |
| durable design choice | [`docs/adr/index.md`](../adr/index.md) | the referenced ADR and current code |

## Evidence and history rules

Current code and current committed validation artifacts outrank prose. A
historical report can explain why a choice was made, but cannot establish that
the current checkout still behaves the same way. Mark superseded campaigns as
historical in their artifact; never delete them merely to simplify the index.

If a required fact cannot be proven from the checkout, record it as unknown
and identify the missing provenance. Do not choose a coordinate transform,
mapping, constitutive fallback or optimizer setting solely because it improves
the fit.

## Agent-specific entry points

`.claude/rules/` may add path-scoped instructions for Claude Code. Those files
must remain shortcuts to this index or to canonical documentation. They must
not be the only place where a scientific or numerical rule is recorded.
