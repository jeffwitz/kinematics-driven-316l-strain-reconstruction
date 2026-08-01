# Negative-control observed EVM fields, partition 43

The two negative controls of the observed-EVM candidate comparison, archived
2026-07-31.

They were **not** archived when the first comparison ran: the campaign read
them from a session scratchpad, so the control-based conclusions of
`observed_evm_candidate_comparison_results.md` were not reproducible. The
scratchpad was still intact when the defect was found and the fields recovered
from it; the two `fem_observed_evm.npy` reproduce the SHA-256 recorded in that
campaign's `report.json` exactly, so these are the fields it used, not
regenerated substitutes.

| Control | `fem_observed_evm.npy` SHA-256 | Matches v1 |
|---|---|---|
| homogeneous | `ca4044fe…7581df58` | yes |
| translated | `1cd91dbb…c3798acefe59` | yes |

`homogeneous` gives every grain the same material parameters; it must be
rejected, and the registered E2 rejects it. `translated` displaces the material
maps while preserving their distributions, so the microstructure is right and
its placement is wrong; it must be rejected, and the v1 criteria set failed to
reject it. That failure is what motivated
`observed_evm_morphology_criteria_preregistration.md`.

`fem_displacement_image_grid.npy` is kept so the controls can be re-observed
through a different DISFlow profile without rerunning mechanics — the second
profile of the blind confirmation needs exactly that.

## The `declared_medium_v4` re-observations

`homogeneous_declared_medium_v4/` and `translated_declared_medium_v4/` were
produced 2026-08-01 for the blind confirmation of the v2 criteria set. The four
models were already archived under that profile; the controls were not.

They come from `replay_dic_observation` on the archived control campaigns
(`results/control-{homogeneous,translated-maps}-local-p0043-pad150`), patch 8
stride 3, prepared case `data/processed/case_study`. **No mechanics was rerun**:
the replay checks the source `U.npy` against the immutable campaign status
before observing it.

Note that `dic_evm.npy` is byte-identical across profiles, SHA-256
`f8cde6b0…`. The DIC EVM is reconstructed from the measured displacements of
the prepared case and never passes through DISFlow, so only the FEM observation
depends on the profile.

`SHA256SUMS` covers every file here.
