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

`SHA256SUMS` covers every file here.
