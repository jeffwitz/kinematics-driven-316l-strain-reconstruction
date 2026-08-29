# Archived SRIX parametric-field inventory

This inventory was produced by inspecting the archived `.npz` payloads only.
No mechanical forward and no finite-difference sensitivity was run here.

The reports use eight scored states, so a displacement Jacobian with shape
`(7056, 4)` corresponds to `8 × 21 × 21 × 2`; the M100 shape
`(163216, 4)` corresponds to `8 × 101 × 101 × 2`.

| Artefact | Key | Shape | Dtype | Meaning | Parameter point | Usable for full offline DIC weighting |
|---|---|---:|---|---|---|---|
| `p0043_synthetic_identification_v1/fields.npz` | `truth_displacement` | `(32, 21, 21, 2)` | `float64` | truth displacement history | synthetic truth | no, state field only |
| same | `identified_displacement` | `(32, 21, 21, 2)` | `float64` | identified displacement history | fitted synthetic point | no, state field only |
| same | `target_residual` | `(7056,)` | `float64` | scored residual vector | fitted synthetic point | no, residual only |
| same | `jacobian_truth` | `(7056, 4)` | `float64` | displacement Jacobian in the identity-observation synthetic record | synthetic truth | **yes**, transfer can be applied; whitener payload still required |
| `p0043_synthetic_multistart_v1` | `fields.npz` | — | — | no file archived | four synthetic starts are represented in `report.json` | **no**, only SVD summaries are available |
| `p0043_synthetic_scaleup_v1/fields.npz` | `truth_displacement` | `(32, 101, 101, 2)` | `float64` | truth displacement history | synthetic M100 truth | no, state field only |
| same | `initial_displacement` | `(32, 101, 101, 2)` | `float64` | M20-initialised displacement history | initial synthetic point | no, state field only |
| same | `identified_displacement` | `(32, 101, 101, 2)` | `float64` | fitted displacement history | identified synthetic point | no, state field only |
| same | `initial_residual` | `(163216,)` | `float64` | scored residual vector | initial point | no, residual only |
| same | `identified_residual` | `(163216,)` | `float64` | scored residual vector | identified point | no, residual only |
| same | `jacobian` | `(163216, 4)` | `float64` | displacement Jacobian in the identity-observation synthetic record | identified synthetic point | **yes**, transfer can be applied; whitener payload still required |
| `p0043_experimental_srix_m20_v1/fields.npz` | `prior_displacement` | `(32, 21, 21, 2)` | `float64` | prior displacement history | experimental prior | no, state field only |
| same | `best_displacement` | `(32, 21, 21, 2)` | `float64` | best fitted displacement history | experimental best point | no, state field only |
| same | `target_displacement` | `(32, 21, 21, 2)` | `float64` | measured target history | registered experimental data | no, data field only |
| same | `prior_jacobian` | `(7056, 4)` | `float64` | scalar-whitened parametric Jacobian | experimental prior | no, not pre-observation; scalar factor only |
| same | `final_jacobian` | `(7056, 4)` | `float64` | scalar-whitened parametric Jacobian | experimental best point | no, not pre-observation; scalar factor only |
| `p0043_experimental_raw_femu_m20_v1/fields.npz` | `prior_displacement` | `(32, 21, 21, 2)` | `float64` | prior displacement history | experimental prior | no, state field only |
| same | `best_displacement` | `(32, 21, 21, 2)` | `float64` | best fitted displacement history | experimental best point | no, state field only |
| same | `target_displacement` | `(32, 21, 21, 2)` | `float64` | measured target history | registered experimental data | no, data field only |
| same | `prior_jacobian` | `(7056, 4)` | `float64` | raw displacement parametric Jacobian | experimental prior | **yes**, transfer can be applied; whitener payload still required |
| same | `final_jacobian` | `(7056, 4)` | `float64` | raw displacement parametric Jacobian | experimental best point | **yes**, transfer can be applied; whitener payload still required |

## Offline DIC-weighted construction status

The raw displacement Jacobians would be reshaped by scored state and node,
then transformed as

$$
S_{\mathrm{DIC}} = W_D M_D S_u,
$$

where the transfer acts on the model Jacobian only.  The transfer CSV is
present at
`validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv`.
However, the qualified repeated-frame noise source
`validation/reference_data/dic_uncertainty_propagation_p0043_v1/centred_repeat_flow_pixels.npy`
is currently an unhydrated Git LFS pointer (the local file contains only the
pointer metadata).  Therefore the spectral whitener cannot be reconstructed in
this checkout, and the full $S_{\mathrm{DIC}}$ SVD is **blocked**.

The historical experimental “whitened” Jacobian cannot fill this gap: its
whitening is scalar, so it changes singular-value magnitudes but not $V$ or
normalised singular values.  It is a consistency control, not the qualified
spatial DIC weighting requested here.

When the noise payload is available, the required offline inputs are the raw
Jacobian fields above, the transfer CSV, the repeated-frame noise field, the
same support mask and the same state/parameter ordering.  No new mechanical
solve is required.
