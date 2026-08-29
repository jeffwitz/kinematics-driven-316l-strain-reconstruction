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
| same | `jacobian_truth` | `(7056, 4)` | `float64` | displacement Jacobian in the identity-observation synthetic record | synthetic truth | **yes**, processed in the offline report |
| `p0043_synthetic_multistart_v1` | `fields.npz` | — | — | no file archived | four synthetic starts are represented in `report.json` | **no**, only SVD summaries are available |
| `p0043_synthetic_scaleup_v1/fields.npz` | `truth_displacement` | `(32, 101, 101, 2)` | `float64` | truth displacement history | synthetic M100 truth | no, state field only |
| same | `initial_displacement` | `(32, 101, 101, 2)` | `float64` | M20-initialised displacement history | initial synthetic point | no, state field only |
| same | `identified_displacement` | `(32, 101, 101, 2)` | `float64` | fitted displacement history | identified synthetic point | no, state field only |
| same | `initial_residual` | `(163216,)` | `float64` | scored residual vector | initial point | no, residual only |
| same | `identified_residual` | `(163216,)` | `float64` | scored residual vector | identified point | no, residual only |
| same | `jacobian` | `(163216, 4)` | `float64` | displacement Jacobian in the identity-observation synthetic record | identified synthetic point | **yes**, processed in the offline report |
| `p0043_experimental_srix_m20_v1/fields.npz` | `prior_displacement` | `(32, 21, 21, 2)` | `float64` | prior displacement history | experimental prior | no, state field only |
| same | `best_displacement` | `(32, 21, 21, 2)` | `float64` | best fitted displacement history | experimental best point | no, state field only |
| same | `target_displacement` | `(32, 21, 21, 2)` | `float64` | measured target history | registered experimental data | no, data field only |
| same | `prior_jacobian` | `(7056, 4)` | `float64` | scalar-whitened parametric Jacobian | experimental prior | no, not pre-observation; scalar factor only |
| same | `final_jacobian` | `(7056, 4)` | `float64` | scalar-whitened parametric Jacobian | experimental best point | no, not pre-observation; scalar factor only |
| `p0043_experimental_raw_femu_m20_v1/fields.npz` | `prior_displacement` | `(32, 21, 21, 2)` | `float64` | prior displacement history | experimental prior | no, state field only |
| same | `best_displacement` | `(32, 21, 21, 2)` | `float64` | best fitted displacement history | experimental best point | no, state field only |
| same | `target_displacement` | `(32, 21, 21, 2)` | `float64` | measured target history | registered experimental data | no, data field only |
| same | `prior_jacobian` | `(7056, 4)` | `float64` | raw displacement parametric Jacobian | experimental prior | **yes**, processed in the offline report |
| same | `final_jacobian` | `(7056, 4)` | `float64` | raw displacement parametric Jacobian | experimental best point | **yes**, processed in the offline report |

## Offline DIC-weighted construction status

The raw displacement Jacobians would be reshaped by scored state and node,
then transformed as

$$
S_{\mathrm{DIC}} = W_D M_D S_u,
$$

where the transfer acts on the model Jacobian only.  The registered transfer
CSV is present at
`validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv`,
and the repeated-frame noise payload is hydrated for the offline report at
`validation/reference_data/p0043_parametric_dic_weighting_v1/report.json`.
The payload has shape `(3600, 3100, 2)`, dtype `float32`, size 89280128 bytes,
and SHA-256
`82247ccdbcbccfd31b270dac67e3a717b8e7070c701459ee68abdb790665597b`.

The qualified conversion is
`image_flow_to_canonical(noise[:512,:512], pixel_size_mm=0.00184)`, followed
by 256 seeded windows (`seed=42`), no spatial-mean removal, and a one-node
boundary support mask.  No new mechanical solve is required.  The historical
experimental “whitened” Jacobian remains a scalar consistency control: scalar
rescaling changes singular-value magnitudes but not $V$ or normalised singular
values, so it is not a substitute for this spatial chain.

The periodic and wrap-free spectral surrogate SVDs both retain three
directions at the $10^{-3}$ threshold for the experimental M20 final point and
leave the fourth opposite-sign $Q-b$ direction effectively null.  The
wrap-free full variant has absolute singular values
`0.336070, 0.051059, 0.010771, 0.000008782`; complete values and angles are
in `validation/reference_data/p0043_parametric_dic_weighting_v1/report.json`.
Held-out projections of eight-window noise draws have modal standard
deviations `(7.40, 1.76, 1.96, 4.83)`, so the nominal `1/sigma` values are not
calibrated experimental one-sigma thresholds.  Temporal covariance across
the eight scored states is not measured by this final repeated-frame field.
