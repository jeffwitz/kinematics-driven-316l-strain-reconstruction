# SRIX parametric observability evidence

**Mode:** reference  
**Domain:** evidence

This page records what can be read directly from the archived parametric FEMU
reports.  The sensitivity is $S_\theta=\partial r/\partial\theta$ in the log
coordinates $\theta=(\tau_0,R,Q,b)$, with shadow finite-difference step
`h=0.0015`.  It must not be confused with the free tensor-field operator in
{doc}`../../explanation/identification/dic_weighted_tensor_observability`.

| Claim | Evidence ID | Exact artefact | Recorded values | Boundary |
|---|---|---|---|---|
| Synthetic M20 parametric sensitivity is archived | `E-SRIX-PARAM-OBS-001` | `validation/reference_data/p0043_synthetic_identification_v1/report.json` | singular values `14.1914, 1.92613, 0.512540, 0.00146355`; normalised `1, 0.135725, 0.036116, 0.0001031`; condition `9696.6`; $\rho(Q,b)=0.999999997$ | Smoke/machinery evidence; four-parameter recovery not claimed |
| Synthetic multi-start directions are stable | `E-SRIX-PARAM-OBS-001` | `validation/reference_data/p0043_synthetic_multistart_v1/report.json` | four starts; per-start $V$ and SVD archived; final RMS `5.26e-17` to `4.98e-13` | Synthetic P43 M20 only; weak $Q-b$ direction remains |
| Synthetic M100 scale-up preserves the same parameter geometry | `E-SRIX-PARAM-OBS-001` | `validation/reference_data/p0043_synthetic_scaleup_v1/report.json` | normalised `1, 0.416078, 0.055394, 0.0001431`; condition `6989.7`; $\rho(Q,b)=0.9999999999$; three evaluations | One registered M20-initialised scale-up |
| Experimental whitened sensitivity has a weak fourth direction | `E-SRIX-PARAM-OBS-002` | `validation/reference_data/p0043_experimental_srix_m20_v1/report.json` | normalised `1, 0.143072, 0.016464, 0.00001553`; retained rank 3; $Q-b$ direction in discarded vector | Experimental M20 NO-GO; no uniqueness or calibration claim |
| Experimental raw control has the same direction geometry | `E-SRIX-PARAM-OBS-003` | `validation/reference_data/p0043_experimental_raw_femu_m20_v1/report.json` | normalised `1, 0.143084, 0.016465, 0.00001553`; retained rank 3; no whitening/covariance | Raw mismatch control; similarity with scalar-whitened case is expected algebraically |
| Principal-angle comparisons can be computed from archived $V$ | `E-SRIX-PARAM-OBS-004` | the four reports above | synthetic M20/experimental rank-three angle about `0.265°`; synthetic M100/experimental about `0.139°`; experimental whitened/raw below `1e-5°` as expected for scalar rescaling | Algebraic comparison only; no new forward and no independent whitening validation |
| The pre-hydration audit identified the required full-DIC inputs | `E-SRIX-PARAM-OBS-005` | `docs/_audit/srix_parametric_fields_inventory.md` | raw displacement Jacobians and transfer were present; the repeated-frame noise payload was unavailable in that checkout | Historical blocking audit; no scalar substitute was accepted |
| Full DIC-weighted parametric sensitivity is reconstructed offline | `E-SRIX-PARAM-OBS-006` | `validation/reference_data/p0043_parametric_dic_weighting_v1/report.json` | experimental M20 final: normalised `1, 0.050649, 0.007111, 0.000002408`; condition `4.15e5`; rank `2/3/3` at thresholds `1e-2/1e-3/1e-4`; rank-three raw-to-full angle `0.020°` | Offline post-processing only; registered transfer, canonical repeated-frame noise, 256 windows, seed 42; no new forward or finite differences; experimental calibration not claimed |

The synthetic reports use identity observation and no noise.  The experimental
whitened report uses measured displacement with scalar DIC whitening; the raw
report explicitly uses no observation weighting or covariance.  All records
use the registered `316l_srix_transposed_from_nasri2018_rate_1e-3` preset and
32 path steps.

The singular-value energy fractions are approximately:

| Case | $E_1$ | $E_2$ | Modes for 90% / 95% / 99% |
|---|---:|---:|---|
| synthetic M20 | 0.980656 | 0.998721 | 1 / 1 / 2 |
| synthetic M100 | 0.850203 | 0.997391 | 2 / 2 / 2 |
| experimental whitened | 0.979681 | 0.999734 | 1 / 1 / 2 |
| experimental raw | 0.979677 | 0.999734 | 1 / 1 / 2 |

The archived right-singular vectors use the parameter order
`(tau0_mpa, R_mpa, Q_mpa, b)`.  The weak vector is approximately the
opposite-sign $Q/b$ combination, so a low residual does not imply independent
recovery of those two parameters.  The experimental gates explicitly record
`parameters_identified=false` and `m100_authorized=false`.

The archived raw Jacobians are sufficient to form
$S_{\mathrm{DIC}}=W_D M_D S_u$.  The qualified repeated-frame noise field is
now hydrated and used by the offline report.  Its canonical conversion,
registered support mask and sinusoidal transfer are recorded in
`validation/reference_data/p0043_parametric_dic_weighting_v1/report.json`;
no scalar-whitening result is used as a replacement for the spatial DIC chain.

These records support synthetic sensitivity geometry, a registered
experimental NO-GO, and a full offline DIC-weighted sensitivity calculation.
The latter retains three directions at the $10^{-3}$ threshold while leaving
the opposite-sign $Q-b$ direction effectively null.  They do not support
experimental 316L parameter identification or uniqueness.
