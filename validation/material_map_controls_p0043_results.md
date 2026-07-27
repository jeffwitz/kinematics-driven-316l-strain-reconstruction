# P43 material-map control results

Date: 2026-07-27

The mapped local reference, homogeneous nominal control and translated-map
control all converged for 20 increments without cutback. The primary
machine-readable result is
`validation/reference_data/material_map_controls_p0043_v1/report.json`.

## Primary EVM comparison

| Case | Correlation | Relative L2 | RMSE | Top-10% IoU | Absolute DIC-q90 IoU | Predicted q90-active area |
|---|---:|---:|---:|---:|---:|---:|
| mapped local | 0.3791 | 0.9516 | 0.003988 | 0.2073 | 0.2042 | 17.80% |
| homogeneous | 0.4197 | 0.3506 | 0.001469 | 0.0989 | 0.0000 | 0.00% |
| translated maps | 0.1402 | 0.7442 | 0.003119 | 0.1131 | 0.1258 | 17.69% |

No Helmholtz post-filter was applied.

## Interpretation

The homogeneous model gives the lowest global error and even a slightly
higher correlation, but it predicts no point above the DIC q90 threshold.
Its apparent global success comes from reproducing the low-amplitude
background while suppressing the experimentally active bands. Global L2 and
correlation are therefore insufficient objectives for this problem.

Translating the maps reduces correlation from 0.379 to 0.140 and top-10% IoU
from 0.207 to 0.113 while leaving their distributions and pixel-wise
`sigma_y`/`K` pairing unchanged. The original spatial placement of the maps
therefore contains real information about localisation. This is stronger
evidence than a comparison against the homogeneous case alone.

The mapped model nevertheless overpredicts peak amplitude and active area.
The controls support two simultaneous conclusions:

1. boundary kinematics alone explain much of the smooth global EVM field;
2. the spatial maps add localisation information, but their current
   amplitudes or constitutive interpretation are not yet predictive.

PEEQ remains a model output. Its amplitude is not compared with DIC.

## Computational record

| Control | Wall time | Newton iterations | Cutbacks | Maximum plane-stress residual |
|---|---:|---:|---:|---:|
| homogeneous | 674 s | 92 | 0 | 3.61e-14 MPa |
| translated maps | 854 s | 117 | 0 | 2.07e-13 MPa |

The mapped reference used 129 Newton iterations and 939 s in its archived
campaign. Timing differences are descriptive only because the material fields
change the nonlinear trajectory.
