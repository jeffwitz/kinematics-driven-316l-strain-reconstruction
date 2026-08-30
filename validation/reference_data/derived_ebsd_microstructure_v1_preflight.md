# Derived EBSD microstructure preflight

> **STOP:** the stored Euler maps do not expose a defensible numerical quantisation gap. No grain segmentation or HDF5 product was generated.

- Source: `/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5`
- Source SHA256: `e2684b5353a53b03871c8ced5ed457c3d2de88de3fb8b7560071bf6d3cda28fb`
- Grid: `[3600, 3100]`
- Orientation datasets: `orientation/phi1`, `orientation/Phi`, `orientation/phi2` (`float64`, degrees, finite)
- Source export: rows `400:4000`, columns `1211:4311` of `[4400, 5400]`; advertised `pixel_size_um = 1.84` remains semantically unresolved
- Exact triplets: `29238`
- Exact neighbour equality: `0.960815`
- Positive neighbour-difference quantiles [deg]: `[0.0018111616373062134, 0.048569679260253906, 0.2758979797363281, 1.34750634431839, 2.954536646604538, 31.536414623260498, 110.34287929534912, 198.0254729092121, 255.00170254707336, 1410.3348503112793]`
- Smallest positive difference [deg]: `0.0018111616`

The non-zero differences form a continuous distribution from approximately `1.8e-3 deg` upward; no empty interval separates numerical quantisation from orientation changes. Exact triplets would fragment the map into boundary/noise components, while an angular tolerance would be an unverified physical rule.

Next action: audit the source export/quantisation or supply an independently justified numerical tolerance. The candidate HDF5 remains ungenerated and is not golden.
