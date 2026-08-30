# P43 microstructure-`k_perp` screening

## Preflight status

**E_data_geometry_insufficient** — CP_dataset contains orientation and Schmid fields but no grain ID or segmentation dataset; no P43 microstructural screening was run.

The source `CP_dataset.h5` contains per-pixel grain-mean Euler orientations and Schmid data, but no grain-ID or segmentation dataset. The M20 crop contains 15 unique orientation values, so it cannot supply a defensible grain-boundary geometry.

The HDF5 metadata records `pixel_size_um = 1.84`, but the native EBSD acquisition step is not independently documented. It must not be substituted for the DIC scale.

No grain descriptors, mechanical perturbations, `k_perp` projections, figures or candidate rankings were computed. A provenance-backed grain map is required before resuming this screening.
