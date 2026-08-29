# P43 experimental-data inventory

**Mode:** reference  
**Domain:** data

This inventory is the readable provenance boundary for the P43 experiment. It
distinguishes bytes received by the project, prepared or derived fields,
working hypotheses and facts that remain unavailable. An unavailable item is
not inferred from a field fit.

## Provenance status at a glance

| item | status | known | not known or not proven |
|---|---|---|---|
| raw DIC images | **received** | 42 grayscale TIFF frames, supplied outside the repository | acquisition log and exact production software version |
| `U_40`, `V_40` fields | **derived / prepared** | canonical support `3600 x 3100`, P43 step-40 fields, hashes recorded in the input manifest | exact reconstruction history of the historical processing chain |
| frame ordering | **provisional / working hypothesis** | reference frame, monotone sequence and repeated final frame are identified by the supplied inventory | synchronized load-cell timestamps |
| physical load history | **unavailable** | manuscript reports displacement-controlled monotone tension | numerical force/time series and image/load synchronization |
| DIC scale and crop | **derived / prepared** | `0.00184 mm/pixel`, crop and canonical axes are recorded | this is not an EBSD native step size |
| EBSD orientation export | **received, external** | orientation and Schmid datasets exist in an HDF5 export on the supplied workstation | versioned repository payload, acquisition geometry and native step size |
| EBSD axes and global geometry | **unavailable / not proven** | array dimensions are known where exported | independent axis metadata and specimen-frame geometry |
| DIC--EBSD correspondence | **provisional** | a declared index mapping is usable for registered-case calculations | independent physical co-registration proof |
| post-test topography | **received, external** | a separate height grid is listed in the HDF5 export | units, scale and registration to DIC/EBSD |
| repeated-frame noise | **received / derived** | a registered source exists for measurement-sensitivity work | complete temporal covariance of the scored sequence |

The registration qualification report records
`fullfield_analysis_completed=true`, but also
`ebsd_global_geometry_known=false`, `ebsd_axis_metadata_found=false` and
`registration_proven=false`. It also records that no FEM solve or material
identification was used in that data-only analysis.

## DIC data and transformations

The versioned preparation contains the final displacement fields:

| field | canonical component | support | scale |
|---|---|---:|---:|
| `U_40.npy` | $u_y$ (tensile direction) | `3600 x 3100` | pixel values converted with `0.00184 mm/pixel` |
| `V_40.npy` | $u_x$ (transverse direction) | `3600 x 3100` | pixel values converted with `0.00184 mm/pixel` |

The raw image sequence supplied with the case contains 42 uncompressed
`5400 x 4400` TIFF images. The declared crop produces the prepared support,
but the acquisition log and exact image-to-load timestamps are absent. The
mapping of a reference frame, 40 monotone frames and a repeated final frame is
therefore a documented working correspondence, not a proven load chronology.

The physical DIC sampling scale and the native EBSD acquisition step are
separate facts. The former is recorded as `0.00184 mm/pixel`; the latter is not
recorded in the accessible export. No EBSD spacing is inferred from the DIC
array dimensions.

## EBSD and registration boundary

The external HDF5 export contains grain-mean orientation/Schmid fields on the
same exported support as the prepared DIC fields, plus a separate post-test
topography grid. The export declares the arrays as cropped/co-registered, but
does not provide the acquisition origin, physical axes, native step size or a
registration uncertainty. The declaration is consequently a working input
contract, not independent experimental proof.

The data-only registration report tested several spatial and traction-axis
hypotheses and produced correlation and permutation indicators. Those tests
show that hypotheses can be evaluated; they do not license selecting a mapping
because it gives the best mechanical correlation and then calling that mapping
validated. Such a choice would be circular.

## What the statuses permit

* **Received** data may be cited with its external/repository provenance.
* **Prepared** fields may be used when the preparation transform, axes, units,
  crop and hash are recorded.
* **Provisional** mappings may support explicitly labelled registered-case
  calculations and sensitivity studies.
* **Unavailable / not proven** metadata must remain visible and cannot be
  filled by an inferred rotation, scale or best-fit registration.

Repeated-frame data provide a registered source for spatial measurement
sensitivity, but they do not establish the full temporal covariance of the
eight-state DIC sequence.

## Provenance sources

The detailed hashes and preparation steps are maintained in
`data/raw/case_study/manifest.json`, `data/raw/case_study/README.md`, the input
contract and the registered P43 validation reports. The external EBSD HDF5,
raw TIFF sequence and manuscript are provenance inputs, not versioned project
artifacts. This page records availability and status; it does not certify the
scientific correctness of an external dataset.
