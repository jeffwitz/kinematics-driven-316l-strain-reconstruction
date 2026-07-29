# Experimental data inventory

**Category: Reference.** This page records what experimental evidence is
currently accessible to the repository. It distinguishes archived data,
statements in the supplied manuscript, and information that is not available.

Inventory date: **2026-07-29**.

## Summary

| Item | Available evidence | Status |
|---|---|---|
| DIC steps | 42 raw TIFF images, plus prepared `U_40.npy` and `V_40.npy` | **raw sequence available** |
| Full monotonic history | `000294.tif`--`000335.tif`; acquisition log absent | **available, frame-to-load map provisional** |
| Unloading images | no reverse-branch images found | **not available** |
| Load-cell history \(F(t)\) | figure in the manuscript, no numerical time series | **not available** |
| Image/load synchronisation | no timestamps or synchronisation table found | **not available** |
| Specimen thickness | manuscript states 2 mm | **reported, measurement method unavailable** |
| Gauge width | initial and cropped observation-window dimensions only | **not available** |
| Static image pair | `000334.tif` and `000335.tif` are a candidate repeated final state | **strongly supported, acquisition log absent** |
| Raw speckle images | 42 uncompressed 8-bit TIFF images, \(5400\times4400\) pixels | **available outside the repository** |
| EBSD-derived fields | orientation and Schmid fields found in an external HDF5 file | **available but not versioned** |
| Native EBSD step size | not recorded in the accessible file | **not available** |
| EBSD/DIC registration | arrays are declared co-registered after cropping; method absent | **partially documented** |
| DISFlow parameters | manuscript variational settings plus supplied legacy source with scale 0, patch 4 and stride 1 | **partially reproducible** |

An explicit `not available` means that no supporting file or metadata was
found in the repository, its immediate scientific parent directories, or the
supplied manuscript. It does not prove that the data never existed.

## DIC displacement data

The repository versions four received arrays under
`data/raw/case_study/`. Two are final-state displacement measurements:

| File | Repository convention | Shape | Unit | SHA-256 |
|---|---|---:|---|---|
| `U_40.npy` | \(u_y\) | \(3600\times3100\) | pixel | `f9a308b43db2adc5068f4728d9553011715fd5854664fd5f66b7c9cd035e831f` |
| `V_40.npy` | \(u_x\) | \(3600\times3100\) | pixel | `d7c7725bc7f60f9de97aadc850753343fe6cbe322f5ef02e86c94386b79df2b0` |

The physical sampling used by the reconstruction is
\(1.84\,\mu\mathrm m\) per pixel. The retained field therefore covers
\(6.624\times5.704\,\mathrm{mm}^2\).

The following external directory was supplied on 2026-07-29:

```text
/home/jeff/CNRS/Theses/Adil/essais/9_numerical/DIC_images/
```

It contains 42 distinct uncompressed grayscale TIFF images named
`000294.tif` through `000335.tif`. Every image is \(5400\times4400\) pixels.
The crop `rows[400:4000], columns[1211:4311]` produces the
\(3600\times3100\) support of the prepared fields. The sequence is from the
experiment reported in Qi Hu's thesis, as confirmed by the data owner.

The following mapping is strongly supported by file count and image
registration, but remains **provisional until the acquisition log is found**:

- `000294.tif`: undeformed reference;
- `000295.tif`--`000334.tif`: monotonic steps 1--40;
- `000335.tif`: repeated image at the final state.

The last pair has the same estimated global translation relative to the
reference and a substantially smaller grey-level change than the preceding
loading pair. It is therefore the pre-registered candidate for the null test.
Raw-image timestamps and their load-cell correspondence are not embedded in
the TIFF files.

The supplied numerical README resolves the previous component-name conflict:
`U2` or `U` is the traction-axis displacement and `U1` or `V` the transverse
displacement in the source calculations. This agrees with the versioned
repository convention \(U_40=u_y\), \(V_40=u_x\). Generic `x`/`y` attributes
found in one HDF5 export are not used to override this experiment-specific
mapping.

## Loading history and force

The supplied manuscript reports:

- an Instron 5580 electromechanical test machine;
- displacement-controlled monotonic tension;
- crosshead speed \(1\,\mathrm{mm\,min^{-1}}\);
- approximate engineering strain \(0.25\%\);
- approximate strain rate \(2\times10^{-3}\,\mathrm{s^{-1}}\).

The manuscript contains a plotted macroscopic stress--strain curve and reports
macroscopic parameters derived from it. The numerical samples, load-cell
signal, acquisition frequency and image/load synchronisation are not present
in the accessible files.

The test described in the manuscript is monotonic. No unloading branch is
reported or archived.

## Geometry

| Quantity | Value | Evidence level |
|---|---:|---|
| specimen thickness | 2 mm | manuscript statement |
| initial observation window | \(7\times10\,\mathrm{mm}^2\) | manuscript statement |
| cropped reconstruction window | \(6.624\times5.704\,\mathrm{mm}^2\) | manuscript and array geometry |
| gauge width | not available | no accessible geometry file |
| thickness measurement method | not available | referred to external publication |
| proof that a complete traction section lies inside the crop | not available | cannot be inferred from ROI dimensions alone |

The 2 mm value is suitable as reported specimen geometry. It does not establish
the thickness assigned to the historical Abaqus CPS4 section, because the
original input file is unavailable.

## DISFlow production settings

Section 2.2 of the supplied manuscript gives the reported
variational-refinement parameter set:

| Parameter | Value | Reported role |
|---|---:|---|
| \(\alpha\) | 100 | spatial smoothness regularisation |
| \(\delta\) | 1 | brightness-constancy weight |
| \(\gamma\) | 0 | gradient-constancy weight |
| Charbonnier \(\epsilon\) | 0.002 | robust-penalty regime |
| gradient-descent iterations | 30 maximum | iterative optical-flow solve |

The supplied source
`references/legacy_dic/dic_displacement_fields.py` adds:

| Setting | Historical source |
|---|---:|
| factory | `cv2.DISOpticalFlow_create()` without an argument |
| finest scale | 0 |
| patch size | 4 px |
| patch stride | 1 px |

The source does **not** set gradient-descent iterations, mean normalisation or
spatial propagation. Under OpenCV 4.14, the no-argument object returns 16,
enabled and enabled for these three getters after the explicit historical
setters are applied. Those are current-library defaults, not certified values
of the historical executable.

The original OpenCV version, effective no-argument preset and factory values
used during production are not archived. The historical mask file is also
absent. Consequently, the production chain cannot be reproduced bit for bit.
This absence does not block new all-valid-window metrology or symmetric P43
replay, provided the declared boolean mask and its hash are recorded and no
historical-mask claim is made.

The new raw sequence makes two declared **reproduction implementations**
possible: `legacy_script_2021`, based on the supplied source, and
`declared_medium_v4`, the fully explicit earlier metrology profile. Their
OpenCV version, requested settings and every queryable value are written into
each validation report. Neither is described as the exact historical
production chain.

## EBSD-derived and topography data

One external, unversioned file was found:

```text
/home/jeff/CNRS/Theses/Adil/essais/CP_dataset.h5
```

Its size is 179,379,410 bytes and its SHA-256 is
`e2684b5353a53b03871c8ced5ed457c3d2de88de3fb8b7560071bf6d3cda28fb`.
It declares the following datasets:

| Dataset | Shape | Declared support |
|---|---:|---|
| `orientation/phi1`, `Phi`, `phi2` | \(3600\times3100\) | grain-mean Euler angles, cropped and co-registered |
| `schmid/max_schmid_factor` | \(3600\times3100\) | grain-mean maximum Schmid factor, co-registered |
| `displacement/U`, `V` | \(3600\times3100\) | step-40 displacement |
| `topography/height_after_test` | \(3855\times5613\) | separate grid |

The orientation and Schmid datasets declare a crop from rows `400:4000` and
columns `1211:4311` of \(4400\times5400\) source arrays, then use the
\(1.84\,\mu\mathrm m\) DIC grid. This is a resampled/exported spacing; the
native EBSD acquisition step is not recorded.

The method and uncertainty of EBSD-to-DIC registration are not recorded.
The topography dataset has unknown units, unknown pixel size, a distinct grid
and 1,052,123 non-finite pixels; it must not be co-registered by assumption.

The orientation/Schmid export also contains 60 clearly out-of-range values
and six pixels where all exported orientation and Schmid values are zero.
Their physical meaning and masking policy are not documented. They must be
resolved before computing a microstructural correlation length.

## Unloading decision

**Decision:** an unloading/reloading branch should be acquired with the same
optical and load-cell setup if that setup can be recovered. It is required
before identifying or discriminating kinematic hardening from a macroscopic
reverse branch.

Until such data are supplied:

- KD-064 remains blocked;
- no Bauschinger amplitude is claimed;
- local decreases of an equivalent-strain invariant may only identify
  candidate non-proportional or unloading regions;
- they are not treated as evidence of stress reversal.

If an existing numerical load record is later recovered without images, it
can support a macroscopic reverse-force analysis but not a local DIC
unloading map.

## Provenance

The main sources used by this inventory are:

1. `ArticleSource/ArticleAdil.pdf`, SHA-256
   `482abeee48d7e8b6791bdcc308cb14a84fc8988e8360c2095d31437822aeb7b8`;
2. `data/raw/case_study/manifest.json`;
3. `data/raw/case_study/README.md`;
4. `docs/reference/input_contract.md`;
5. `references/legacy_abaqus/Case5.py`;
6. the external HDF5 file identified above;
7. `/home/jeff/CNRS/Theses/Adil/essais/9_numerical/README_EN.txt`;
8. the external TIFF sequence, with endpoint hashes:
   `000294.tif` =
   `9a4cf912fcc7f989072e1acc075ae40e02f782faf14c4c70535d48daf52282da`,
   `000334.tif` =
   `04f38d166cc186403b85424ccb14f15621c532fef0f1338e6bd6207ba8b8d830`,
   and `000335.tif` =
   `2d086df7381856e6d6252721b2a0451ab020bd0d496718b9011d25856917519d`.

This page records availability only. It does not certify the scientific
correctness of any external dataset.
