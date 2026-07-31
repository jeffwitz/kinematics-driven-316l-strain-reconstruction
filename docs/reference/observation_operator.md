# DIC observation operator

**Category: Reference.**

The comparison operator $\mathcal M_{\mathrm{DIC}}$ maps a FEM result to the
support and convention of the experimental observation.

## Public modes

`direct_displacement_to_evm`
: Reconstruct EVM directly from canonical displacement. This is the
  historical comparison mode and remains available for regression.

`synthetic_disflow`
: Convert archived FEM displacement to image coordinates, warp the reference
  speckle image, re-estimate displacement with a declared DISFlow profile,
  convert back to canonical coordinates and reconstruct EVM.

Both modes:

1. obtain solve and core bounds from the campaign manifest;
2. verify the archived displacement hash against `status.json`;
3. reconstruct the required nodal support;
4. apply the shared `reconstruct_historical_evm` path;
5. use one explicit valid-data mask;
6. compute metrics on identical finite core samples.

The current historical EVM chain is implemented through the shared
`reconstruct_historical_evm` path. Plotting and validation code must reuse it
rather than reimplementing the formula.

## Image-level V3 operator

The optional measurement-chain implementation adds an image-level operator:

```text
reference speckle image
→ known or FEM displacement warp
→ OpenCV DISFlow
→ reconstructed displacement
→ historical EVM
```

Two immutable profile names are supported:

| Profile | Factory | Patch / stride | Other explicit DIS settings |
|---|---|---:|---|
| `legacy_script_2021` | no-argument factory | 4 / 1 | finest scale 0 |
| `declared_medium_v4` | medium preset | 8 / 3 | finest scale 0, GD 30, mean normalisation and spatial propagation enabled |

Both request variational refinement
\(\alpha=100,\delta=1,\gamma=0,\epsilon=0.002\) for 30 iterations. Parameters
left unset by the legacy source are represented by `None`: their setters are
not called, and every getter exposed by OpenCV is queried afterward.

Every run queries these values back from the OpenCV object and stores them in
its report. `legacy_script_2021` is primary by source provenance and
`declared_medium_v4` is a sensitivity. Neither may be selected by its
FEM/DIC score. These are **reproduction implementations**, not bitwise copies
of the historical executable whose OpenCV version and factory values were not
archived.

The nominal warp is `iterative_forward_inverse`. For a direct spatially
varying displacement it solves

\[
x_d=x_s+u(x_s)
\]

for the inverse sampling coordinates by fixed point. It reports convergence,
final coordinate residual and the minimum forward-map Jacobian. The
`legacy_approximate_inverse` mode,
\(x_s\approx x_d-u(x_d)\), is retained only for reproducing earlier synthetic
artefacts.

The canonical axis contract is defined in
{doc}`DIC axis conventions </reference/dic_axis_conventions>`. Image rows map
to canonical \(x,u_x\); image columns map to canonical \(y,u_y\). OpenCV flow
itself remains ordered `(column displacement, row displacement)`.

The unavailable historical mask is not guessed. New V3 artefacts use an
explicit all-valid boolean mask and record its type, unique value and hash.
This is sufficient for the P43 replay because no historical mask
reproduction is claimed.

The earlier `finest_scale=1` measurement-chain artefact is retained only as
an invalidated protocol record. Stopping above scale 0 discards native image
resolution and is prohibited for spatial metrology.

The pre-registered measurement-chain and P43 replay evidence is summarised in
{doc}`../explanation/current_evidence`; the runnable procedures are in
{doc}`../how-to/characterise_dic_measurement_chain`.

## Prohibited substitutions

- Do not compare PEEQ amplitude directly with DIC EVM.
- Do not filter the primary coupled FEM EVM after convergence.
- Do not infer core bounds from array shape.
- Do not transpose or flip fields independently for plotting.
- Do not tune a measurement filter separately for every candidate.

## Recorded metadata

The operator fingerprint and cache key include:

- archived FEM displacement hash and reference-image hash;
- profile name, requested and queried OpenCV settings;
- axis and displacement convention;
- pixel size and millimetre-to-pixel conversion;
- interpolation and warp mode;
- solve/core bounds and mask hash;
- EVM operator identity and post-filter status.

The replay writes DIC EVM, raw FEM EVM, observed FEM EVM and recovered image
flow separately. It never mutates the mechanical campaign and never applies a
post-filter to the primary EVM.

### Grid contract

The FEM displacement is nodal, shape `(nx + 1, ny + 1, 2)`, in canonical axes
`(x, y)` with components `(ux, uy)` in millimetres. The image crop is taken at
exactly that shape, so **one node is one pixel and the interpolation onto the
image grid is the identity**. The operator asserts the two shapes agree and
raises otherwise; it never resamples silently.

Canonical-to-image conversion swaps axes: image flow component 0 is the column
displacement `uy`, component 1 is the row displacement `ux`, both in pixels.

EVM is element-centred, shape `(nx, ny)`. The nodal and element-centred
lattices therefore differ by half a pixel. That offset is identical for DIC and
FEM, so it cancels in this comparison — but any geometric object expressed in
pixel coordinates, such as a band centreline, must declare which lattice it
lives on.

### Audit artefacts

| File | Contents |
|---|---|
| `dic_evm.npy` | experimental EVM on the comparison support |
| `fem_raw_evm.npy` | FEM EVM without observation, known-biased control |
| `fem_observed_evm.npy` | FEM EVM after the symmetric operator |
| `fem_displacement_image_grid.npy` | imposed flow in pixels, on the image grid |
| `synthetic_deformed_image.tif` | the warped reference image fed to DISFlow |
| `recovered_displacement_column.npy` | recovered column flow, pixels |
| `recovered_displacement_row.npy` | recovered row flow, pixels |
| `observed_flow_pixels.npy` | both recovered components, as one array |
| `valid_mask.npy` | comparison mask |
| `report.json` | the observation manifest |

Three names differ from the observed-EVM comparison specification, which asks
for `fem_evm_raw.npy`, `fem_evm_observed.npy` and `observation_manifest.json`.
The existing names are kept because archived reports hash them by name;
renaming would break the traceability of results already committed.

`valid_mask.npy` is currently `declared_all_valid`: no invalid region exists in
this dataset. It is written so downstream tooling can rely on the file being
present, not because it filters anything today.

### Metrological guard

The symmetric replay refuses a DISFlow profile whose finest scale is not the
native scale 0. The first measurement-chain campaign ran at scale 1, which
skips full-resolution variational refinement and reported an MTF-50 near
127 px against 49 px at native scale; that run was invalidated. Both archived
profiles pass the guard.

## Photometric-quality diagnostic

`diagnose-dic-photometric-quality` evaluates the direct brightness residual

\[
\operatorname{cell\_average}
\left(
\left|I_k(x+u_{\mathrm{DIC}}(x))-I_0(x)\right|
\right)
\]

on the element support used by V3. Sampling is bilinear and destination
coordinates outside the image are excluded geometrically. The operation does
not use the unavailable historical `mask.png`, does not normalise intensity
and does not alter DIC or FEM fields.

The command reports residual/error Pearson and Spearman association, fixed
residual deciles, primary unmasked field metrics and a sensitivity excluding
only residuals above the core q90. That sensitivity is never substituted for
the primary comparison. A low association is a negative result; it is not
permission to tune another residual threshold.
