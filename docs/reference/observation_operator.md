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
