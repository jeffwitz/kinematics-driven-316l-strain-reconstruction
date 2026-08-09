# EBSD orientation assignment: the contract

This page is the current EBSD orientation contract. The orientation plumbing
and pixel-map provider are implemented: `mode: ebsd` accepts Bunge Euler angles
in degrees, converts them to validated rotations, and replicates the map to
the material points required by the selected batch. The P43 SRIX workflow has
also been qualified on an EBSD map, including the P43 M100 campaign. This page
therefore records both the implemented interface and the provenance conditions
that still apply to each experimental map.

The chain is

```text
EBSD pixel  ->  grain  ->  element  ->  Gauss point  ->  Q_global_to_material
```

and each arrow loses information or makes a choice. What follows fixes those
choices in advance so that a later map cannot quietly adopt a different one.

## What the code accepts

`PixelOrientationProvider` accepts an Euler-angle array of shape `(nx, ny, 3)`
through `orientation_provider_from_mapping({"mode": "ebsd", ...})`. It
converts each Bunge triple to a rotation, validates orthogonality and
determinant `+1`, and returns the per-point rotations consumed by the material
batch. A direct `(n_points, 3, 3)` rotation array is also accepted by the lower
level orientation contract.

The provider assigns the supplied pixel map to the case's material points; it
does not perform grain segmentation, grain averaging, or a physical
pixel-to-element homogenisation. Those remain case-level choices and must be
recorded when they are applied.

## Qualified use

The qualified P43 SRIX + EBSD workflow uses the registered EBSD provider with
the GPS backend and composite tangent described in
{doc}`../how-to/choose_mfront_backend`. The independent 3D-condensation route
accepts the same orientation map and remains the numerical reference. This
qualification validates the declared case and map; it does not make arbitrary
unproven orientation files interchangeable.

## The rotation convention

**`Q` maps global components to crystal components.**

$$\boldsymbol\varepsilon_{\text{crystal}} = Q\,\boldsymbol\varepsilon_{\text{global}}\,Q^{T},
\qquad
\boldsymbol\sigma_{\text{global}} = Q^{T}\boldsymbol\sigma_{\text{crystal}}\,Q$$

MGIS expects the opposite layout, material-to-global. That transpose happens in
exactly one function, `mgis_rotation_argument`, and nowhere else in the
codebase. A map arriving in the other convention must be transposed **before**
it reaches the orientation provider, not after.

## Euler angles

- **Bunge ZXZ**, the convention every EBSD system exports: rotate by
  $\varphi_1$ about $z$, then by $\Phi$ about the new $x$, then by $\varphi_2$
  about the new $z$.
- **Degrees** at the interface. `rotation_from_euler_bunge_deg` takes degrees;
  radians anywhere in a file must be converted at the reader, and the file must
  say which it holds.
- The triple $(\varphi_1, \Phi, \varphi_2)$ produced by the acquisition software
  is the **crystal-to-sample** orientation in most vendors' export. Whether the
  supplied map follows that or its inverse **must be established against the
  acquisition software's documentation, not guessed**, and recorded in the
  manifest. Getting it backwards is invisible in a cubic-symmetric average and
  fatal for a single grain.

## Frames

Three frames must be reconciled, and they are not the same.

| frame | axes | source |
|---|---|---|
| image | row, column, pixel index origin at a corner | the EBSD map file |
| sample | the specimen's own axes | the acquisition setup |
| mesh | `axis 0 = x` transverse, `axis 1 = y` tensile | `docs/scientific_contract.md` |

The mesh convention is already fixed by this project: axis 0 is physical `x`,
axis 1 is physical `y`, and the historical DIC names map as `V -> u_x`,
`U -> u_y`. **The EBSD-to-mesh mapping must be stated as an explicit axis
permutation and sign convention**, verified on a feature visible in both the
EBSD map and the DIC field, and recorded. A transposed map is not detectable
from statistics alone.

## Grains

- **Segmentation threshold.** The misorientation angle above which two
  neighbouring pixels belong to different grains must be stated with its value
  and its disorientation definition (minimum over the 24 cubic symmetries).
- **One orientation per grain, or per pixel?** Both are defensible. A
  grain-averaged orientation suppresses intragranular spread, which for this
  project's purpose is a modelling choice, not a cleanup. Whichever is used
  must be declared; if averaged, the averaging must be a proper mean of
  rotations, not a component-wise mean of matrices or of Euler angles.
- **Grain identifier.** Carried alongside the orientation so a result can be
  aggregated per grain. `FEMResult` does not yet carry it; adding it is part of
  the implementation this contract precedes.

## Cubic symmetry

Two orientations differing by one of the 24 proper cubic rotations describe the
**same crystal**. `cubic_rotations` provides them.

Consequences that must be respected: a disorientation is the minimum over the
group; averaging orientations requires reducing to a fundamental zone first, or
the mean is meaningless; and comparing two maps requires the same reduction.
Improper operations are excluded — a reflection maps a right-handed crystal onto
a left-handed one and `validate_rotations` refuses it.

## Non-indexed pixels

An EBSD map always has them: low band contrast, grain boundaries, inclusions.
The contract forbids the two easy answers.

- **Not silently filled** with a neighbour. That fabricates a measurement.
- **Not silently dropped**, leaving an element with no orientation.

The required behaviour is that a non-indexed fraction is **reported per element**
and that an element whose non-indexed fraction exceeds a declared threshold is
either excluded from the analysis or flagged in the result. The threshold is a
registered parameter, not a default.

## Interpolation and the pixel-to-element step

The EBSD pixel grid and the mesh do not coincide. The contract requires:

- **Orientations are never interpolated component-wise.** A matrix built by
  averaging nine numbers is not a rotation. Use a proper rotation mean, or
  nearest-neighbour assignment, and say which.
- **Nearest-neighbour is the default** and must be justified if departed from.
  It preserves the property that every assigned orientation is one that was
  actually measured.
- **Sub-element heterogeneity is reported**, not averaged away: an element
  spanning a grain boundary contains two crystals and one orientation cannot
  represent it. The count of distinct grains per element belongs in the result.

## Grain boundaries

An element straddling a boundary is the case the whole chain is weakest at, and
it is common: with an element size of `1.84 µm` and grains of a few tens of
microns, a boundary crosses a non-negligible fraction of the mesh. The contract
requires that such elements be **identified and counted**, and that any
conclusion drawn from a region be reported with the fraction of its elements
that are boundary-crossing.

## Provenance

The orientation map is an input like any other and is hashed like one:

- SHA-256 of the orientation map file;
- its shape, its pixel size, and the frame it is stated in;
- the Euler convention, in the terms above;
- the segmentation threshold and the grain-averaging choice;
- the non-indexed fraction, globally and per element;
- the assignment rule from pixel to Gauss point;
- the acquisition software and version, if the file records it.

This goes in the campaign manifest beside the existing input digests, in the
same way `crystal_parameters` does today.

## Remaining provenance requirements

- Do not draw a grain-level conclusion unless the orientation convention has
  been verified against a feature visible in both the EBSD map and the DIC
  field.
- Do not commit or publish an orientation map without its provenance block.
- Report the mapping, non-indexed fraction, assignment rule and any grain
  averaging or segmentation applied to the case.
