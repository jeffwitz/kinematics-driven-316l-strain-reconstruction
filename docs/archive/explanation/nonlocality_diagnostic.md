# Historical record: nonlocality diagnostic

:::{admonition} Superseded
:class: warning
Historical record. Superseded for current scientific interpretation.
:::

## The scientific question

The finite-element solution can concentrate plastic deformation into bands
that are narrower and sharper than the localization observed by DIC. Before
introducing a coupled nonlocal material model, this diagnostic asks a smaller
question:

> With the constitutive law, loading, and mechanical solution left unchanged,
> does one spatial smoothing length improve agreement between FEM and DIC?

This is an output-only experiment. It does not modify the J2 law, MFront,
Newton iterations, finite-element unknowns, or stresses. A favourable length
is therefore a **diagnostic spatial scale**, not an identified material
internal length.

## Element-centred Helmholtz problem

For an element field \(q\), the filtered field \(\bar q\) satisfies

$$
\bar q-\ell^2\Delta\bar q=q,
\qquad
\nabla\bar q\cdot\mathbf n=0
\quad\text{on }\partial\Omega.
$$

The implementation works directly at the centres of the structured
rectangular elements:

$$
\left(I+\ell^2L_h\right)\bar{\mathbf q}=\mathbf q,
$$

where \(L_h\) is the positive finite-difference approximation of
\(-\Delta\). Missing neighbour terms are omitted at a boundary, which is the
discrete zero-flux condition.

No element-to-node-to-element projection is introduced. Consequently,
\(\ell=0\) returns an exact copy of the source array and measures the existing
comparison pipeline without additional smoothing.

## Why a cosine transform?

For an \(n_x\times n_y\) element grid with spacings \(h_x,h_y\), the
orthonormal DCT-II diagonalizes the discrete Neumann operator. Its
eigenvalues are

$$
\lambda^x_k=\frac{2-2\cos(\pi k/n_x)}{h_x^2},
\qquad
\lambda^y_l=\frac{2-2\cos(\pi l/n_y)}{h_y^2},
$$

and each transformed coefficient is divided by

$$
1+\ell^2\left(\lambda^x_k+\lambda^y_l\right).
$$

This gives an \(O(N\log N)\) deterministic solve without assembling or
factorizing a sparse matrix. The zero-frequency coefficient is unchanged, so
the full-domain mean is conserved to floating-point precision. Unit tests
compare the spectral solution with a direct sparse solution on small grids.

## Comparable observable on both sides

The primary observable is `EVM_HISTORICAL`. It is reconstructed independently
from DIC and FEM nodal displacements through exactly the same chain:

```text
strain_from_displacement
  -> plane_stress_equivalent_strain(engineering shear)
  -> cell_average
```

Only the FEM field is filtered. The DIC reference is never filtered.

`PEEQ` is also filtered to show how the internal plastic localization changes,
but it is not an amplitude-equivalent DIC observable. The workflow therefore
reports its diffusivity and normalized localization overlap, never a
PEEQ-to-DIC-EVM RMSE or MAE.

## Padding and the retained core

The Helmholtz equation is solved over the complete saved partition, including
its padding. Metrics are then evaluated only on the retained core. Filtering
the core alone would impose a new artificial zero-flux boundary exactly where
the campaign is meant to suppress boundary influence.

For every positive length, the workflow records

$$
r_{\mathrm{padding}}
=\frac{\text{minimum physical artificial padding}}{\ell}.
$$

Lengths below the configured ratio, four by default, remain available for
inspection but are labelled `boundary_contaminated`. Padding is measured from
the partition metadata, not inferred from array shapes.

## What the metrics distinguish

Amplitude agreement is described by RMSE, MAE, signed bias, maximum absolute
error, and relative \(L_2\) error. Pearson correlation measures linear spatial
association. Quantile masks compare where the strongest fractions of the two
fields occur.

Equal-fraction masks cannot show whether a band became wider. A second family
therefore computes an absolute threshold from a DIC quantile and applies that
same numerical threshold to DIC and filtered FEM. It records the two active
areas, IoU, precision, and recall.

Diffusivity metrics track mean drift, standard deviation, peak attenuation,
gradient RMS, spacing-aware discrete total variation, and the relative change
from the raw FEM field.

## Evidence from article partition 0

The preserved campaign filters the saved `510×460`-element MFront solution,
including 150 pixels of padding, and evaluates the `360×310` retained core.
It sweeps 0, 1, 2, 4, 8, 16, and 32 physical pixels
(\(0\) to \(58.88\,\mu\mathrm m\)).

| Metric | Raw FEM | FEM at \(58.88\,\mu\mathrm m\) |
|---|---:|---:|
| RMSE | `2.6859e-3` | `1.3577e-3` |
| relative \(L_2\) | `0.81743` | `0.41319` |
| Pearson correlation | `-0.02920` | `0.09257` |
| top-10% IoU | `0.05030` | `0.13116` |
| standard-deviation ratio | `1.000` | `0.2316` |
| peak ratio | `1.000` | `0.1225` |

The main errors fall by 49.45%, and the quantile localization improves.
However, all exploratory rankings select the largest tested length, so the
best scale is not bracketed. The final correlation is still only `0.0926`,
and the largest length suppresses every FEM value above the DIC 90th
percentile threshold. The more balanced absolute-threshold overlap occurs at
\(14.72\,\mu\mathrm m\).

The initial implementation-partition conclusion was therefore:

> **The spatial-width hypothesis is partially supported on this exploratory
> partition.**

Scalar smoothing helps width and some overlap measures, but does not reproduce
the DIC texture or establish a material length. Confirmatory evidence requires
selecting a length on one partition and applying it unchanged to held-out
partitions.

The complete fields, metrics, figures, hashes, and reports are preserved in
`validation/reference_data/nonlocality_helmholtz_article_p0000_v1`.

## Pre-registered selection and confirmation

Partition 0 was subsequently excluded from length selection because it is not
representative of the agreement visible in the published figures. Partition
48 was declared as the selection region before its in-house solve. On P48,
`58.88 µm` improves all three primary pre-registered spatial metrics:

| Metric | Raw P48 | Filtered P48 |
|---|---:|---:|
| Pearson correlation | `0.29828` | `0.61597` |
| top-10% IoU | `0.15984` | `0.28224` |
| DIC-q90 absolute-threshold IoU | `0.16757` | `0.30852` |
| relative L2 | `0.80956` | `0.28649` |

The candidate was then frozen and applied without adjustment to partition 42,
which had been declared as held out before P48 was calculated:

| Metric | Raw P42 | Filtered P42 |
|---|---:|---:|
| Pearson correlation | `0.40068` | `0.70360` |
| top-10% IoU | `0.13340` | `0.27594` |
| DIC-q90 absolute-threshold IoU | `0.17735` | `0.25726` |
| relative L2 | `0.80047` | `0.27669` |

All automatic confirmatory thresholds pass. The filtered q90 active fraction
is `7.74%`, inside the pre-declared `[5%,20%]` anti-collapse interval for the
10% DIC reference.

The stage-1 conclusion is now:

> **The spatial-width hypothesis is supported.**

The same candidate improves amplitude and localization metrics on the
selection and held-out partitions. This does not identify a material internal
length: `58.88 µm` is still the upper boundary of the original sweep, only one
held-out partition has been tested, and the mechanical solution remains
strictly local and unchanged.
