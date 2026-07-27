# Input data contract

## Coordinate and array convention

Every structured array follows one convention:

| Quantity | Meaning |
|---|---|
| axis 0 | physical $x$, transverse direction |
| axis 1 | physical $y$, tensile/loading direction |
| $u_x$ | displacement along $x$ |
| $u_y$ | displacement along $y$ |

For `numpy.gradient(a, dx, dy)`, the returned derivatives are therefore along
$(x,y)$, in that order.

## Raw arrays

All four received arrays have shape `(3600, 3100)` and are immutable Git LFS
objects.

| Raw file | Physical meaning | Raw unit | Canonical mapping |
|---|---|---|---|
| `U_40.npy` | vertical DIC displacement | pixel | `displacement_y_mm` |
| `V_40.npy` | transverse DIC displacement | pixel | `displacement_x_mm` |
| `el_thresh50.npy` | local initial yield stress | MPa | `yield_stress_mpa` |
| `Hardening_coeff_el_Thresh50.npy` | local hardening multiplier | dimensionless | `hardening_coefficient_mpa` |

The `U`/`V` mapping is historical and must not be inferred from the letters
alone.

## Canonical arrays

For an element grid `(nx, ny)`:

| Canonical file | Location | Shape | Unit | Domain |
|---|---|---:|---|---|
| `displacement_x_mm.npy` | nodes | `(nx+1, ny+1)` | mm | finite |
| `displacement_y_mm.npy` | nodes | `(nx+1, ny+1)` | mm | finite |
| `yield_stress_mpa.npy` | elements | `(nx, ny)` | MPa | finite, strictly positive |
| `hardening_coefficient_mpa.npy` | elements | `(nx, ny)` | MPa | finite, non-negative |

For the complete ROI, `nx=3600` and `ny=3100`. The physical spacing is

$$
\Delta x=\Delta y=0.001\ \mathrm{mm}\times1.84
             =0.00184\ \mathrm{mm}.
$$

The resulting domain is $6.624\times5.704\ \mathrm{mm}^2$.

## Canonical transformations

The nominal preparation profile applies:

$$
u_x = 0.00184\,V_{40},\qquad
u_y = 0.00184\,U_{40},
$$

with displacement expressed in millimetres, and

$$
K(x,y)=380\ \mathrm{MPa}\times
\texttt{Hardening\_coeff\_el\_Thresh50}(x,y).
$$

`yield_stress_mpa` is copied from `el_thresh50.npy` without a scale factor.

The historical generator used `396 MPa` for the hardening scale. That value is
available only through an explicit preparation option and is not the nominal
article profile.

## Nodal completion

The received displacement support is `3600 × 3100`, while a mesh with that
number of elements requires `3601 × 3101` nodes. The supported
`edge-pad-upper` rule duplicates the final row and final column:

```text
u[nx, :] = u[nx-1, :]
u[:, ny] = u[:, ny-1]
```

This introduces a zero normal gradient at the newly created upper boundaries.
The rule is named and stored in the preparation manifest; it is not hidden in
the solver.

## Non-finite hardening values

Nine hardening multipliers are non-finite:

```text
(273, 3096), (904, 2), (933, 5), (933, 6), (1591, 2),
(1602, 6), (1949, 0), (1949, 1), (3599, 357)
```

The default policy is `error`. The nominal complete preparation explicitly
selects `nearest`, which copies the closest finite value and records every
modified index. The repaired fraction is $9/11{,}160{,}000$, but the repair
remains a scientific assumption.

## Baseline limitation

Only DIC step 40 is versioned. Historical baseline steps 1–5 are absent.
Current calculations use the provided final displacement directly and do not
claim to reproduce an unavailable baseline subtraction.

The broader availability audit, including images, loading, geometry, DISFlow
and EBSD-derived fields, is maintained in
{doc}`experimental_data_inventory`.

## Provenance hierarchy

Three manifests preserve distinct identities:

1. `data/raw/case_study/manifest.json` identifies received bytes;
2. the prepared-data manifest identifies transformations and canonical bytes;
3. the calculation manifest identifies code, configuration, layout, and input
   hashes.

Every partition status then identifies the six result files.
