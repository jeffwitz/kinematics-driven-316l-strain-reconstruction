# Output contract

## Final fields

For a solved element grid `(nx, ny)`, the public arrays are:

| Name | Python attribute | Location and shape | Unit |
|---|---|---|---|
| `U` | `displacement_mm` | nodes `(nx+1, ny+1, 2)` | mm |
| `S` | `stress_mpa` | elements `(nx, ny, 3)` | MPa |
| `E` | `total_strain` | elements `(nx, ny, 3)` | — |
| `PE` | `plastic_strain` | elements `(nx, ny, 3)` | — |
| `PEEQ` | `equivalent_plastic_strain` | elements `(nx, ny)` | — |
| `RF` | `reaction_force` | nodes `(nx+1, ny+1, 2)` | N for implicit 1 mm thickness |
| `S_3D` | `stress_tensor_mpa` | elements `(nx, ny, 3, 3)` | MPa |
| `E_3D` | `total_strain_tensor` | elements `(nx, ny, 3, 3)` | — |
| `EE_3D` | `elastic_strain_tensor` | elements `(nx, ny, 3, 3)` | — |
| `PE_3D` | `plastic_strain_tensor` | elements `(nx, ny, 3, 3)` | — |
| `S33_RESIDUAL_MPA` | `plane_stress_residual_mpa` | elements `(nx, ny)` | MPa |

The component order is:

| Field | Components |
|---|---|
| `U` | $(u_x,u_y)$ |
| `S` | $(S_{11},S_{22},S_{12})$ |
| `E` | $(E_{11},E_{22},\gamma_{12})$ |
| `PE` | $(PE_{11},PE_{22},\gamma^p_{12})$ |
| `PEEQ` | scalar |
| `RF` | $(R_x,R_y)$ |

Shear strain in element vectors is engineering shear:
$\gamma_{12}=2\epsilon_{12}$. Shear stress is tensorial $S_{12}$.

The historical six arrays retain their previous shapes, values, and component
conventions. The five new arrays are additional fields; no 2D result is
replaced.

## Complete tensor contract

All four tensors are exactly symmetric. Their component layout is:

$$
\mathbf{S}_{3D} =
\begin{bmatrix}S_{11}&S_{12}&0\\S_{12}&S_{22}&0\\0&0&S_{33}\end{bmatrix},
\qquad
\mathbf{E}_{3D} =
\begin{bmatrix}E_{11}&\gamma_{12}/2&0\\
\gamma_{12}/2&E_{22}&0\\0&0&E_{33}\end{bmatrix}.
$$

`EE_3D` and `PE_3D` use the same tensorial shear layout and satisfy

```text
E_3D = EE_3D + PE_3D
```

within the declared numerical tolerance. Associated J2 flow also satisfies
`trace(PE_3D) = 0`.

`S33_RESIDUAL_MPA` is exactly `S_3D[..., 2, 2]`. It is zero by construction
for Python. For MFront it preserves the native numerical residual of the
plane-stress local solve. Components 13 and 23 are zero under the membrane
kinematic assumption.

The `tensor_reconstruction_source` diagnostic is one of:

| Value | Meaning |
|---|---|
| `python_analytical` | completion from the converged Python 2D plastic state |
| `mfront_native_axial_strain` | native MGIS `AxialStrain`, `ElasticStrain`, and `Stress` |
| `mfront_analytical_fallback` | documented analytical completion because native axial strain is unavailable |

See {doc}`../explanation/plane_stress_tensors` for the mechanical derivation.

## Element representation

Stress and strain output is stored as one value per element after
post-processing the four Gauss points. The raw MFront state remains a
Gauss-point implementation detail. Native complete tensors are averaged over
the same points as the historical fields, so their leading dimensions match.

## Reaction convention

`RF[i, j, component]` is the internal nodal force on a prescribed degree of
freedom, using the same axis and sign as displacement. Free degrees of freedom
are zeroed in this output.

The two-dimensional kernel applies no thickness multiplier. With millimetres
and megapascals, reactions therefore correspond to an implicit thickness of
1 mm. The article reports a 2 mm specimen, but the original Abaqus section
thickness remains unverified. Quantitative reaction parity must wait for that
information.

## Optional frames

Requested pseudo-time snapshots contain:

- displacement;
- stress;
- total strain;
- PEEQ.

The final state is always returned independently of snapshot selection.
Complete tensors belong to that final-state result contract; the optional
historical snapshot payload remains unchanged.

## Solver diagnostics

`SolverDiagnostics` records:

| Group | Fields |
|---|---|
| identity | `backend`, `tensor_reconstruction_source` |
| timing | `elapsed_seconds`, initialization, elastic assembly, constitutive, tangent assembly, linear solve, output |
| increments | attempted, converged, cutbacks |
| Newton | total iterations, maximum iterations |
| convergence | final norm, final relative residual, criterion name |

Campaign status also records write time because filesystem output occurs after
the typed solver result has been returned.

## Campaign layout

```text
campaign/
├── manifest.json
└── partitions/
    └── 0000/
        ├── U.npy
        ├── S.npy
        ├── E.npy
        ├── PE.npy
        ├── PEEQ.npy
        ├── RF.npy
        ├── S_3D.npy
        ├── E_3D.npy
        ├── EE_3D.npy
        ├── PE_3D.npy
        ├── S33_RESIDUAL_MPA.npy
        └── status.json
```

`status.json` is written last, after all arrays have been atomically replaced.
It contains:

- `complete: true`;
- the partition identifier;
- the campaign-manifest hash;
- solver diagnostics;
- one SHA-256 per array.

## Stitched fields

Stitched arrays retain the same component contract at global shape. Every
global element and node has one deterministic owner. Padding is omitted from
the stitched field.

## Reading earlier result directories

`load_full_tensor_state(directory, poisson_ratio=...)` loads the five new
files when present. A directory containing only historical `S.npy`, `E.npy`,
and `PE.npy` can be reconstructed analytically only when `poisson_ratio` is
provided. Without that material property, the loader raises a clear error.
Partially present new tensor files are rejected rather than silently mixed
with reconstructed values.
