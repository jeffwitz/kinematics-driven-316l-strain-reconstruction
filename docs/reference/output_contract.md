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
| `PLANE_STRESS_RESIDUAL_MPA` | `plane_stress_residual_vector_mpa` | elements `(nx, ny, 3)` | MPa |
| `S33_RESIDUAL_MPA` | `plane_stress_residual_mpa` | elements `(nx, ny)` | MPa |
| `PEEQ_NONLOCAL` | `nonlocal_equivalent_plastic_strain` | elements `(nx, ny)` | — |
| `PEEQ_MISMATCH` | `equivalent_plastic_strain_mismatch` | elements `(nx, ny)` | — |
| `NONLOCAL_HARDENING_MPA` | `nonlocal_hardening_mpa` | elements `(nx, ny)` | MPa |
| `YIELD_SURFACE_RADIUS_MPA` | `yield_surface_radius_mpa` | elements `(nx, ny)` | MPa |
| `NONLOCAL_RESIDUAL` | `nonlocal_residual` | elements `(nx, ny)` | — |

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
conventions. The six new arrays are additional fields; no 2D result is
replaced.

The five nonlocal fields are present only when micromorphic plasticity is
enabled. `PEEQ` remains the local Gauss-point-averaged accumulated plastic
strain \(p_e\). `PEEQ_NONLOCAL` is \(\chi\), `PEEQ_MISMATCH` is
\(p_e-\chi\), and `NONLOCAL_RESIDUAL` is
\(\chi-\mathcal H_\ell(p_e)\). They are products of the coupled constitutive
solve, not output-only filtered EVM fields.

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

`PLANE_STRESS_RESIDUAL_MPA` is ordered
`[S33, S13, S23]`. `S33_RESIDUAL_MPA` is exactly its first component and
`S_3D[..., 2, 2]`. It remains a compatibility view for older consumers.
Python returns an exact zero vector. Both MFront paths preserve their native
numerical residual instead of replacing it after integration. The maximum is
controlled at each Gauss point before element averaging.

The `tensor_reconstruction_source` diagnostic is one of:

| Value | Meaning |
|---|---|
| `j2_isotropic_analytical` | completion from the converged Python J2 plastic state |
| `mfront_native_plane_stress` | native MGIS `PlaneStress` state |
| `mfront_3d_local_condensation` | local condensation of a six-component MFront law |

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
| timing | `elapsed_seconds`, initialization, elastic assembly, constitutive, tangent assembly, fixed sparse assembly, free-system extraction, linear solve, output |
| PARDISO phases | analysis (11), numerical factorization (22), solve (33): elapsed time and call count for each phase |
| increments | attempted, converged, cutbacks |
| Newton | total iterations, maximum iterations |
| convergence | final norm, final relative residual, criterion name |
| local plane stress | maximum Gauss-point residual, maximum and mean local iterations, local failures, maximum `cond(Cbb)` |
| micromorphic coupling | enabled flag, \(\ell\), \(H_\chi\), \(\omega\), iterations per Newton and increment, total/maximum/mean iterations, final coupling residual, maximum Helmholtz residual, mean drift, Helmholtz/MFront time, failures |

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
        ├── PLANE_STRESS_RESIDUAL_MPA.npy
        ├── S33_RESIDUAL_MPA.npy
        ├── PEEQ_NONLOCAL.npy          # coupled campaigns only
        ├── PEEQ_MISMATCH.npy          # coupled campaigns only
        ├── NONLOCAL_HARDENING_MPA.npy # coupled campaigns only
        ├── YIELD_SURFACE_RADIUS_MPA.npy
        ├── NONLOCAL_RESIDUAL.npy
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

`load_full_tensor_state(directory)` loads complete tensor files when present.
An earlier complete-tensor directory without the vector residual is accepted;
its scalar `S33_RESIDUAL_MPA` is promoted to `[S33,0,0]`.

A directory containing only historical `S.npy`, `E.npy`, and `PE.npy` is
reconstructed only when both `poisson_ratio` and the explicit capability
`completion_strategy="j2_isotropic_analytical"` are supplied. This prevents a
J2-specific closure from being silently applied to another material law.
Other partially present tensor files are rejected.
