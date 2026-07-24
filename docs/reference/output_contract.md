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

## Element representation

Stress and strain output is stored as one value per element after
post-processing the four Gauss points. The raw MFront state remains a
Gauss-point implementation detail and is not the public result contract.

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

## Solver diagnostics

`SolverDiagnostics` records:

| Group | Fields |
|---|---|
| identity | `backend` |
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
