# NumPy SRIX backend

The repository keeps MFront/MGIS as the default SRIX backend.  The optional
`numpy-srix` backend is an independent, vectorised implementation of the
qualified Forest--Rubin SRIX equations and is selected explicitly through
`create_plane_stress_material_batch(..., backend="numpy-srix", ...)`.

## Contract and conventions

The public object implements `PlaneStressMaterialBatch` and preserves the
transactional `evaluate` / `commit` / `revert` contract.  The backend uses the
registered `SrixParameterSet` objects; it does not carry a second parameter
table.  Orientations are `Q_global_to_material`, matching
`crystal_orientation.py`.

The six-component Kelvin order is `[xx, yy, zz, xy, xz, yz]`.  Plane stress
eliminates all three transverse components `[zz, xz, yz]`, not only `zz`, and
condenses the algorithmic tangent with a batched linear solve.

## Local integration

The local Newton unknowns are the six elastic-strain increments and twelve
slip increments.  `Deq` is built from the Newton unknowns, as in
`validation/mfront/Fcc316LForestRubinSrixGeneric3D.mfront`; inactive systems
use the zero derivative of the Macaulay bracket.  The returned tangent is the
implicit Jacobian tangent, not a production finite-difference estimate.

Points are vectorised.  `batch_size` can split the point batch to bound Newton
workspace memory; no Python loop iterates over material points.

## Qualification status

The backend is an opt-in implementation and is not yet a replacement for
MFront.  The current fast tests cover elastic and plastic batch integration,
heterogeneous orientations, chunking, transaction behaviour, and the three
equations of plane-stress closure.  MFront-vs-NumPy path-by-path qualification,
full tangent tolerances, and performance benchmarks remain required before a
scientific qualification claim.

CuPy is intentionally out of scope for this first implementation.  The core
uses NumPy array operations and keeps the constitutive equations separate from
the plane-stress adapter so a future array-backend port can be audited against
this implementation.
