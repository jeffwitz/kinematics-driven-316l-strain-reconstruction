# E-SRIX-P43-EBSD-MAPPING-AUDIT-001

## Index-only result

The mandatory non-square sentinel test used a `3 x 5` grid and
`StructuredMesh.elem_ids` as the source of truth. It found a real ordering
incompatibility:

```text
StructuredMesh element ids: Fortran numbering
PixelOrientationProvider default: C flattening
```

The current provider disagrees with the mesh association for **12 of 15
elements**. The explicit `element_order="F"` provider agrees for all 15
elements. The complete association is archived in
`reference_data/p0043_ebsd_mapping_audit_v1/element_mapping.csv`.

The mismatch is not a transpose of a square image; on the rectangular grid it
is the direct consequence of using C-order pixel flattening against the
StructuredMesh F-order element numbering.

## Scope and status

This ticket has not run mechanics, MFront, Schmid calculations, or parameter
identification. It proves an index-order defect for the classical
`StructuredMesh` FEM contract and adds an explicit F-order option. The
spectral pixel solver keeps its documented C-order convention and is not
silently changed by this audit.

The remaining DIC/EBSD axis direction, crop registration, and sample-frame
orientation are **not proven** by this ticket. Historical experimental
localisation claims therefore remain suspended pending those audits and one
old-vs-corrected mechanical forward comparison.

## Regression protection

The unit tests now cover:

* rectangular sentinel mapping against `mesh.elem_ids`;
* all material points receiving the element's pixel orientation;
* explicit distinction between spectral C-order and StructuredMesh F-order.

The implementation change is explicit: classical `run_fem` EBSD mappings
default to F-order at the StructuredMesh boundary, while direct spectral
pixel calls retain C-order unless they request another convention.

## Registration closure update

The nominal M20 C/F forwards and the independent Schmid diagnostic have now
also been run. The explicit F mapping changes the classical prior EVM field
substantially (`relative L2(F-C) = 0.2827`), so the correction is mechanically
important. This is not a C-versus-F physical comparison: the C ordering is
retained only as a historical control.

The HDF5 metadata records the full crop and pixel size, but not the physical
EBSD-to-DIC transform, image-axis direction, or specimen-frame rotation. The
maintained DIC contract proves the canonical DIC axes; it does not prove those
three EBSD relations. The internal Schmid check on the P43 crop selects Euler
axis 1 as the best match to the archived HDF5 Schmid map, but this is only an
internal consistency diagnostic and is not acquisition provenance.

The corrected F RAW M20 exploratory run was completed with the provisional
rank-7 SVD basis. It reduced the raw displacement RMS from `4.0672e-6 mm` to
`3.5764e-6 mm` (12.07%); the Gauss--Newton linear prediction was 19.76%. The
optimizer stopped after three evaluations with a positive directional
derivative and several physical bounds active, and its final verification
residual was `5.80e-7`; therefore this is not a converged minimum. It is a
completed diagnostic, not an identification result.

The machine-readable closure status is in
`reference_data/p0043_ebsd_mapping_audit_v1/registration_provenance.json`.
