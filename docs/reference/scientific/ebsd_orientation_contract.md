# EBSD orientation contract

**Mode:** reference  
**Domain:** crystal-plasticity

Each material point may carry one crystal orientation. The declared convention
is $Q_{global\to material}$: for a tensor expressed in the structural frame,

$$
A_m=Q_{global\to material}A_gQ_{global\to material}^{T}.
$$

The same local orientation is used for the cubic elastic law and the twelve
FCC Schmid tensors. Spatial assignment from an exported EBSD array to a
material-point array follows mapping convention $F$. Internal spectral arrays
use storage order $C$. These are separate contracts:

```text
Q = frame rotation
F = EBSD-pixel → material-point assignment
C = array storage layout
```

Neither $F$ nor $C$ is evidence of physical co-registration between EBSD and
DIC acquisitions. A calculation must record the orientation source, Euler
convention, assignment rule, crop, axes, units and all transformation hashes.
No rotation, scale or origin may be inferred from a field fit.

## P43 metadata boundary

The accessible P43 export provides orientation-derived arrays on an exported
`3600 x 3100` support, but the registered report records:

```text
ebsd_global_geometry_known = false
ebsd_axis_metadata_found   = false
registration_proven        = false
```

The DIC scale `0.00184 mm/pixel` must not be reused as the native EBSD step
size. The latter, together with scan origin and specimen-frame axes, is not
independently documented. The working $F$ mapping is therefore provisional.

See {doc}`../../explanation/measurement/ebsd_registration_and_orientation`
for the scientific interpretation and
{doc}`../data/dic_axis_conventions` for the DIC component convention.
