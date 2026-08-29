# EBSD registration and crystal orientation

**Mode:** explanation  
**Domain:** crystal-plasticity

EBSD contributes a local crystal orientation; it is not itself a stress or
slip measurement. Three relationships must remain distinct:

```text
EBSD acquisition pixels
        │ physical registration (not yet proven)
        ▼
material-point assignment F
        │ local crystal orientation Q
        ▼
crystal/material frame
```

The first arrow is a physical statement about the specimen and two
acquisitions. The second is a numerical assignment convention. A coherent
indexing rule does not prove that an EBSD pixel and a DIC location are the same
physical point.

## Orientation versus assignment versus layout

For a tensor or gradient in the structural frame, the material-frame value is

$$
A_m=Q_{global\to material}A_gQ_{global\to material}^{T}.
$$

The orientation $Q$ rotates the cubic elastic law and the FCC Schmid tensors.
The mapping $F$ assigns an exported EBSD value to a material point. The array
layout $C$ is only the internal organisation of spectral storage:

```text
crystallographic orientation Q  ≠  assignment convention F
assignment convention F         ≠  storage layout C
storage layout C                ≠  physical registration proof
```

Confusing these operations can preserve orientation histograms, grain
fractions or other global distributions while moving every local orientation
to the wrong material point and destroying spatial correlations.

## What EBSD contributes to mechanics

```text
EBSD orientation Q
        ↓
rotated cubic elasticity + rotated FCC slip systems
        ↓
resolved shears and internal plane-stress couplings
```

The orientation therefore affects resolved shear stresses and the three
transverse couplings in structural plane stress. EBSD does **not** directly
provide stress, active slip, hardening, plastic strain or an identified SRIX
parameter. Those remain constitutive predictions conditioned on orientation,
loading path, boundary data and parameter provenance.

## Registered P43 status

The P43 data-only registration report completed its indicator analysis, but its
recorded statuses remain:

```text
ebsd_global_geometry_known = false
ebsd_axis_metadata_found   = false
registration_proven        = false
```

The working $F$ mapping can therefore be used for explicitly labelled
registered-case calculations. It must not be described as independently
verified experimental co-registration, and it must not be selected solely
because it maximises a mechanical correlation.

## What would upgrade the status?

An independent upgrade would require a documented specimen/acquisition
correspondence, such as common fiducials or landmarks, the scan origin and
axes, the EBSD native step size, and a recorded transformation with its
uncertainty. Until those are supplied, orientation correctness and physical
registration remain separate claims.

The input convention is specified in
{doc}`../../reference/scientific/ebsd_orientation_contract`; DIC axes and
components are specified in {doc}`../../reference/data/dic_axis_conventions`.
