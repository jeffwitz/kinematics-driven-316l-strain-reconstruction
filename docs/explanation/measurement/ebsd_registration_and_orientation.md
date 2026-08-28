# EBSD registration and crystal orientation

**Mode:** explanation  
**Domain:** crystal-plasticity

EBSD contributes a local crystal orientation; it is not itself a stress or
slip measurement.  Three coordinate spaces must remain distinct:

```text
EBSD pixel coordinates -> spatial registration -> material-point coordinates
                                              -> crystal/material frame
```

The repository records spatial assignment as mapping convention **F** and
stores spectral arrays in **C** order.  F is an assignment rule, not a global
transpose or a change of array layout.  Confusing the two can preserve global
histograms while moving every local orientation to the wrong material point.

For a tensor or gradient in the structural frame, the material-frame value is

$$A_m = Q_{global\to material} A_g Q_{global\to material}^{T}.$$

The orientation therefore changes the cubic elastic response, Schmid tensors,
resolved shears and all three transverse couplings in structural plane stress.
Registration errors can consequently change localisation without changing the
orientation histogram.

EBSD does not provide the active slip set, local resistance, stress or an
identified SRIX parameter.  Those are constitutive predictions conditioned on
the orientation, loading path, boundary data and parameter provenance.

The input convention is specified in
{doc}`../../reference/scientific/ebsd_orientation_contract`; the DIC mapping
contract is {doc}`../../reference/data/dic_axis_conventions`.
