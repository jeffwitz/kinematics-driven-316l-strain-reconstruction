# Add a nonlocal criterion

**Mode:** how-to  
**Domain:** software

Define the criterion and its units in the constitutive reference, implement
it behind the existing material/field interface, and add a small synthetic
test before using experimental data.  Record length scales, boundary
conditions, padding and convergence tolerances in the run manifest.

Qualify the local and nonlocal fields separately; a smoother field is not by
itself evidence of a better reconstruction.  See
{doc}`../../reference/scientific/nonlocal_parameters` and the maintainer
extension interfaces for the stable contracts.
