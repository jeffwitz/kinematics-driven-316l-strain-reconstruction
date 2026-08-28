# Input-data contract

**Mode:** reference  
**Domain:** data

Canonical arrays use support `(x, y)`: axis 0 is transverse `x`, axis 1 is
tensile `y`, and displacement components are `[u_x, u_y]` in millimetres.
The nominal P43 preparation maps `V_40` to `u_x` and `U_40` to `u_y` using
`1.84 micrometres/pixel`; masks, crop, non-finite-value policy and edge
padding are declared in the preparation manifest.

Finite-element inputs contain nodal displacement and element material fields;
shapes, units and hashes are part of the calculation manifest.  Historical
component names do not override the declared experiment-specific mapping.
