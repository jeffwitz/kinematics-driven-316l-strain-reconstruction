# DIC axis and displacement conventions

**Mode:** reference  
**Domain:** dic

The maintained observation operator maps image arrays to canonical fields with
support `(x, y)` and components `[u_x, u_y]` in millimetres.  Image rows map to
canonical transverse `x`, image columns to tensile `y`; the received flow
components are exchanged accordingly and the spatial support is not
transposed.

Historical `U`/`V` names are interpreted from the registered P43 provenance:
`V_40` is transverse `u_x` and `U_40` is tensile `u_y`.  Masks, crop, pixel
size, units and any interpolation are recorded in the case manifest.  Plotting
code must not add an unrecorded transpose, flip or component exchange.

See {doc}`../scientific/observation_operator` for the operator contract and
{doc}`../../explanation/measurement/dic_observation_limits` for its limits.
