# EBSD orientation contract

**Mode:** reference  
**Domain:** crystal-plasticity

Each material point may carry its own crystal orientation. The repository
stores the declared `Q_global_to_material` convention and applies it to the
crystal systems and elastic law. Spatial assignment from EBSD pixels to
material points follows mapping convention F; internal spectral storage keeps
its C ordering. These transformations are independent and must be recorded
separately in provenance.

The axes, Euler convention and source metadata are part of the input manifest;
no rotation may be inferred from a field fit.
