# Configuration reference

**Mode:** reference  
**Domain:** software

Configuration is layered as case data, numerical solver options and
constitutive options.  Backend selection and plane-stress closure are
independent where supported; defaults must be recorded in the resolved
configuration used for a run.  Unknown options are rejected before solving.

The resolved file is part of the reproducibility manifest, together with
parameter provenance, tolerances, iteration limits, backend versions and
thread settings.
