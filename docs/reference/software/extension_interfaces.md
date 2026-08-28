# Extension interfaces

**Mode:** reference  
**Domain:** software

Extensions must preserve the solver contracts for array layout, units,
transactional material state, response levels and diagnostics.  A new
constitutive backend implements the common plane-stress material protocol;
new transform backends implement the spectral transform protocol without
changing the global residual definition.

Every extension needs focused tests, a provenance record and an explicit
qualification status.  See {doc}`../architecture/documentation_architecture`
for the documentation contract and the relevant numerical references.
