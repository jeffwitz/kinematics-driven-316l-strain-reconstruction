# Constitutive and nonlocal extension interfaces

**Category: Reference.**

## Constitutive plugin

`register_constitutive_plugin(identifier, builder)` registers a process-local
backend. Identifiers contain lower-case letters, digits, `.`, `_` or `-` and
cannot silently replace an existing entry.

The builder receives `PlaneStressMaterialRequest`, including the heterogeneous
legacy J2 maps for compatibility, generic `constitutive_options`, the MFront
library, thread count, plane-stress controls and optional coupling modulus. It
returns an object satisfying `PlaneStressMaterialBatch`.

Installed builders are discovered from `fem_inhouse.constitutive_plugins`.

The global solver only consumes:

- point count and backend metadata;
- in-plane stress and tangent trials;
- full-state completion after convergence;
- tangent matrix type;
- transactional `commit` and `revert`.

## MFront catalogue

`MFRONT_BEHAVIOURS` maps a stable identifier to `MFrontBehaviourSpec`. A
specification records behaviour names, variable bindings, hypothesis support,
rotation requirements, tangent capability and bridge profile. The catalogue
does not itself implement an unknown state integration algorithm.

The built-in `ludwik_j2_v1` bridge profile is executable by the current MGIS
adapter. A crystal-plasticity profile is supplied through a constitutive
plugin until a generic CP bridge is implemented and verified.

## Nonlocal criterion

`NONLOCAL_CRITERIA` maps a stable identifier to a factory returning
`ScalarNonlocalCriterion`. The criterion controls source extraction, external
field injection, the safety observable and the spatial regularisation. The
fixed-point algorithm controls convergence and transactions.

Installed criterion factories are discovered from
`fem_inhouse.nonlocal_criteria`.

The version-one contract is scalar. This restriction is explicit in the type
and configuration names so a later tensor-field interface can coexist without
changing the meaning of archived PEEQ results.
