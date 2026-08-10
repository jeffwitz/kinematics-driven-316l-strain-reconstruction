# Architecture decisions

The repository records current and superseded architecture decisions. This
page is the English public summary; the original ADR files remain the
authoritative versioned records.

## ADR 0001 — Restrict the software to the article case

Supported physics is limited to small-strain plane stress, a structured CPS4
mesh with 2 × 2 quadrature, J2/Ludwik plasticity, DIC-prescribed boundary
displacement, and padded 25- or 100-partition reconstruction.

**Consequence:** maturity is measured against the published case, not by the
number of general FE features.

## ADR 0002 — Require PyPardiso in production

PyPardiso/MKL is required before production assembly starts. SciPy/SuperLU
remains available only for explicitly configured small diagnostics.

**Consequence:** missing performance-critical dependencies fail early instead
of producing an unusably slow “successful” calculation.

## ADR 0003 — Stitch uniquely owned cores

Every partition solves its padded region, but only its core is retained. Every
global element and node has exactly one deterministic owner.

**Consequence:** no implicit averaging can hide interface artefacts; execution
order is irrelevant.

## ADR 0004 — Prepare DIC input explicitly

Raw files remain immutable. Axis mapping, pixel-to-millimetre conversion,
hardening scaling, non-finite repair, nodal completion, and central cropping
are named, hashed, and written to a separate directory.

**Consequence:** any preparation variant receives a distinct output directory
and can be audited byte-for-byte.

## ADR 0005 — Use analytical MFront by default

The nominal solver uses `PixelLudwikJ2Plasticity` through MGIS, with analytical
Ludwik hardening after a finite first interval and no upper PEEQ cap. The
1000-point Python table is an explicit historical regression path.

**Consequence:** production requires a compiled MFront behaviour; every
comparison must identify whether it targets the analytical model or the legacy
table.

## ADR 0006 — Superseded: condense 3D laws behind a plane-stress protocol

The original decision introduced the transactional plane-stress material
protocol and external 3D condensation. It is superseded by the current
structural plane-stress contract, which also provides a registered generic
closure while retaining external condensation as an independent reference.

**Consequence:** the historical protocol remains supported for compatibility;
current backend selection and validity are defined in
`reference/numerics/mfront_structural_plane_stress`.

## ADR 0007 — Reuse a fixed free-system CSR graph

The free--free stiffness matrix is represented by one CSR object whose
`indptr` and `indices` arrays never change during a solve. A precomputed
element-contribution map updates only `data`. PARDISO keeps this graph under
the real nonsymmetric matrix type (`mtype=11`), performs symbolic analysis
phase 11 once, and executes numerical factorization phase 22 followed by solve
phase 33 for every new tangent.

**Consequence:** COO construction, CSR conversion, free-system slicing, and
repeated symbolic analysis are removed from the Newton loop. This decision
does not assume matrix symmetry and does not modify Newton or the constitutive
tangent.

## ADR 0008 — Select PARDISO matrix type from a material capability

The verified Python and MFront J2/Ludwik behaviours declare their algorithmic
tangent symmetric positive definite. Their free--free matrix therefore stores
only its upper CSR triangle and uses PARDISO `mtype=2`. The constitutive
tangent is not symmetrized: its relative skew part is measured at runtime and
the solve is rejected above `1e-12`.

Any unclassified behaviour uses complete CSR storage and `mtype=11`. This is
the default contract for future crystal plasticity, anisotropic plasticity, or
other behaviours until symmetry and positive definiteness have been
demonstrated for their exact stress/strain measures, integration algorithm,
plane-stress condensation, and loading paths.

**Consequence:** current J2 solves benefit from symmetric factorization
without turning symmetry into a global assumption of the FEM kernel.

## ADR 0009 — Extend the observation operator rather than add a second one

The symmetric `synthetic_disflow` chain already existed and produced the
archived V3 and loading-path results. It is extended with the missing audit
artefacts, grid contract and metrological guard instead of being reimplemented.

**Consequence:** archived observed-EVM results stay comparable with future
candidates, which a second operator would have broken.

## ADR 0010 — Preserve the production nonlocal coupling strategy as the robustness reference

The canonical partitioned nonlocal method is the nested fixed point executed
inside every mechanical Newton evaluation. It reuses
`evaluate_nonlocal_fixed_point()` and preserves MFront transactions,
relaxation/Aitken, convergence criteria, line-search and cutback semantics.

The method is the robustness reference for candidate monolithic or Schur
couplings. A simplified alternation may be useful experimentally, but must be
labelled as such and cannot silently replace the reference.

See {doc}`../adr/0010-nonlocal-coupling-reference-strategy` for the decision.
