# Architecture decisions

The repository records five accepted architecture decisions. This page is the
English public summary; the original ADR files remain the authoritative
versioned records.

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

