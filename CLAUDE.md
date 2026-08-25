# Coding-agent bootstrap

This document is the repository bootstrap for coding agents, despite the
historical filename `CLAUDE.md`. It is intentionally short. Detailed project
knowledge is routed through [`docs/agent/README.md`](docs/agent/README.md).

## Repository knowledge architecture

Knowledge is split by role:

1. **`CLAUDE.md` — universal bootstrap.** This file contains the project
   purpose, non-negotiable scientific rules, repository layout, current
   high-level status and the context-retrieval policy.
2. **`docs/agent/` — canonical agent-oriented index and durable decisions.**
   Start with [`docs/agent/README.md`](docs/agent/README.md), then read only
   the domain documents relevant to the task. Do not recursively ingest the
   whole documentation tree.
3. **`.claude/rules/` — Claude Code path-scoped rules.** These are conditional
   context-loading shortcuts, not the canonical documentation. Rules needed
   by every agent must also be stated here or point to `docs/agent/`.
4. **`validation/` — quantitative evidence.** Numerical claims, benchmarks,
   gates and provenance belong in versioned validation artifacts.
5. **`docs/agent/history/` — superseded investigations.** Historical results
   must not silently override current code or current validation evidence.

The old [`Claude.md`](Claude.md) is a legacy project log. It is retained for
traceability; it is not a second bootstrap or a source of current status.

## Context retrieval policy

Before a non-trivial task:

1. inspect the current Git state;
2. locate relevant symbols, callers, dependencies and tests with the available
   code-knowledge tools (use repository search only when those tools are not
   available);
3. consult `docs/agent/README.md`;
4. read only the routed domain documents;
5. inspect current `validation/` artifacts when numerical claims matter;
6. inspect the implementation before trusting historical prose.

Evidence priority is:

```text
current code
> current committed validation artifacts
> canonical documentation
> historical documentation
> archived reports and comments
```

Do not assume that a commit, benchmark or solver status written in prose is
still current without checking it.

## Project and scientific invariants

This repository reconstructs 316L microscale mechanics from DIC kinematics,
with crystal-plasticity/MFront and spectral FEM workflows. The scientific
objective is experimental adequacy, not a general Abaqus replacement.

- Keep the distinction between forward mechanics, constitutive replay,
  sensitivity, identification and diagnostics explicit.
- Do not claim parameter identifiability from displacement-only data when the
  validation spectrum contains weak or gauge directions.
- Treat DIC boundary preparation and its units, axes and provenance as part of
  the experiment; do not silently filter, interpolate or delete frames to make
  a solve converge.
- A failed constitutive or equilibrium calculation is a diagnostic result,
  not permission for an undeclared fallback.
- For spatial mappings, use the actual mesh/index contract. In particular,
  the spectral solver's `(x, y, subcell)` material batch is C-order, while
  classical `StructuredMesh` element numbering may be Fortran-order; these
  conventions must not be conflated.
- Quantitative claims require a current artifact under `validation/` and its
  exact provenance.

## Repository layout

```text
src/fem_inhouse/       implementation
tests/                 unit and integration tests
scripts/               reproducible diagnostics and campaigns
docs/                  user, scientific and reference documentation
docs/agent/            agent routing and durable decisions
validation/            versioned evidence and campaign artifacts
mfront/                MFront source and build inputs
```

Use the installation and environment instructions in `README.md` and the
routed documentation. MFront/MGIS workflows require the locally installed
TFEL environment; do not infer a successful constitutive run from a Python
environment that cannot import MGIS.

## Current high-level status

The repository contains several validated numerical cores and historical
investigations. The status of any particular FEMU, SRIX, TANN or P43 campaign
must be read from its current validation artifact, not inferred from this
bootstrap. In particular, synthetic validation does not authorize an
experimental identification campaign by itself.
