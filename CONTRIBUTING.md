# Contribution guide

This repository is research software for one published case study. Changes
must preserve the scope in `docs/adr/0001-case-study-scope.md`.

## Before opening a pull request

Run the same checks as CI:

```bash
.venv/bin/ruff check .
.venv/bin/mypy src/fem_inhouse
.venv/bin/pytest --cov=fem_inhouse --cov-branch
.venv/bin/pip wheel . --no-deps
```

Do not update a numerical threshold after inspecting the result it is meant to
accept. Record the threshold, units, field location and convention first.

## Numerical-formula review

A change to element kinematics, constitutive integration, tangent operators,
assembly, invariants or strain/stress reconstruction requires:

1. a second reviewer familiar with the corresponding mechanics;
2. a closed-form, finite-difference, or independent-reference test;
3. an explicit statement of axes, shear convention, units and field location;
4. a comparison against the previous implementation when results should be
   unchanged;
5. a performance/memory measurement when array shapes or sparse operations
   change.

The author of the formula must not be its only reviewer. Missing Abaqus or DIC
reference data must be reported as a blocker, never replaced by a synthetic
claim of scientific parity.

## Scope and legal status

General-purpose elements or material plugins are out of scope. The repository
does not yet declare a software license; external redistribution and
contributions remain subject to the project owner's licensing decision.
