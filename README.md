# Kinematics-Driven 316L Strain Reconstruction

Research software supporting the case study described in
`ArticleSource/ArticleAdil.pdf`: reconstruction of mechanically admissible
microscale strain-localization fields from DIC kinematics in 316L stainless
steel.

This project is deliberately **not** a general replacement for Abaqus. Its
supported scope is a regular CPS4 plane-stress mesh, J2 plasticity with
Ludwik-Hollomon hardening, DIC-prescribed boundary displacements, and
partitioned reconstruction of the article's pixel-resolved region of interest.

## Current status

The numerical prototype is being converted into tested, reproducible research
software. The live implementation plan and validation register are maintained
in [`Claude.md`](Claude.md). Scientific conventions are specified in
[`docs/scientific_contract.md`](docs/scientific_contract.md).

Known limitations at this stage:

- the full 11.16-million-element partitioned workflow is not implemented yet;
- the historical plotting scripts still depend on external case data;
- Abaqus parity is not yet established from the original `.inp` and ODB
  extraction scripts.

## Development setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/ruff check .
```

PyPardiso/MKL is a required runtime dependency for production solves.
