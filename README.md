# Kinematics-Driven 316L Strain Reconstruction

Research software for full-field computational mechanics and inverse identification from measured kinematics in 316L stainless steel. The project combines DIC kinematics and EBSD crystal orientations with mechanically admissible reconstruction, three-dimensional constitutive laws under structural plane stress, spectral and matrix-free mechanics, and observation-aware identification methods.

## Origin and relationship to the original implementation

This project grew out of the **DIC-based in-house FEM implementation developed by Adil Kılınç**, available at:

<https://github.com/kilincadil/DIC-based-inHouse-FEM>

That implementation provided the initial computational basis for the kinematics-driven reconstruction workflow, in particular the structured finite-element treatment of DIC-prescribed boundary conditions and the J2/Ludwik reference model. Its scientific contribution remains part of the provenance of this repository and should be acknowledged when the present software is used or cited.

The present repository is intentionally maintained as a **separate downstream research project rather than as a replacement for, or a merge into, the original repository**. Since the initial import, its scope and software architecture have diverged substantially: the original solver has been refactored into a modular package and the project now includes structural plane-stress condensation of 3D material laws, MFront/MGIS integration, Forest--Rubin SRIX crystal plasticity, native NumPy/Numba constitutive kernels, spectral and matrix-free mechanics, DIC/EBSD registration and qualification, non-local formulations, inverse identification and observability analysis, automated validation, CI, and a dedicated scientific documentation stack.

Keeping the repositories distinct preserves the provenance and maintenance boundaries of both developments. Generic improvements that are useful to the original implementation can still be transferred independently when appropriate.

Citation metadata in [`CITATION.cff`](CITATION.cff) retain Adil Kılınç and the authors of the associated scientific work. The documentation also records the distinction between inherited foundations, current implementations, and validated scientific claims.

## What the current project does

The current workflow imposes measured displacement on the boundary of a structured plane-stress domain. Interior displacement, strain, stress and crystal-plasticity fields then follow from compatibility and mechanical equilibrium.

The principal modelling paths are:

- **J2/Ludwik**, retained as the historical and numerical baseline;
- **MFront/MGIS**, used as the qualified reference backend for three-dimensional constitutive behaviour under structural plane stress;
- **Forest--Rubin SRIX**, available through MFront and through a native NumPy/Numba implementation;
- **spectral and matrix-free mechanics**, developed for full-Dirichlet reconstruction, scalable nonlinear solves, sensitivities and inverse problems;
- **DIC/EBSD observation and identification workflows**, including measurement-chain qualification, registration, FEMU/SVD observability analysis and reproducible evidence tracking.

This is deliberately not a general Abaqus replacement. It is a research platform centred on mechanically admissible reconstruction and identification from measured kinematics.

## Current scientific status

The repository records qualification evidence for DIC/EBSD preparation, structural plane stress, SRIX against MFront, spectral mechanics and the native NumPy/Numba implementation. P43 is the principal current experimental demonstrator; its data provenance and material conclusions are still being consolidated.

FEMU/SVD defines the main identification and observability framework, but no production boundary-only FEMU driver is currently registered. REGM remains a screening investigation whose observed-DIC transfer is negative. Historical J2 and micromorphic results are retained as baselines and scientific branches rather than presented as validated final conclusions.

Claims and their evidence boundaries are registered in `validation/documentation_evidence_registry.json` and exposed in the documentation.

## Install

TFEL/MFront, MGIS and PyPardiso/MKL are required for the nominal workflow. After installing TFEL/MGIS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,docs]'
source /home/jeff/.local/share/tfel/env/env.sh
cmake -S . -B build/mfront
cmake --build build/mfront
fem-inhouse backend
```

See the [installation guide](docs/how-to/installation/install.md) for supported versions and platform details.

## First reconstruction

```bash
fem-inhouse prepare-case \
  --raw data/raw/case_study \
  --output data/processed/tutorial-10x10 \
  --crop-nx 10 --crop-ny 10 \
  --hardening-scale-mpa 380 \
  --nonfinite-policy nearest \
  --nodal-completion edge-pad-upper

fem-inhouse partition \
  --input data/processed/tutorial-10x10 \
  --output results/tutorial-local \
  --parts-x 1 --parts-y 1 \
  --padding 0 --increments 10 \
  --constitutive-backend mfront-native-plane-stress \
  --partition-id 0
```

The [tutorial](docs/tutorials/first_reconstruction.md) explains what is measured, imposed and reconstructed. The [How-to guides](docs/how-to/index.md) cover backend selection, qualification and production workflows.

## Documentation and quality

The documentation follows the Diátaxis structure:

- [Tutorials](docs/tutorials/index.md) for guided entry points;
- [How-to guides](docs/how-to/index.md) for operational workflows;
- [Explanation](docs/explanation/index.md) for scientific and methodological background;
- [Reference](docs/reference/index.md) for software and numerical contracts;
- [Evidence](docs/evidence/index.md) for registered validation results and claim boundaries.

Run the quality checks below with the virtual environment active. Building `latexpdf` requires `lualatex` and `latexmk` (Debian/Ubuntu packages: `texlive-luatex`, `texlive-latex-extra`, `latexmk`).

```bash
ruff check .
mypy src tests
pytest
make -C docs html latexpdf
```

## Citation and licence

Citation metadata are in [`CITATION.cff`](CITATION.cff). The preferred citation is the associated scientific article listed there, and the provenance of the software includes the original **DIC-based-inHouse-FEM** implementation by Adil Kılınç.

No software licence has been declared; reuse outside the project therefore requires permission.
