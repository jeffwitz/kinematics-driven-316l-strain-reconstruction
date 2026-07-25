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
in [`Claude.md`](Claude.md). The English user documentation starts at
[`docs/index.rst`](docs/index.rst) and follows the Diátaxis structure:
a first reconstruction tutorial, task-oriented how-to guides, precise
reference contracts, and scientific explanations.

An article-sized corner partition (`510×460`, 234,600 elements) has now been
run with the default analytical MFront law directly from the versioned DIC
inputs and the article's 100-partition, 150-pixel-padding layout. It converged
in 20 increments without cutback in `650.08 s` wall time. All six raw fields,
logs, hashes, resource measurements, derived maps and the comparison with the
historical tabulated Python run are preserved under
[`validation/reference_data/article_100p_pad150_p0000_mfront_v1`](validation/reference_data/article_100p_pad150_p0000_mfront_v1).

The default constitutive backend is MFront 5.1.0/MGIS 3.1, compiled for the
same plane-stress J2/Ludwik material. It is connected to the
finite-element Newton loop with trial/commit/revert state transactions and its
consistent tangent is assembled at every Gauss point. Material-point tests and
the saved DIC-driven `10×10` comparison both pass their declared thresholds.
The same law is now also compiled as a genuine six-component
`Tridimensional` behaviour. The experimental
`mfront-3d-condensed-plane-stress` backend solves locally for
`[epsilon33, gamma13, gamma23]` and passes an explicit Schur-complement tangent
to the unchanged 2D solver. The global Newton loop now depends only on a common
plane-stress material protocol, so a future 3D crystal-plasticity behaviour can
replace J2 inside this adapter without changing the FEM kernel.

Every converged final state now also exposes complete symmetric `3×3` stress,
total-strain, elastic-strain, and plastic-strain tensors. This is an
output-only completion of the existing 2D plane-stress state: the mesh,
unknowns, Newton system, constitutive updates, and in-plane results are
unchanged. MFront retains its native `AxialStrain`, `ElasticStrain`, and
numerical `S33` residual.
Installation, tensor conventions, exact metrics, and reproduction commands are
documented in [`docs/mfront.md`](docs/mfront.md).

The complete-tensor DIC campaign is preserved under
[`validation/reference_data/plane_stress_tensor_reconstruction_dic_10x10_v1`](validation/reference_data/plane_stress_tensor_reconstruction_dic_10x10_v1).
Its maximum native MFront `|S33|` is `1.046e-14 MPa`, maximum plastic-trace
residual is `1.406e-19`, and maximum additive-decomposition residual is
`1.355e-19`. Regression against the earlier artifacts confirms that historical
2D outputs changed by at most `4.263e-14 MPa` (MFront round-off; Python is
identical).

The native-versus-condensed J2 campaign is preserved under
[`validation/reference_data/mfront_3d_condensed_dic_10x10_v1`](validation/reference_data/mfront_3d_condensed_dic_10x10_v1).
Both paths converge the same DIC 10×10 case in 66 global Newton iterations
without cutback. Their maximum in-plane stress difference is
`4.804e-08 MPa`; the condensed path reaches a maximum Gauss-point transverse
residual of `2.705e-08 MPa` in at most four local iterations, with no local
failure.

All three backends were also timed in fresh processes on the same central
`100×100` DIC crop, with 20 increments and three repetitions per backend.
Median complete-process wall times are `134.36 s` for Python, `27.03 s` for
native MFront plane stress, and `83.43 s` for condensed 3D MFront. Median peak
RSS values are respectively `248.96`, `269.65`, and `320.30 MiB`. All nine
runs converge without cutback. Native and condensed MFront agree to
`2.307e-07 MPa` maximum stress difference; the independent Python
implementation agrees within the declared case-study tolerances, with a
maximum stress difference of `6.763e-02 MPa`. The native path therefore
remains the production default for the present J2 law, while the condensed
path is the validated extension point for a future three-dimensional law.
Every raw run and full result field is preserved under
[`validation/reference_data/plane_stress_backend_performance_100x100_v1`](validation/reference_data/plane_stress_backend_performance_100x100_v1).

The output-only Helmholtz diagnostic has now been repeated with a
pre-registered selection/confirmation design. On representative partition 48,
`58.88 µm` reduces RMSE by 64.61%, raises correlation from `0.2983` to
`0.6160`, and raises top-10% IoU from `0.1598` to `0.2822`. Applied unchanged
to held-out partition 42, it reduces relative L2 by 65.43%, raises correlation
from `0.4007` to `0.7036`, and passes the declared equal-fraction and absolute
DIC-threshold localization criteria. The stage-1 conclusion is therefore
**spatial-width hypothesis supported**, while `58.88 µm` remains a diagnostic
scale rather than an identified material internal length. The selection and
confirmation campaigns are preserved under
[`validation/reference_data/nonlocality_helmholtz_article_p0048_v1`](validation/reference_data/nonlocality_helmholtz_article_p0048_v1)
and
[`validation/reference_data/nonlocality_helmholtz_article_p0042_confirmatory_v1`](validation/reference_data/nonlocality_helmholtz_article_p0042_confirmatory_v1);
the method and command are documented in
[`docs/explanation/nonlocality_diagnostic.md`](docs/explanation/nonlocality_diagnostic.md)
and
[`docs/how-to/diagnose_nonlocality.md`](docs/how-to/diagnose_nonlocality.md).

The next pre-registered stage is now implemented at software level. Two new
MFront behaviours add the energetic micromorphic correction
`Hchi * (PEEQ - chi)` while preserving the reference behaviours. The existing
element-centred Helmholtz solver computes `chi` inside every mechanical Newton
trial; MGIS state remains transactional and is committed only after global
convergence. Both native plane stress and condensed 3D adapters are supported.
Five optional fields preserve the nonlocal state, mismatch, hardening
correction, yield radius, and final coupling residual. With `Hchi=0`, a
complete-Newton regression reproduces the local MFront solution to the declared
`1e-10` relative tolerance.

The development ROI is P154 in a `20×20` layout. Its retained core is
`180×155`; its validation domain is `436×411` with 128 pixels of padding,
exactly four times the candidate length of `58.88 µm`. The protocol is frozen
in
[`validation/nonlocal_p154_preregistration.md`](validation/nonlocal_p154_preregistration.md).
The model and operational sequence are documented in
[`docs/explanation/micromorphic_plasticity.md`](docs/explanation/micromorphic_plasticity.md)
and
[`docs/how-to/run_coupled_nonlocal_p154.md`](docs/how-to/run_coupled_nonlocal_p154.md).
The `validate-coupled-nonlocal` command compares the raw coupled displacement
solution with DIC on the manifest-declared core and evaluates the
pre-registered P154 criteria without post-filtering. Scientific P154 results
are recorded in
[`validation/nonlocal_p154_validation_results.md`](validation/nonlocal_p154_validation_results.md).
All positive candidates converged without cutback. The best tested candidate,
`alpha=2` (`Hchi=13095.061233 MPa`), improves correlation by `0.1643`,
reduces relative L2 by `42.17%`, and raises top-10% IoU by `0.0331`.
It nevertheless predicts `21.85%` active area at the DIC-q90 threshold,
outside the pre-registered `[5%, 20%]` interval. The defensible conclusion is
therefore **coupled spatial interaction partially supported**; no coupling
modulus is frozen for transfer.

The reproducible P154 alpha visualisation is documented in
[`docs/how-to/run_coupled_nonlocal_p154.md`](docs/how-to/run_coupled_nonlocal_p154.md)
and is available as `fem-inhouse plot-coupled-alpha-fields`. It compares the
raw EVM reconstructed from DIC/FEM displacements for alpha = 0, 0.5, 1, and 2,
plus the internal PEEQ redistribution. No Helmholtz filter is applied to the
primary FEM EVM figures; `plot_metadata.json` records the hashes, limits, and
plotting choices.

Because P154 is too homogeneous to identify a coupling length robustly, the
repository now provides `fem-inhouse select-dic-partition`. It ranks the full
10 x 10 DIC decomposition using the morphology of coherent high-strain bands:
elongation, occupied area, contrast, continuity, and boundary contacts.
Distribution-only indicators such as kurtosis remain diagnostics but no longer
drive the selection. The automated score ranks P17 first, but the required
visual review showed that P43 `(4, 3)` is the better scientific calibration
ROI: its core contains two distinct diagonal deformation bands rather than one
component selected mainly by the ranking heuristic. P43 is therefore the
registered target for the next coupled campaign. No P43 alpha sweep has been
launched yet.

Before that campaign, the native MFront micromorphic hot path was made
lightweight. Fixed-point evaluations now request only PEEQ and do not compute
a tangent or reconstruct full 3D tensors. One tangent evaluation is performed
after fixed-point convergence, and full tensors are completed only for the
final converged FEM state. On the complete `360 x 310` P43 core this reduces
the constitutive benchmark from `14.36 s` to `7.61 s` (`1.89x`) and peak RSS
from `796,856` to `564,508 KiB` (`-29.2%`), with bit-identical stress, tangent,
PEEQ, and nonlocal fields. A complete FEM benchmark on a real band-containing
intermediate crop reduces wall time from `396.78 s` to `273.56 s` (`1.45x`)
and peak RSS by `12.7%`, while preserving convergence decisions exactly and
all physical fields below the declared `1e-10` relative threshold. The
versioned evidence is reported in
[`validation/performance/nonlocal_hot_path_optimization.json`](validation/performance/nonlocal_hot_path_optimization.json).
The following fixed-CSR/PARDISO phase optimization reduces the same complete
P187 gate from `273.56 s` to `244.67 s` (`-10.6%`) and peak RSS by `16.7%`.
Sparse assembly falls by `73.4%`, free-system extraction by `99.4%`, and
PARDISO time by `48.3%`. The solver performs phase 11 once and 139 phase 22/33
pairs while retaining nonsymmetric `mtype=11`; Newton, the fixed point, and the
tangent are unchanged. Evidence is preserved in
[`validation/performance/fixed_csr_explicit_pardiso_p0187.json`](validation/performance/fixed_csr_explicit_pardiso_p0187.json).
The verified J2 behaviours now declare their tangent symmetric positive
definite. Their fixed free-system CSR stores only the upper triangle and
PARDISO uses `mtype=2` by default. On the identical P187 gate this reduces
wall time again from `244.67 s` to `227.34 s` (`-7.1%`), PARDISO time by
`38.0%`, factorization time by `46.2%`, and peak RSS by `8.7%`. The largest
measured relative tangent asymmetry is `6.46e-16`, against a runtime rejection
threshold of `1e-12`. Unclassified behaviours retain full CSR storage and
`mtype=11`; a future crystal-plasticity adapter therefore remains
nonsymmetric until its own tangent contract has been verified. Evidence is in
[`validation/performance/symmetric_spd_pardiso_p0187.json`](validation/performance/symmetric_spd_pardiso_p0187.json).

On a one-minute constitutive benchmark (200,000 points, 20 increments, two
repetitions), the eight-thread MGIS backend is 3.50× faster than the current
Python update; MFront serial is 8.0% slower. This excludes assembly and
PyPardiso. Raw timings and final states are preserved under
[`validation/reference_data/mfront_performance_v1`](validation/reference_data/mfront_performance_v1).
On the complete article-sized partition, MFront reduces process wall time by
40.35% (`1089.80 → 650.08 s`) and constitutive time by a factor of 6.90. The
measured full-process peak RSS nevertheless increases by 10.49%
(`3,768,132 → 4,163,308 KiB`): removing the 1000-point table does not imply a
lower process peak because MGIS state/tangent storage and the sparse FEM
working set dominate this measurement.

Known limitations at this stage:

- the complete 11.16-million-element ROI has not yet been executed and
  stitched; the largest interior padded partitions are also larger than the
  completed corner partition;
- only DIC step 40 is available; the baseline steps 1–5 are not versioned;
- Abaqus parity is not yet established from the original `.inp` and ODB
  extraction scripts and is intentionally deferred until the DIC-first
  workflow is stable.

## Documentation

The documentation is written in English and uses the Read the Docs theme.
Install its pinned dependencies and build the strict HTML and PDF outputs with:

```bash
.venv/bin/pip install -r requirements-docs.txt
PATH="$PWD/.venv/bin:$PATH" make -C docs html
PATH="$PWD/.venv/bin:$PATH" make -C docs latexpdf
```

The generated entry points are
`docs/_build/html/index.html` and
`docs/_build/latex/kinematics-driven-316l-strain-reconstruction.pdf`.
The figures used in the documentation are reproducibly generated as SVG and
PDF pairs by:

```bash
.venv/bin/python scripts/build_documentation_figures.py
```

Read the tutorial first for a guided DIC-driven calculation, use the how-to
guides for production tasks, consult the reference section for exact
interfaces, and use the explanation section for the scientific and numerical
rationale. Read the Docs is configured by [`.readthedocs.yaml`](.readthedocs.yaml)
to publish both HTML and PDF.

## Reproduce from the versioned DIC data

The four raw scientific arrays are versioned with Git LFS under
[`data/raw/case_study`](data/raw/case_study). From a fresh clone:

```bash
git lfs install
git lfs pull
python -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -e . --no-deps

.venv/bin/fem-inhouse prepare-case \
  --raw data/raw/case_study \
  --output data/processed/case-study \
  --nonfinite-policy nearest
```

This command verifies every raw SHA-256, maps `V → u_x` and `U → u_y`,
converts pixels to millimetres, applies the article's nominal `K=380 MPa`,
repairs the nine declared non-finite hardening multipliers, completes the nodal
grid and writes a manifest for all generated arrays.

A fast smoke calculation uses a real central `10×10` crop:

```bash
bash examples/run_dic_smoke.sh
```

The script is equivalent to:

```bash
.venv/bin/fem-inhouse prepare-case \
  --raw data/raw/case_study \
  --output data/processed/case-study-10x10 \
  --crop-nx 10 \
  --crop-ny 10

.venv/bin/fem-inhouse partition \
  --input data/processed/case-study-10x10 \
  --output results/dic-smoke-10x10 \
  --count 25 \
  --padding 0 \
  --increments 10 \
  --solve-pending
```

Every partition preserves the historical final fields (`U`, `S`, `E`, `PE`,
`PEEQ`, `RF`) and the additional complete fields (`S_3D`, `E_3D`, `EE_3D`,
`PE_3D`, `PLANE_STRESS_RESIDUAL_MPA`, `S33_RESIDUAL_MPA`) together with
convergence diagnostics and output hashes. The vector residual is ordered
`[S33, S13, S23]`; the scalar field remains its first component for backward
compatibility. The full production run uses the same prepared contract, with
padding and partition execution distributed according to the available memory. See
[`docs/from_dic_to_reconstruction.md`](docs/from_dic_to_reconstruction.md).

## Development setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -e . --no-deps
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/mypy src/fem_inhouse
```

PyPardiso/MKL and the source-built MFront/MGIS backend are required for default
production solves. The historical Python return mapping remains explicitly
selectable for regression and Abaqus-table reproduction. See
[`docs/mfront.md`](docs/mfront.md) for the pinned installation and comparison.
After building the behaviour, partition solves select it with
`--constitutive-backend mfront --mfront-library`
`build/mfront/src/libBehaviour.so --mfront-threads N`.
Use `--constitutive-backend mfront-3d-condensed-plane-stress` to exercise the
experimental six-component condensation path.

The installed CLI provides the routine entry points:

```bash
fem-inhouse backend
fem-inhouse validate --nx 10 --ny 10
fem-inhouse example --nx 10 --ny 10 --output results/reduced
fem-inhouse prepare-case --help
fem-inhouse partition --help
fem-inhouse compare-fields --help
fem-inhouse diagnose-nonlocality --help
fem-inhouse estimate-nonlocal-reference --help
fem-inhouse validate-coupled-nonlocal --help
fem-inhouse plot-coupled-alpha-fields --help
```

See [`docs/reduced_example.md`](docs/reduced_example.md) for the interpretation
of the example and its declared thresholds.

The historical Abaqus generators are kept byte-for-byte under
[`references/legacy_abaqus`](references/legacy_abaqus) solely for scientific
provenance. The production package neither imports nor executes them.
Initial PyPardiso timing and memory measurements are recorded in
[`docs/performance.md`](docs/performance.md).
The resumable CLI and Slurm-array workflow are documented in
[`docs/partitioning.md`](docs/partitioning.md).
Architecture decisions are recorded in [`docs/adr`](docs/adr), and numerical
review requirements are defined in [`CONTRIBUTING.md`](CONTRIBUTING.md).
The raw-to-canonical choices are recorded specifically in
[`ADR 0004`](docs/adr/0004-dic-input-preparation.md).

## Typed solver API

```python
import numpy as np

from fem_inhouse import CaseStudyConfig, MeshConfig, run_case_study

mesh = MeshConfig(nx=20, ny=20)
config = CaseStudyConfig(mesh)
shape_nodes = (mesh.nx + 1, mesh.ny + 1)
shape_elements = (mesh.nx, mesh.ny)

result = run_case_study(
    config,
    displacement_x_mm=np.zeros(shape_nodes),
    displacement_y_mm=np.zeros(shape_nodes),
    yield_stress_mpa=np.full(shape_elements, 250.0),
    hardening_coefficient_mpa=np.full(shape_elements, 500.0),
)
print(result.equivalent_plastic_strain.max())
print(result.stress_tensor_mpa.shape)
print(result.total_strain_tensor[..., 2, 2].min())
print(np.abs(result.plane_stress_residual_mpa).max())
print(np.abs(result.plane_stress_residual_vector_mpa).max())
print(result.diagnostics)
```

`result.diagnostics` records the backend, timings, converged increments,
cutbacks, Newton iterations, final convergence criterion, and whether the
complete tensors came from native MFront plane stress, analytical J2
completion, or local condensation of a 3D law. The condensed backend also
records local iteration counts, failures, the maximum Gauss-point residual,
and the maximum condition number of `Cbb`. See
[`docs/explanation/plane_stress_tensors.md`](docs/explanation/plane_stress_tensors.md)
for result completion and
[`docs/explanation/mfront_3d_condensation.md`](docs/explanation/mfront_3d_condensation.md)
for the constitutive architecture.

The top-level `fem_pixel.py` file remains only as a compatibility entry point
for existing case-study scripts.

## Citation and licensing

Citation metadata and the associated article authors are recorded in
[`CITATION.cff`](CITATION.cff). The repository does not yet declare a software
license: that legal choice remains an explicit project-owner decision and must
be resolved before a public release.
