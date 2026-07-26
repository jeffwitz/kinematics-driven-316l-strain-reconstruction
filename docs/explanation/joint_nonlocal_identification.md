# Joint identification of the nonlocal parameters

## Scientific question

The micromorphic model contains a coupling modulus \(H_\chi\) and a spatial
length \(\ell\):

\[
R(p,\chi)=R_\mathrm{local}(p)+H_\chi(p-\chi),
\qquad
\chi-\ell^2\Delta\chi=p.
\]

The purpose of the identification workflow is to determine whether these two
parameters can be distinguished by the available DIC observations. It does
not identify \(\ell\) as a material length. That interpretation requires an
unchanged-parameter validation on another region, another DIC resolution and,
ideally, another loading history.

## Reused numerical architecture

The workflow is deliberately a thin orchestration layer around the validated
project components:

1. the element-centred DCT Helmholtz solver computes \(\chi\);
2. the historical EVM reconstruction applies the same strain operator to DIC
   and FEM displacements;
3. the existing field metrics operate on the partition core;
4. the existing `PartitionWorkflow` runs every reduced mechanical case;
5. its immutable manifests, atomic outputs, MFront transactions, Newton
   iterations, cutbacks and PARDISO path remain unchanged.

Consequently, F1 is not a competing finite-element pipeline. It prepares a
coarser but physically coextensive input dataset and delegates the solve to
the production workflow.

## Three fidelity levels

### F0: frozen-field screen

F0 starts from the converged local PEEQ field \(p_0\). For every candidate
length it computes

\[
\chi_\ell=\mathcal H_\ell(p_0),\qquad r_\ell=p_0-\chi_\ell.
\]

The DCT is evaluated once per length. All coupling moduli then reuse the same
residual through \(q_\mathrm{nl}=H_\chi r_\ell\). This makes a dense screen
cheap enough to expose weak, excessive or nearly equivalent parameter pairs.
Because plasticity is not reintegrated, F0 is an identifiability heuristic,
not a mechanical prediction.

### F1: reduced mechanical ranking

F1 replays the loading history from the initial state on a grid reduced by a
configurable integer factor. Element properties are area-averaged, nodal DIC
displacements are sampled on the coincident coarse nodes, the physical extent
and core/padding split are preserved, and the existing coupled solver is run
unchanged. A point is excluded when

\[
\ell/h_\mathrm{F1}<3.
\]

F1 may rank candidates only after its ordering and error have been checked
against existing F2 points.

### F2: scientific calculation

F2 is the unchanged full-resolution calculation. The workflow may generate a
manifest containing at most five proposed F2 runs, including
\(\ell=58.88\,\mu\mathrm m,\alpha=6\), but it never launches them
implicitly. Human approval is a hard gate.

## Parameter coordinates and expected degeneracy

The public parameters are

\[
\alpha=H_\chi/H_\mathrm{ref},\qquad \ell,
\]

where \(H_\mathrm{ref}\) is read from the material or campaign metadata. The
workflow also records

\[
A_\chi=H_\chi\ell^2
\]

in \(\mathrm{MPa\,mm^2}\), and uses
\(\theta_H=\log H_\chi\), \(\theta_A=\log A_\chi\) for interpolation.
For \(\ell k\ll1\), the spectral coupling behaves approximately as
\(H_\chi\ell^2k^2\). A cost valley following constant \(A_\chi\) is therefore
expected when the DIC does not resolve frequencies around \(k\approx1/\ell\).

The local case is stored once as `alpha = 0`; its length, \(A_\chi\), and
logarithmic coordinates are explicitly undefined.

## Observation contract

All DIC comparisons pass through a recorded observation operator
\(\mathcal M_\mathrm{DIC}\). It specifies the grid transformation, spatial
averaging, finite-value mask, core selection, strain convention and any
missing-value policy. The primary comparison is total equivalent strain
reconstructed from displacement. PEEQ is never presented as an experimental
observable.

## Selection without a hidden scalar objective

Amplitude and localization remain separate objectives. The workflow reports
global errors, quantile-ratio amplitude errors, relative top-10 overlap,
absolute DIC-q90 overlap, gradient measures and spatial spectra. It constructs
the non-dominated front in at least
\((J_\mathrm{amp},J_\mathrm{loc})\), labels each fidelity level, and reports
the knee separately from the best-amplitude and best-localization points.

No material-length conclusion is allowed until selected candidates are
transferred to another band-containing ROI without recalibration.
