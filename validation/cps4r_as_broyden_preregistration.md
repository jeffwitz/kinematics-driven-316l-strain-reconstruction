# Limited-memory Broyden for CPS4R-AS — preregistration

Date: 2026-08-04
Written before any Broyden-accelerated solve exists. The reduced coordinates and
the multisecant algebra are implemented and unit-tested; the solver is untouched.

## What is being accelerated, and what must not move

`assumed_strain_energy` with the ASMD projection, exactly as it stands. Its
residual, its stabilising force, its stresses, its slips and its converged
solution are **not** subjects of this work. Only the matrix used to obtain the
Newton correction changes.

The defect is measured, not supposed: the physical element tangent is consistent
to `1.9e-6`, the stabilisation tangent is wrong by `370 %`, the total by `36 %`,
because `f_stab(u, C(u))` is differentiated holding `C` fixed. The consequence is
47 Newton iterations against CPS4's 37, and that is the whole of the
constitutive speed-up shortfall.

## Thresholds, frozen

Against `assumed_strain_energy` **without** Broyden, on the registered SRIX case
— Bunge (35, 20, 15), 12x12, non-affine boundary, eight increments, five
repetitions:

| criterion | bound |
|---|---|
| `E_u`, `E_sigma`, `E_Gamma` against the un-accelerated run | **< 1e-6** |
| errors against CPS4 | unchanged within **0.02 point** |
| Newton iterations | **<= 40**, a reduction of at least 15 % |
| additional cutbacks | **none** |
| constitutive speed-up | **> 3.5** |
| total speed-up | **> 1.8** |
| additional constitutive calls | **zero** |

Memory is swept over `1, 3, 5` and over nothing else. A memory that wins only on
the homogeneous case does not become the default.

## Falsifiers

F1 the residual or converged solution moves. F2 a constitutive call is added.
F3 iterations increase. F4 a cutback appears where the un-accelerated run has
none. F5 the correction puts force on a rigid mode. F6 the reduced matrix goes
singular without a clean fallback to `K_0`. F7 the gain needs a memory chosen
after seeing results. F8 the assembly cost eats the constitutive gain.

## A tension in the specification, flagged before it can be argued about

Section 8 states that the `H^T dG T` construction guarantees an affine field is
unmodified, and section 22.8 asks for no extra stabilising force on one.
**Measured: `K_B u_affine` is not zero.**

This is not a construction defect. `T = [B_c; H]`; an affine field has zero
hourglass amplitudes but a non-zero central strain, and the term being learned is
exactly `(df_stab/dC)(dC/du)` with `C` moving with the central strain. A
correction annihilating affine directions could not learn the thing it exists
for.

What is true and is what the patch test needs: **the residual is untouched**, and
at an affine state the amplitudes are zero so the stabilising force is zero. The
correction changes the matrix, never the force. Rigid modes *are* in the kernel,
structurally, and that is verified.

Reading 22.8 as a statement about the force rather than about the matrix is
therefore proposed, and recorded here rather than settled by zeroing three
columns of `dG`.

## What is implemented, and what is not

Implemented and tested: the reduced coordinates `T`, `H`, `L`, `T^+`, `G_0` and
the `H^T dG T` expansion; the modal projection check at `1e-16` against a `1e-10`
bound; the circular per-element memory with normalisation by `||s||`; the
minimum-Frobenius multisecant solve `dG = Z S^+` through an SVD with rank
detection; the deterministic fallback to a zero correction on any degeneracy.
Secant conditions are satisfied to `1e-12` for one to five directions, colinear
directions reduce the rank to one without producing a large correction, and the
returned solution is verified to be the minimum-norm one.

**Not implemented: the solver wiring.** No Newton iteration uses this yet, so no
criterion above has been evaluated and no verdict of section 35 is available.
