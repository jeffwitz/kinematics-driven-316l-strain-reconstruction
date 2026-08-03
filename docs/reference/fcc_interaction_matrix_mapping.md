# FCC interaction matrix: from six published coefficients to seven MFront slots

The sources this project takes its latent-hardening coefficients from state
**six** numbers, one per physical interaction class. MFront takes **seven**.
Neither convention says which of the seven slots each of the six belongs in, and
before this page the correspondence was carried by the order of the numbers in a
single literal and by nothing else.

This page states it, and `fem_inhouse.core.fcc_interaction_matrix` derives it
from the geometry of the slip systems rather than asserting it. The derivation
is compared against `mfront-query --interaction-matrix` on all 144 entries in
`tests/unit/core/test_fcc_interaction_matrix.py`, so the two cannot drift.

## The twelve octahedral systems, in MFront's order

The order matters beyond this page: every per-system array in a result —
`PlasticSlip`, `EquivalentPlasticSlip`, `BackStrain` — is indexed by it, and so
is every row and column below.

| index | Burgers `b` | plane `n` | | index | Burgers `b` | plane `n` |
|---:|---|---|---|---:|---|---|
| 0 | `[0,1,-1]` | `(1,1,1)` | | 6 | `[0,1,-1]` | `(1,-1,-1)` |
| 1 | `[1,0,-1]` | `(1,1,1)` | | 7 | `[1,0,1]` | `(1,-1,-1)` |
| 2 | `[1,-1,0]` | `(1,1,1)` | | 8 | `[1,1,0]` | `(1,-1,-1)` |
| 3 | `[0,1,1]` | `(1,1,-1)` | | 9 | `[0,1,1]` | `(1,-1,1)` |
| 4 | `[1,0,1]` | `(1,1,-1)` | | 10 | `[1,0,-1]` | `(1,-1,1)` |
| 5 | `[1,-1,0]` | `(1,1,-1)` | | 11 | `[1,1,0]` | `(1,-1,1)` |

Source: `mfront-query --slip-systems-by-index`, TFEL 5.1.0. Every direction
satisfies $\mathbf b\cdot\mathbf n = 0$, which is asserted.

## The interaction classes

Two systems $(s, r)$ are classified from their geometry alone.

| class | condition | MFront slot | ordered pairs |
|---|---|---:|---:|
| self | same plane, same Burgers direction | **0** | 12 |
| coplanar | same plane, different direction | **1** | 24 |
| Hirth lock | $\mathbf b_s\cdot\mathbf b_r = 0$ | **2** | 24 |
| Lomer, sessile junction | junction in neither plane | **3** | 24 |
| glissile junction, gliding in the **first** plane | junction lies in $\mathbf n_s$ | **4** | 24 |
| collinear | same Burgers direction, different plane | **5** | 12 |
| glissile junction, gliding in the **second** plane | junction lies in $\mathbf n_r$ | **6** | 24 |

The junction is $\mathbf j = \mathbf b_s \pm \mathbf b_r$, taking whichever sign
gives a $\langle 110\rangle$ vector; it lies in a plane when
$\mathbf j\cdot\mathbf n = 0$.

**Collinear is the strongest interaction** and carries `12.3` in the registered
set, against `0.6` to `1.8` for everything else. The reason is physical: two
dislocations sharing a Burgers direction can annihilate, which no junction does.

## Why seven slots for six coefficients

MFront splits the glissile junction into **two** ranks, 4 and 6, according to
which of the two systems can glide the junction. A glissile junction is by
definition sessile in one of the two planes and glissile in the other, so the
pair $(s, r)$ and the pair $(r, s)$ describe the same junction seen from the two
sides — and MFront gives them different slots.

The consequence is worth stating plainly, because it is easy to miss:

> **The rank matrix is not symmetric.** Entry `(0, 7)` is slot 6 while entry
> `(7, 0)` is slot 4. The *numerical* matrix is symmetric only when both glissile
> slots hold the same number.

The six-coefficient publication convention therefore maps onto MFront by
**writing the single glissile coefficient into both slots**, which is what
`from_publication_coefficients` does and the whole reason it exists rather than a
comment beside a literal:

```python
from fem_inhouse.core.fcc_interaction_matrix import from_publication_coefficients

# self, coplanar, Hirth, Lomer, glissile, collinear
from_publication_coefficients((1.0, 1.0, 0.6, 1.8, 1.6, 12.3))
# -> (1.0, 1.0, 0.6, 1.8, 1.6, 12.3, 1.6)
#                              ^^^        ^^^  the same number, twice
```

Giving slots 4 and 6 *different* values is legal in MFront and produces a
non-symmetric hardening matrix, in which system $s$ hardens $r$ differently from
how $r$ hardens $s$. That is a departure from the convention of the sources, and
`is_symmetric` exists so a parameter set can be checked rather than assumed.

## The registered 12×12

With the coefficients of `316l_srix_transposed_from_nasri2018_rate_1e-3`,
$(1.0,\ 1.0,\ 0.6,\ 1.8,\ 1.6,\ 12.3,\ 1.6)$, the slot matrix is

```text
       0  1  2  3  4  5  6  7  8  9 10 11
  0 |  0  1  1  2  3  4  5  6  6  2  4  3
  1 |  1  0  1  3  2  4  4  2  3  6  5  6
  2 |  1  1  0  6  6  5  4  3  2  3  4  2
  3 |  2  3  4  0  1  1  2  4  3  5  6  6
  4 |  3  2  4  1  0  1  6  5  6  4  2  3
  5 |  6  6  5  1  1  0  3  4  2  4  3  2
  6 |  5  6  6  2  4  3  0  1  1  2  3  4
  7 |  4  2  3  6  5  6  1  0  1  3  2  4
  8 |  4  3  2  3  4  2  1  1  0  6  6  5
  9 |  2  4  3  5  6  6  2  3  4  0  1  1
 10 |  6  5  6  4  2  3  3  2  4  1  0  1
 11 |  3  4  2  4  3  2  6  6  5  1  1  0
```

and substituting the coefficients gives a symmetric matrix with `1.0` on the
diagonal, `12.3` on the twelve collinear pairs, and `0.6`, `1.6` or `1.8`
elsewhere. Read the diagonal band structure as a check: systems `0-2`, `3-5`,
`6-8` and `9-11` are the four `{111}` families, and each 3×3 diagonal block is
the coplanar block, `1` off its own diagonal.

## Comparison with the TFEL gallery example

The gallery behaviour `MericCailletaudSingleCrystalViscoPlasticity`, shipped with
TFEL 5.1.0 at
`share/doc/tfel/web/gallery/viscoplasticity/MericCailletaudSingleCrystalViscoPlasticity.mfront`,
declares

```text
@InteractionMatrix{1, 1, 0.6, 1.8, 1.6, 12.3, 1.6};
```

which is **identical to ours**, slot by slot, and identical again in the
finite-strain and numerical-Jacobian variants of the same gallery.

That is a corroboration and not a proof: the gallery is a worked example rather
than a normative statement of the ordering, and agreeing with it would not have
saved a project that had the classes wrong in the same way. What establishes the
placement here is the derivation from geometry checked against
`mfront-query --interaction-matrix` on this exact file. The gallery agreeing
independently is simply the reassurance that the widely-copied literal and this
project's literal mean the same thing.

The practical rule is unchanged: a reader importing coefficients from a paper, or
from another code's example, must map them through `from_publication_coefficients`
rather than by position. Position happens to work here; it is not guaranteed to.

## References

- B. Devincre, T. Hoc and L. Kubin, *Dislocation mean free paths and strain
  hardening of crystals*, Science **320**(5884), 1745–1748, 2008.
  DOI [10.1126/science.1156101](https://doi.org/10.1126/science.1156101).
  The interaction classes and their relative strengths for FCC.
- P. Franciosi, M. Berveiller and A. Zaoui, *Latent hardening in copper and
  aluminium single crystals*, Acta Metallurgica **28**(3), 273–283, 1980.
  DOI [10.1016/0001-6160(80)90162-5](https://doi.org/10.1016/0001-6160(80)90162-5).
  The origin of the latent-hardening matrix and of the six-class description.
- L. Méric and G. Cailletaud, *Single crystal modeling for structural
  calculations. Part 2: finite element implementation*, Journal of Engineering
  Materials and Technology **113**(1), 171–182, 1991.
  DOI [10.1115/1.2903375](https://doi.org/10.1115/1.2903375).
  The hardening law the coefficients are stated for.
- M. A. Nasri et al., Comptes Rendus Mécanique **346**, 132–151, 2018.
  DOI [10.1016/j.crme.2017.11.009](https://doi.org/10.1016/j.crme.2017.11.009).
  Source of the numerical values used here.
- TFEL/MFront, *Méric-Cailletaud single crystal plasticity*:
  <https://thelfer.github.io/tfel/web/MericCailletaudSingleCrystalPlasticity.html>.
- `mfront-query --slip-systems-by-index`, `--interaction-matrix-structure` and
  `--interaction-matrix`, TFEL 5.1.0, run on
  `mfront/Fcc316LForestRubinSrix.mfront`. These are the authority for the
  ordering, and their output is archived in the test.
