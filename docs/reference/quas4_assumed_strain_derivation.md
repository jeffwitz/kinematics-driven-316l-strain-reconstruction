# QUAS4 and the Assumed Strain projections: derivation from the sources

Section 11 of the 2026-08-04 specification. This page reconstructs the
formulation from the primary documents rather than restating it, and states
every convention explicitly, because the published formulae are written in
conventions that are not this project's.

## The sources, and how they were obtained

The reference document is

> **Code_Aster R3.06.10**, *Élément quadrangulaire à un point d'intégration,
> stabilisé par la méthode « Assumed Strain »*, responsible Sébastien Fayolle,
> dated 25/02/2014, revision 12047, fascicule r3.06, GNU FDL.
> Original text by N. Tardieu and S. Limouzi, EDF R&D/AMA, for Aster 7.2.

`code-aster.org` returns 404 for every path to this file today; the copy used
here came from the UQTR mirror,
`https://ericca.uqtr.ca/fr12.4/man_r/r3/r3.06.10.pdf`. **The 13.6 mirror is
truncated at 19 of 23 pages and loses the bibliography** — a reader reproducing
this page should use the 12.4 copy.

Its four references, with DOIs checked against Crossref rather than transcribed:

- J. O. Hallquist, *Theoretical manual for DYNA3D*, UCID-19401, Lawrence
  Livermore National Laboratory, 1983. Source of the one-point gradient
  operator. R3.06.10 prints the report number as "UC1D-19401"; LLNL numbers this
  series UCID.
- T. Belytschko and W. E. Bachrach, *Efficient implementation of quadrilaterals
  with high coarse-mesh accuracy*, Computer Methods in Applied Mechanics and
  Engineering **54**(3), 279–301, 1986.
  DOI [10.1016/0045-7825(86)90107-6](https://doi.org/10.1016/0045-7825(86)90107-6).
  Source of the enriched displacement field. **R3.06.10 prints volume 43;
  Crossref gives volume 54** for the same title, authors, pages and year.
- T. Belytschko and L. P. Bindeman, *Assumed strain stabilization of the 4-node
  quadrilateral with 1-point quadrature for nonlinear problems*, Computer
  Methods in Applied Mechanics and Engineering **88**, 311–340, 1991.
  DOI [10.1016/0045-7825(91)90093-L](https://doi.org/10.1016/0045-7825(91)90093-L).
  **The primary source of the assumed-strain field and of the projections.**
- J.-L. Batoz and G. Dhatt, *Modélisation des structures par éléments finis,
  volume 2*, Hermès, 1990.

## What QUAS4 is

A four-node quadrilateral integrated at **one** point, the centre `(0,0)`, with
weight `4` — the same parent-area convention this project already uses for
`cps4r`. Code_Aster activates it through the `C_PLAN_SI` and `D_PLAN_SI`
modellings, small strain only.

The element carries **two families of Gauss points**: the single central point,
where the constitutive law is integrated and where the stresses, internal
variables and tangent live; and a second family identical to the classical
`QUAD4`'s four points, used **only** to build the stabilisation matrix. That is
exactly the split section 5 of the specification requires, and it is the
source's own architecture, not an adaptation of it.

## The central operator

With `b_x` and `b_y` the shape-function derivatives at the centre, R3.06.10
éq 2.1-1 writes the discretised gradient operator in the engineering-shear
convention, one 2-column block per node:

$$B_c=\begin{bmatrix} b_x^{t} & 0\\ 0 & b_y^{t}\\ b_y^{t} & b_x^{t}\end{bmatrix}$$

and gives them in closed form, éq 2.1-3, as constants over the element:

$$b_x^{t}=\frac{1}{2A}\big[(y_2-y_4),\,(y_3-y_1),\,(y_4-y_2),\,(y_1-y_3)\big]$$
$$b_y^{t}=\frac{1}{2A}\big[(x_4-x_2),\,(x_1-x_3),\,(x_2-x_4),\,(x_3-x_1)\big]$$

with $A=\tfrac12\big[(x_3-x_1)(y_4-y_2)+(x_2-x_4)(y_3-y_1)\big]$ the element
area. These are Hallquist's operators. The unstabilised stiffness and internal
force are éq 2.1-4 and 2.1-5:

$$K_c=A\,B_c^{T}C\,B_c,\qquad f_{\text{int}}=A\,B_c^{T}\sigma_c .$$

**The area factor is missing from the printed éq 2.1-5**, which reads
`F_int = K_e·U = B_c^T·σ_c`. It has to be there dimensionally, and éq 3.3-4 puts
it in the stiffness; this page uses $A\,B_c^{T}\sigma_c$.

**`C` is the current constitutive tangent, not an elastic reference.** R3.06.10
is explicit: *"soit la matrice de comportement élastique pour les calculs en
élasticité, soit la matrice tangente pour les calculs en plasticité. Notons
qu'au cours de tels calculs, c'est l'intégration de la loi de comportement au
point de Gauss (au centre dans notre cas) qui détermine la valeur des
coefficients de `C`."* This is the single point on which the project's existing
`cps4r` departs from QUAS4, and it is the reason the specification asks for a
new formulation.

## Why one point fails, and what has to be restored

At the centre alone, `K_e` is singular: its kernel has dimension **five**, the
three rigid-body modes plus **two hourglass modes**. R3.06.10 §2.2 and §2.3 put
it physically — on an hourglass mode the strain at the centre is exactly zero,
in agreement with beam theory in pure bending, so the element cannot tell a
deformed state from an undeformed one.

## The enriched operator

From Belytschko and Bachrach, the displacement field is written (éq 3.2-1)

$$u_i=\big(\Delta^{T}+x\,b_x^{T}+y\,b_y^{T}+h\,\gamma^{T}\big)\,u_i$$

giving (éq 3.2-2)

$$B=B_c+B_n=\begin{bmatrix}
b_x^{t}+h_{,x}\gamma^{t} & 0\\
0 & b_y^{t}+h_{,y}\gamma^{t}\\
b_y^{t}+h_{,y}\gamma^{t} & b_x^{t}+h_{,x}\gamma^{t}
\end{bmatrix}$$

with $h=\xi\eta$, and

$$\Delta=\tfrac14\big[t-(t^{T}x)b_x-(t^{T}y)b_y\big],\qquad
\gamma=\tfrac14\big[h-(h^{T}x)b_x-(h^{T}y)b_y\big].$$

Here $t=(1,1,1,1)^{T}$ and $h=(1,-1,1,-1)^{T}$ are the nodal values of the
constant and hourglass patterns. **$\gamma$ is the hourglass pattern purged of
its linear content**, which is what makes it orthogonal to every affine field —
the property the patch test needs. R3.06.10 also gives it in closed nodal form,
éq 3.2-3:

$$\gamma=\tfrac14\begin{bmatrix}
x_2(y_3-y_4)+x_3(y_4-y_2)+x_4(y_2-y_3)\\
x_3(y_1-y_4)+x_4(y_3-y_1)+x_1(y_4-y_3)\\
x_4(y_1-y_2)+x_1(y_2-y_4)+x_2(y_4-y_1)\\
x_1(y_3-y_2)+x_2(y_1-y_3)+x_3(y_2-y_1)
\end{bmatrix}$$

Written this way, `B` is *equivalent to the QUAD4 operator*: the enrichment is
not an approximation, it is a regrouping that separates the terms integrated at
the centre from the terms that will be stabilised. Improving the element then
means acting on the second group only.

## The assumed strain field, and the projections

Belytschko and Bindeman postulate (éq 3.3-1)

$$\boldsymbol\varepsilon^{\text{as}}=\begin{bmatrix}
\varepsilon_{x}^{c}+q_x e_1 h_{,x}+q_y e_2 h_{,y}\\
\varepsilon_{y}^{c}+q_x e_2 h_{,x}+q_y e_1 h_{,y}\\
2\varepsilon_{xy}^{c}+q_x e_3 h_{,y}+q_y e_3 h_{,x}
\end{bmatrix},
\qquad q_x=\gamma\cdot u_x,\quad q_y=\gamma\cdot u_y$$

where $q_x,q_y$ are the two **hourglass amplitudes** of the element. A choice of
$(e_1,e_2,e_3)$ *is* a choice of element:

| element | $e_1$ | $e_2$ | $e_3$ |
|---|---:|---:|---:|
| QUAD4 | 1 | 0 | 1 |
| ASMD | 1/2 | −1/2 | 1 |
| **ASBQI** | 1 | **−ν̄** | 0 |
| **ASOI** | 1 | −1 | 0 |
| **ASOI(1/2)** | 1/2 | −1/2 | 0 |

Two things are decided by this table.

**The shear row.** The *OI family* sets $e_3=0$: the hourglass contribution to
the shear strain is **cancelled**. That is the cure for the shear locking of
QUAD4 in bending, which §1 of R3.06.10 diagnoses as excessive stiffness from the
shear terms of the discretised gradient.

**ASMD keeps $e_3=1$.** It is not an OI variant: it acts on the two normal rows
only, leaving the shear untouched. So "every assumed-strain variant cancels the
shear" is false, and matters here because ASMD is the default of this project —
it buys frame invariance (below) rather than the bending cure.

**The coupling between the two normal rows.** $e_2$ ties $\varepsilon_x$ and
$\varepsilon_y$ together. ASBQI sets $e_2=-\bar\nu$, so **ASBQI depends on
Poisson's ratio**; ASOI and ASOI(1/2) set it to $-1$ and $-\tfrac12$, constants,
and depend on nothing. The resulting operators (éq 3.3-2) are

$$B_n^{\text{ASBQI}}=\begin{bmatrix}
h_{,x}\gamma^{t} & -\bar\nu\,h_{,y}\gamma^{t}\\
-\bar\nu\,h_{,x}\gamma^{t} & h_{,y}\gamma^{t}\\
0&0\end{bmatrix},
\qquad
B_n^{\text{ASOI}(1/2)}=\begin{bmatrix}
\tfrac12 h_{,x}\gamma^{t} & -\tfrac12 h_{,y}\gamma^{t}\\
-\tfrac12 h_{,x}\gamma^{t} & \tfrac12 h_{,y}\gamma^{t}\\
0&0\end{bmatrix}$$

against the plain QUAD4 regrouping
$B_n=\big[h_{,x}\gamma^{t},0;\;0,h_{,y}\gamma^{t};\;h_{,y}\gamma^{t},h_{,x}\gamma^{t}\big]$.

### Which projection this project can use

Two criteria decide it, and the second was only found by measuring.

**Poisson dependence disqualifies ASBQI.** A single $\bar\nu$ presumes the
transverse coupling is isotropic. A cubic crystal at a general orientation has a
condensed plane-stress operator with extension–shear coupling and no single
Poisson ratio — this project measured that coupling directly, and refuses an
isotropic reference tangent for the same reason. R3.06.10 does not say how
$\bar\nu$ would be defined for an anisotropic tangent, so ASBQI is not
implemented rather than implemented with an invented rule.

**Frame invariance disqualifies ASOI and ASOI(1/2).** Write the enrichment as
the outer product $G=\mathbf q\otimes\nabla h$, with
$\mathbf q=(q_x,q_y)$ the hourglass amplitudes. Then the added strain is

$$\varepsilon^{\text{stab}}=\begin{bmatrix}
e_1G_{11}+e_2G_{22}\\ e_2G_{11}+e_1G_{22}\\ e_3(G_{12}+G_{21})\end{bmatrix}$$

and this is a **tensor** operation on $G$ only for particular $(e_1,e_2,e_3)$:

| projection | $(e_1,e_2,e_3)$ | tensorial? | what it is |
|---|---|---|---|
| QUAD4 | $(1,0,1)$ | **yes** | $\operatorname{sym}G$ |
| ASMD | $(\tfrac12,-\tfrac12,1)$ | **yes** | $\operatorname{dev}\operatorname{sym}G$ |
| ASOI | $(1,-1,0)$ | no | shear row zeroed in the global frame |
| ASOI(1/2) | $(\tfrac12,-\tfrac12,0)$ | no | the same, halved |

Measured two ways and agreeing: algebraically, a random $G$ rotated then
projected differs from projected then rotated by an $O(1)$ relative amount for
ASOI and ASOI(1/2) and by `6e-16` for QUAD4 and ASMD; and on the element, a
`0.7 rad` rotation of a distorted quadrilateral changes the stabilisation energy
by **37.7 %** for both ASOI variants and by `3e-16` for the other two.

This is not an implementation defect. Setting $e_3=0$ *is* the suppression of the
hourglass shear in the element's own axes, which is what cures bending locking,
and R3.06.10 itself points at the remedy — *"effectuer les calculs en se plaçant
dans un repère tournant avec l'élément"*, noted there as not implemented in
Code_Aster.

**So the default here is ASMD**, not ASOI(1/2). It is the deviatoric part of the
enrichment: tensorial, free of any material constant, and therefore usable with
an anisotropic crystal tangent — which is exactly the "projection tensorielle ne
supposant pas une isotropie caractérisée par un unique coefficient de Poisson"
the specification asks for. ASOI and ASOI(1/2) remain selectable and are worth
comparing on this project's axis-aligned pixel meshes, where every element shares
one frame and the objection has no bite; they must not be used on a general mesh.

An earlier draft of this page recommended ASOI(1/2) on the Poisson criterion
alone. That was wrong, and the frame-invariance measurement is what corrected it.

## Stiffness and internal force

R3.06.10 éq 3.3-3 to 3.3-6:

$$K_e=K_c+K_{\text{stab}},\qquad K_c=A\,B_c^{T}C\,B_c$$

$$K_{\text{stab}}=\sum_{g=1}^{4}\mathrm{JAC}(g)\Big[
B_c^{T}C\,B_n^{(g)}+B_n^{(g)T}C\,B_c+B_n^{(g)T}C\,B_n^{(g)}\Big]$$

$$f_{\text{int}}=\sum_{g=1}^{4}\mathrm{JAC}(g)\,
\big(B_c+B_n^{(g)}\big)^{T}\big(C\,(\varepsilon_c+\varepsilon^{(g)}_{\text{stab}})\big)$$

This is exactly the trial form the specification proposes in its section 4, and
it is the source's, not an adaptation.

Two consequences matter for the implementation.

**The stabilisation uses `C` from the centre.** The four geometric points supply
$B_n^{(g)}$, $\mathrm{JAC}(g)$ and $w_g$ and nothing else. R3.06.10 is explicit:
*"Bien que le calcul de $K_{\text{stab}}$ nécessite une somme sur les quatre
points de Gauss, l'intégration de la loi de comportement qui détermine la valeur
des termes de `C` s'effectue au centre."*

**The cross terms vanish, and not only on a parallelogram.** Since
$\sum_g \mathrm{JAC}(g)\,h_{,x}^{(g)}=\sum_g \mathrm{JAC}(g)\,h_{,y}^{(g)}=0$,
we get $\sum_g \mathrm{JAC}(g)B_n^{(g)}=0$ and both cross products drop out:

$$K_{\text{stab}}=\sum_g \mathrm{JAC}(g)\,B_n^{(g)T}C\,B_n^{(g)} .$$

Measured to machine precision on a square, a sheared quadrilateral, a strong
trapezoid and a heavily distorted quadrilateral — so this is not a
regular-mesh convenience. Two consequences: the stabilisation is exactly
orthogonal to the constant-strain part, and $K_{\text{stab}}$ inherits the
symmetry of `C` rather than acquiring an asymmetry of its own.

The general three-term form is nonetheless what gets implemented, with the
cancellation asserted numerically. A vanishing term that is *known* to vanish
costs nothing to compute and catches a geometry the assumption would break on.

The internal-force expression above writes even the central stress as
$C\varepsilon_c$, which is the elastic reading. In plasticity the central stress
is the integrated $\sigma_c$, and only the stabilisation part goes through the
tangent:

$$f_{\text{int}}=\underbrace{A\,B_c^{T}\sigma_c}_{\text{physical}}
+\underbrace{\sum_g \mathrm{JAC}(g)\,\big(B_c+B_n^{(g)}\big)^{T}
C\,\varepsilon^{(g)}_{\text{stab}}}_{\text{stabilisation}}$$

which is the split section 8 of the specification asks for, and which carries no
history of its own: $\varepsilon_{\text{stab}}$ is a function of the current
displacement alone.

## Conventions, rebuilt

Copying the formulae without this table would be a mistake in this codebase.

| quantity | R3.06.10 | this project |
|---|---|---|
| strain vector | $[\varepsilon_x,\varepsilon_y,2\varepsilon_{xy}]$, engineering shear | identical |
| component order | $xx, yy, xy$ | identical |
| central quadrature | one point, weight 4 on the parent square | identical (`CPS4R_QUADRATURE`) |
| geometric quadrature | four points, weight 1 each | identical (`CPS4_QUADRATURE`) |
| `JAC(g)` | weight times Jacobian determinant | `w_g * det J_g` |
| thickness | implicit unit thickness in 2D | identical; reactions are per mm |
| tangent | current, from the central integration | **new here**; the existing `cps4r` uses a frozen elastic reference |
| Kelvin | not used | not used at element level; Kelvin appears only inside the MFront bridge |

## How this differs from the formulation already in the project

The shipped `cps4r` uses
$K_{hg}=\beta\big(K^{4pt}_{ref}-K^{1pt}_{ref}\big)$ built on a **fixed elastic
reference**. Three differences, all consequential:

1. **The tangent.** QUAS4 stabilises with the current `C`; the existing element
   keeps the elastic one. After yielding the elastic reference over-stiffens the
   hourglass modes while every other mode softens — measured on this project's
   own qualification, where `beta = 1` turned out to be the *least* accurate
   choice and lowering `beta` recovered accuracy.
2. **The projection.** QUAS4 cancels the hourglass shear ($e_3=0$) and couples
   the normal components ($e_2\ne 0$). The difference form does neither: it
   keeps whatever the full-integration operator produces.
3. **The free coefficient.** QUAS4 has none. `beta` exists only because the
   difference form needed a knob, and section 7 of the specification removes it.

## What this page does not settle

- The choice between ASOI and ASOI(1/2) is argued but not measured; that belongs
  to the qualification.
- $\bar\nu$ in ASBQI is described in R3.06.10 without saying how it is computed
  from an anisotropic tangent, because the question does not arise there. Since
  ASBQI is not being implemented, the question is left open rather than answered
  by invention.
- R3.06.10 reports a **20 % average wall-time gain** for elastic and
  elastoplastic laws, and adds that *"des gains de temps beaucoup plus
  importants sont attendus pour des lois plus difficiles à intégrer"*. Our SRIX
  law is precisely such a case, which is what makes the specification's target of
  1.8 on total time plausible; but it is a target to measure, not a result to
  expect.
