# Constitutive-model reference

**Category: Reference.**

## Local J2/Ludwik law

The elastic law is isotropic with Young modulus $E$ and Poisson ratio $\nu$.
The yield function is

$$
f=\sigma_{\mathrm{eq}}-R(p),
\qquad
R(p)=\sigma_{y0}+K h(p),
$$

with associative J2 flow and accumulated equivalent plastic strain $p$.
The origin-regularized Ludwik function is

$$
h(p)=
\begin{cases}
p\,p_0^{n-1}, & p\le p_0,\\
p^n, & p>p_0.
\end{cases}
$$

Its derivative is $p_0^{n-1}$ on the first branch and $np^{n-1}$ on the
second. $\sigma_{y0}$ and $K$ may vary by element; $E$, $\nu$, $n$ and $p_0$
are global.

## Implementations

| Name | Role | Representation |
|---|---|---|
| Python table | historical regression oracle | 1000-segment tabulation with historical upper cap |
| analytical MFront plane stress | nominal local backend | analytical regularized law |
| analytical MFront 3D | extension backend | six-component law condensed by the adapter |
| micromorphic MFront plane stress | nominal coupled backend | analytical law plus $H_\chi(p-\chi)$ |
| micromorphic MFront 3D | coupled extension backend | six-component coupled law |

The historical cap is not a feature of the analytical model. Comparisons
between table and analytical implementations state explicitly whether the
tested path remains below it.

## Micromorphic yield radius

At fixed external state $\chi$,

$$
R(p,\chi)=\sigma_{y0}+Kh(p)+H_\chi(p-\chi),
\qquad
\frac{\partial R}{\partial p}=Kh'(p)+H_\chi.
$$

The MFront tangent is consistent with this **fixed-$\chi$ local update**. It is
not the monolithic tangent of the converged Helmholtz-coupled problem.

## State variables and properties

The authoritative names, sizes and offsets are checked at runtime against
MGIS metadata. Required local observables include PEEQ, elastic strain and
yield-surface radius. Coupled behaviours additionally expose
`NonlocalEquivalentPlasticStrain` and the property
`MicromorphicCouplingModulus`.
