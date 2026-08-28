# CPS4R and hourglass contract

**Mode:** reference  
**Domain:** constitutive

`CPS4` uses four Gauss points. `CPS4R` uses one centroid point with weight 4.
The stabilised element is assembled as

$$K_e=K^{1pt}(C_{tangent})+K_{hg},\qquad
K_{hg}=\beta(K^{4pt}_{ref}-K^{1pt}_{ref}).$$

The recorded hourglass energy is

$$E_{hg,e}=\tfrac12u_e^TK_{hg,e}u_e,$$

and the global ratio is reported against accumulated internal work. The
reference term is elastic and fixed; it is not plastic dissipation or crystal
hardening. At \(\beta=1\), CPS4R equals CPS4 only for a constant elastic
tangent. The production plastic qualification therefore remains CPS4.
