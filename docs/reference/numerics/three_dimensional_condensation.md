# Three-dimensional plane-stress condensation

**Category: Reference.**

For a six-component material law, partition engineering strains and stresses:

$$
\varepsilon_a=[\varepsilon_{11},\varepsilon_{22},\gamma_{12}],\qquad
\varepsilon_b=[\varepsilon_{33},\gamma_{13},\gamma_{23}],
$$

$$
\sigma_b=[\sigma_{33},\sigma_{13},\sigma_{23}].
$$

At every Gauss point, the adapter solves $\sigma_b=0$ by local Newton:

$$
C_{bb}\,\Delta\varepsilon_b=-\sigma_b.
$$

Each trial restarts from the same committed material state. After convergence,
the tangent supplied to the 2D solver is the Schur complement

$$
C^{PS}=C_{aa}-C_{ab}C_{bb}^{-1}C_{ba}.
$$

Linear systems are solved; matrices are never explicitly inverted. The
adapter checks residual, iteration count, finite values and conditioning.
This path is verified for the current isotropic J2 law and is the intended
extension point for a general 3D crystal-plasticity behaviour.
