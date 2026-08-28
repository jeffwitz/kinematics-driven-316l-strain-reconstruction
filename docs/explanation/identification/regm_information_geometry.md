# REGM information geometry

**Mode:** explanation  
**Domain:** identification

The comparison is made in the parameter space used by each sensitivity
calculation. The observed FEMU spectrum is normalised as
``(1, 0.542, 0.407, 0.0679)``, while exact REGM gives
``(1, 0.422, 0.0324, 4.65e-5)``. A statewise algorithmic-tangent correction
changes REGM to ``(1, 0.37594, 0.03469, 8.62e-5)`` and leaves a leading
principal-angle mismatch of ``73.9`` degrees.

These results explain why a good exact-twin ranking is insufficient: the
surrogate and the observed FEMU objective do not preserve the same weak and
strong parameter combinations. They also motivate reporting singular values,
right singular vectors and principal angles rather than only a scalar ranking
correlation.
