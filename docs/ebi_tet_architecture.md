# Experimental EBI-TET plane-stress mechanics

This branch evaluates an original SRIX adaptation of the Hookean EBI idea in
[Gehrig and Schneider (2025)](https://onlinelibrary.wiley.com/doi/10.1002/nme.70170).

The full scientific account, including the bounded SRIX falsification result,
is in {doc}`explanation/spectral_mechanics/index`.
The article does not qualify crystal plasticity; this implementation is therefore
experimental.

Each pixel owns two constant-triangle strain samples but one transactional SRIX
state. MFront integrates only the mean strain. Sample stresses are reconstructed
with the immutable, orientation-dependent elastic plane-stress tangent:

```text
sample stress = mean SRIX stress + Ce_ps (sample strain - mean strain)
```

The exact EBI linearisation uses the current algorithmic tangent only for the mean
strain and `Ce_ps` for sample fluctuations. Newton-GMRES applies this Jacobian
matrix-free. A DST-I/B0 operator preconditions interior displacement unknowns;
it never reconstructs physical stresses. No hourglass coefficient, filtering, or
second material state is introduced.

`CellCenteredOnePoint2D` is unchanged and remains the near-hourglass witness.
`TwoSubcellDiagnostic2D` remains the two-state diagnostic. The EBI path is isolated
in `ebi.py` and `newton_ebi.py`.
