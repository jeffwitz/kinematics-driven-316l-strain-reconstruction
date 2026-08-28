# Spectral solver reference

**Mode:** reference  
**Domain:** spectral

The full-Dirichlet solver decomposes displacement as

$$u=u^*+u_f,$$

where the lifting ``u*`` carries the measured boundary values and the
fluctuation ``uf`` is zero on the boundary. The discrete ``B`` maps a
displacement field to strain and ``B^T`` maps stress to the internal residual.
The solver applies the boundary through the lifting, then solves the true
mechanical residual with matrix-free Newton--GMRES. No global Jacobian is
assembled. The homogeneous ``B0^{-1}`` operator is the preconditioner; it is
not the nonlinear residual and does not replace constitutive integration.

The transform layer implements the declared DST-I convention and supports
SciPy or FFTW through `SpectralTransformConfig`. FFTW plan/wisdom settings and
diagnostics are recorded, and numerical equivalence is tested independently
from performance (`E-FFT-006` remains a performance qualification item).

See {doc}`spectral_notation`, {doc}`full_dirichlet_dst1`,
{doc}`newton_gmres_contract` and {doc}`transform_backends` for specialised
contracts. Material trials are evaluated from the last committed state and
only accepted increments are committed.
