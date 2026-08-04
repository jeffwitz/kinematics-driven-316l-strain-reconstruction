# DST-I and the reference Green operator

For zero fluctuation on all four sides, the interior modes are the orthonormal
DST-I basis

\[
\phi_{kl}(i,j)=\sqrt{2/n_x}\sqrt{2/n_y}
\sin(\pi ki/n_x)\sin(\pi lj/n_y),
\]

with \(k=1,\ldots,n_x-1\) and \(l=1,\ldots,n_y-1\). The implementation uses
`scipy.fft.dstn`/`idstn`, type 1, with `norm="ortho"`.

For TET2, with \(\theta_x=\pi k/n_x\) and \(\theta_y=\pi l/n_y\),

\[
L_x=4(h_y/h_x)\sin^2(\theta_x/2),\quad
L_y=4(h_x/h_y)\sin^2(\theta_y/2),\quad L=L_x+L_y.
\]

The isotropic reference operator \(B_0\) is diagonal in this basis:

\[
d_x=2\mu_0L+\lambda_0L_x,\qquad
d_y=2\mu_0L+\lambda_0L_y,
\]
\[
\widehat{G_0p}_x=-\widehat p_x/d_x,
\qquad
\widehat{G_0p}_y=-\widehat p_y/d_y.
\]

\(B_0\) is a preconditioner, not the SRIX constitutive law. The projected
reference parameters are recorded separately from the elastic and algorithmic
plane-stress tangents.

:::{admonition} Literature result
DTT reference operators are the standard mechanism for non-periodic
FFT-based mechanics. The present implementation is a SciPy prototype, not a
native FFTW/2DECOMP implementation.
:::
