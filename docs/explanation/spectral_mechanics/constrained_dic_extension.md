# Extension to constrained DIC data

Full-boundary Dirichlet data are currently represented by \(u=u^\ast+u^f\)
with zero fluctuation on the contour. Partial or interior observations would
instead be written as

\[
Au=d,
\qquad
\begin{bmatrix}J&A^T\\A&0\end{bmatrix}
\begin{bmatrix}\Delta u\\\Delta\lambda\end{bmatrix}
=-\begin{bmatrix}R+A^T\lambda\\Au-d\end{bmatrix}.
\]

This saddle-point or projected formulation is a future extension. It is not
part of the qualified TET2 API and should not be inferred from the current
full-Dirichlet results.
