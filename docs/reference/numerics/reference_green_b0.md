# Reference Green operator contract

The full-Dirichlet Green operator is a preconditioner for the global Newton
linearization. For every transformed mode it applies

\[
\widehat{\delta u}_x=-\widehat p_x/(2\mu_0L+\lambda_0L_x),
\quad
\widehat{\delta u}_y=-\widehat p_y/(2\mu_0L+\lambda_0L_y).
\]

\(\lambda_0\) and \(\mu_0\) are either explicit inputs or the recorded
Kelvin-consistent isotropic projection. They are not SRIX parameters and do
not alter the constitutive response.

Changing both parameters by one common scalar is not a meaningful preconditioner
quality test: it only rescales the inverse in exact arithmetic. A meaningful
campaign changes their ratio or compares an anisotropic reference.
