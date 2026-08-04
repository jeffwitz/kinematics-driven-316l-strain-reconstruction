# TET2 Newton-GMRES

TET2 carries two independent SRIX histories per pixel. At Newton iterate \(k\),

\[
R(u_k)=-\sum_{e,q}w_qA_eB_{eq}^T\sigma_{eq},
\qquad
J_k\delta u=-R(u_k).
\]

The Jacobian is applied matrix-free:

\[
J_kv=-\sum_{e,q}w_qA_eB_{eq}^T
C_{\mathrm{alg},eq}^{ps}B_{eq}v.
\]

GMRES acts only on interior displacement degrees of freedom. The DST-I Green
operator is its preconditioner. Every line-search candidate is integrated from
the last committed state; rejected trials are reverted. At convergence, the
solution is re-integrated after `revert()`, its residual is independently
verified, and only the verified trial is committed.

:::{admonition} Project numerical result
At 24x24 the two-state TET2 solution is within 0.72% in accumulated slip of
CPS4. Verification residuals were \(5.53\times10^{-12}\) at 12x12 and
\(9.74\times10^{-14}\) at 24x24.
:::
