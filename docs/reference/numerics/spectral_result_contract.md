# Spectral result contract

Every reported run must identify:

* grid, stencil and material-state count;
* DST plan and reference \(\lambda_0,\mu_0\);
* equilibrium residual and post-revert verification residual;
* Newton/GMRES iterations and constitutive calls;
* displacement, stress, accumulated-slip and reaction errors against the
  declared oracle;
* high-frequency energy and cutbacks.

The dimensionless equilibrium criterion is reported separately from the
dimensional nodal force norm. Side sums that include corner nodes are labelled
boundary-node sums, not strict face resultants.
