# Run a full-Dirichlet spectral reconstruction

**Mode:** how-to  
**Domain:** spectral

Prepare the measured boundary displacement, build the fluctuation extension,
select the declared constitutive backend, and run the matrix-free Newton/GMRES
solver. Inspect the true mechanical residual and the convergence report before
comparing fields. Use {doc}`../../reference/numerics/newton_gmres_contract`
for stopping criteria and {doc}`../../explanation/spectral_mechanics/index`
for the formulation.
