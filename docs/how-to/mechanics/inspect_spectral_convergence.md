# Inspect spectral convergence

**Mode:** how-to  
**Domain:** spectral

Check the true mechanical residual, Newton history, GMRES iterations and
preconditioner diagnostics in the recorded run report. A small Krylov residual
alone is not sufficient evidence of mechanical convergence. Compare the
reported residual with the stopping contract in
{doc}`../../reference/numerics/newton_gmres_contract` and the spectral result
contract in {doc}`../../reference/numerics/spectral_result_contract`.
