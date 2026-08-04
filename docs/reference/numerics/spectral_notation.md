# Spectral notation

| symbol | meaning | shape or units |
|---|---|---|
| \(u\) | total nodal displacement | `(nx+1, ny+1, 2)` |
| \(u^f\) | zero-boundary fluctuation | same shape |
| \(\varepsilon_{eq}\) | TET2 sample strain | `(nx, ny, 2, 3)` |
| \(\sigma_{eq}\) | TET2 sample stress | `(nx, ny, 2, 3)`, MPa |
| \(z_{eq}\) | TET2 internal state | two states per pixel |
| \(R\) | nodal internal-force convention | interior vector |

Engineering shear is \(\gamma_{12}=2\varepsilon_{12}\). The stress vector
contains \(\sigma_{12}\), and \(\sigma_a^T\varepsilon_a\) is the power pair.
The plane-stress components are \(a=(11,22,12)\) and
\(b=(33,13,23)\).

The code convention is `divergence = -B.T @ stress`. Reactions use the
opposite internal-force convention, `B.T @ stress`, and must not be confused
with the interior equilibrium residual.
