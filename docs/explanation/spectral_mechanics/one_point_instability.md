# One-point instability

The cell-centred one-point stencil has symbols

\[
L_x^{1p}=4(h_y/h_x)\sin^2(\theta_x/2)\cos^2(\theta_y/2),
\]
\[
L_y^{1p}=4(h_x/h_y)\cos^2(\theta_x/2)\sin^2(\theta_y/2).
\]

Near \((\theta_x,\theta_y)=(\pi,\pi)\), both terms vanish. The checkerboard
mode is therefore a quasi-null mode of the mechanical operator. This is a
spatial defect, not an Anderson, tolerance or FFT-library setting.

TET2 removes the cosine factors. Its high-frequency symbol remains non-zero,
which explains why it can converge through plastic activation without an
empirical hourglass term.

:::{admonition} Project numerical result
The one-point witness stalled after 7,928 iterations with a final high-
frequency displacement fraction of about \(1.43\times10^{-4}\). TET2 did not
show the same persistent mode in the registered campaign.
:::
