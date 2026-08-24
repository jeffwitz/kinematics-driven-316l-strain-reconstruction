# Mechanical projection before SRIX replay — M8 twin result

The preregistered test evaluated the existing correction
`-K0^-1 B^T sigma` as a one- or two-pass projection of the transferred
kinematics before causal SRIX replay. The same twenty candidates and the same
observed FEMU target as the placement ablation were used. No new forward FEMU
solve was launched.

| variant | Spearman | log-Pearson | top-five overlap | truth REGM RMS (mm) | history error RMS (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| observed | 0.326 | 0.276 | 2/5 | 2.132e-7 | 1.108e-6 |
| one pass, damping 0.25 | 0.341 | 0.267 | 2/5 | 1.992e-7 | 1.042e-6 |
| one pass, damping 0.50 | 0.341 | 0.262 | 2/5 | 1.861e-7 | 9.806e-7 |
| one pass, damping 1.00 | 0.326 | 0.256 | 2/5 | 1.647e-7 | 8.748e-7 |
| two passes, damping 0.50 | 0.326 | 0.255 | 2/5 | 1.673e-7 | 8.853e-7 |
| two passes, damping 1.00 | 0.341 | 0.251 | 2/5 | 1.381e-7 | 7.566e-7 |

The projection decreases the error of the reference replay and moves the
history closer to the exact latent twin. It does **not** restore the candidate
ranking: all variants remain far below the preregistered Spearman threshold
`0.80` and retain only two of the best five candidates.

This is a useful negative result. The first-order equilibrium correction removes
part of the mechanical residual, but the remaining constitutive-history bias is
still larger than the parameter-discriminating signal. A projection based on a
single SRIX reference law is therefore not sufficient for P43.

The machine-readable report and figure are in
`validation/reference_data/srix_regm_mechanical_projection_v1/`. This result
does not invalidate the exact-space REGM surrogate; it rejects this simple
latent-history reconstruction as the next production method.
