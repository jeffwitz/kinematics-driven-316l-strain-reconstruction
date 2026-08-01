# P43 (ell, alpha) selection — machine summary

Profile `legacy_script_2021`, principal scale `49 px`, generated 2026-08-01T17:46:54.360003+00:00.

**Registered outcome: B_robust_zone.** Several configurations are indistinguishable and form a compromise zone of 10 points, reported as an explicit list rather than a rectangle. No point is chosen.

## Defects at the principal scale

| point | alpha | ell | D_shape | D_amplitude | D_localisation | D_presence | minimax |
|---|---:|---:|---:|---:|---:|---:|---:|
| a0p5-ell20 | 0.5 | 20 | 0.6862 | 0.2173 | 0.2872 | 0.03489 | 0.9176 |
| a0p5-ell40 | 0.5 | 40 | 0.6716 | 0.1374 | 0.2713 | 0.1839 | 0.8793 |
| a0p5-ell58p88 | 0.5 | 58.88 | 0.6663 | 0.09951 | 0.2664 | 0.2505 | 0.8723 |
| a0p5-ell90 | 0.5 | 90 | 0.6681 | 0.05575 | 0.2641 | 0.3149 | 0.8748 |
| a1-ell20 | 1 | 20 | 0.668 | 0.1293 | 0.292 | 0.2292 | 0.8746 |
| a1-ell40 | 1 | 40 | 0.656 | 0.01093 | 0.2942 | 0.4522 | 0.8588 |
| a1-ell58p88 | 1 | 58.88 | 0.6548 | 0.04876 | 0.2849 | 0.5606 | 0.8572 |
| a1-ell90 | 1 | 90 | 0.6535 | 0.1003 | 0.2784 | 0.6545 | 0.8555 |
| a2-ell20 | 2 | 20 | 0.658 | 0.03634 | 0.3128 | 0.4631 | 0.8615 |
| a2-ell40 | 2 | 40 | 0.6441 | 0.1809 | 0.3098 | 0.8146 | 1.054 |
| a2-ell58p88 | 2 | 58.88 | 0.6465 | 0.2864 | 0.3827 | 0.9922 | 1.284 |
| a2-ell90 | 2 | 90 | 0.6412 | 0.3649 | 0.4603 | 1.126 | 1.547 |
| a4-ell58p88 | 4 | 58.88 | 0.6252 | 0.6051 | 0.721 | 1.518 | 2.572 |
| a4-ell90 | 4 | 90 | 0.6342 | 0.7179 | 0.8407 | 1.698 | 3.054 |

Non-converged, excluded: ['a4-ell20', 'a4-ell40'].

## Front, stability and zone

- Pareto front on the raw defects: ['a0p5-ell20', 'a0p5-ell40', 'a0p5-ell58p88', 'a0p5-ell90', 'a1-ell20', 'a1-ell40', 'a1-ell58p88', 'a1-ell90', 'a2-ell40', 'a2-ell90', 'a4-ell58p88']
- most frequent minimax winner: a1-ell90 at 32.9% of 10000 usable draws
- bootstrap verdict: indistinguishable_zone
- indistinguishable zone: ['a0p5-ell20', 'a0p5-ell40', 'a0p5-ell58p88', 'a0p5-ell90', 'a1-ell20', 'a1-ell40', 'a1-ell58p88', 'a1-ell90', 'a2-ell20', 'a2-ell40']

## Iso-Achi pairs

- `Achi = 800`: ['a2-ell20', 'a0p5-ell40'], minimax separation 0.01788
- `Achi = 1600`: ['a4-ell20', 'a1-ell40'], not comparable, a member did not converge

## What this does not license

P43 only. A provisional reconstruction parameterisation, not an internal length of 316L, not transferability, and neither a validation nor a refutation of the nonlocal formulation.
