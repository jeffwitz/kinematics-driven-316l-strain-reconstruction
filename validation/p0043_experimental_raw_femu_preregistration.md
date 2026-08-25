# P43-EXP-RAW-001 — pré-enregistrement FEMU brute

Cette campagne reprend le crop expérimental P43 M20 de `P43-EXP-001`, mais
retire toute pondération par le bruit DIC. Le résidu et la Jacobienne fournis à
l'optimiseur sont directement en millimètres :

```text
r = u_sim - u_DIC
J = d u_sim / d log(theta)   [mm]
observation_weighting = none
noise_model_used = false
covariance_used = false
```

Le chemin DIC, l'EBSD, la loi SRIX, la contrainte plane, les 32 incréments et
`shadow_fd_step=0.0015` restent inchangés. La base SVD est construite sur la
Jacobienne brute au prior, avec rang fixé à trois et `z4` conservé à sa valeur
initiale. Les bornes physiques et les départs réduits sont inchangés.

M20 est exécuté avec quatre départs (nominal + E1--E3), `max_nfev=24` et des
tolérances strictes exprimées sur le résidu physique. M100 reste conditionné au
gate M20 et ne peut pas être lancé en parallèle.
