# E-SRIX-P43-SYNTH-SVD-001 — pré-enregistrement C1–C3

Cette campagne reste limitée au jumeau synthétique P43 M20. Aucun P43
expérimental ni nouveau M100 n'est lancé.

- base SVD : Jacobienne directe M20 à la vérité, archivée dans
  `validation/reference_data/p0043_synthetic_identification_v1/fields.npz` ;
- référence : `eta_ref=log(theta_true)` ;
- rang fixé avant calcul : `3` ;
- pas shadow : `h=0.0015` ;
- quatre départs B1–B4 de `P43-SYNTH-002B` ;
- observation identité, sans bruit, 32 incréments.

Le gate C1 optimise `z1,z2,z3` dans `eta=eta_ref+V3 z`; ni `Q` ni `b` ne
sont fixés. Le C3 initial profile la direction écartée `v4` à
`z4=(-.30,-.20,-.10,0,.10,.20,.30)` en gardant `z1..z3` à la meilleure
solution C1. Cette première version est descriptive (forwards directs), avant
une éventuelle réoptimisation conditionnelle de `z1..z3`.

