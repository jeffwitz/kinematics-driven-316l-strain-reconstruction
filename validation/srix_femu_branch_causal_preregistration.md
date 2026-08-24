# E-SRIX-FEMU-BRANCH-002B — preregistration

Objectif : localiser la première divergence causale entre l'histoire coarse
de 57 pas et une histoire obtenue en raffinant chaque intervalle en deux.

Configuration : M8, SRIX local qualifié, mêmes paramètres et mêmes conditions
de bord que `srix_femu_common_path_gate_v9`, solveur oracle strict, aucune
identification. Les états internes sont lus via les observables déjà exposées
par le bridge MFront (`plastic_slip`, `equivalent_plastic_slip`, `back_strain`,
`elastic_strain`).

Le diagnostic compare les endpoints communs acceptés jusqu'à l'échec, puis
teste des chemins hybrides où les `k` premiers intervalles coarse sont raffinés
(`k = 8, 16, 24, 32, 40, 48, 57`). Les chemins sont diagnostiques et ne
constituent pas un nouveau chemin scientifique validé.

Décision préenregistrée :

- divergence régulière sans changement d'activité : dérive incrémentale ;
- changement brutal d'activité : candidat transition d'ensemble actif ;
- états proches mais échec Newton/Krylov : bassin ou conditionnement global ;
- données insuffisantes ou contradictions : non résolu.

Un résultat non résolu bloque toute reprise de PATH-002, toute identification
et toute campagne P43.
