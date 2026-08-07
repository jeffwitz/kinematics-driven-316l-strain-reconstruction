# Diagnostic exploratoire — branches multiples du problème SRIX en contrainte plane

Date: 2026-08-07
Code: `scripts/diagnose_srix_plane_stress_branches.py`.

**Étude exploratoire, pas une procédure de décision.** Elle explique le F1 de
`validation/srix_umat_gps_closure_results.md` : pourquoi la référence
(`mfront-3d-condensed-plane-stress`) et le backend UMAT
(`mfront-native-generalised-plane-stress`) convergent vers des solutions
différentes du même problème discret. Aucun backend n'est sélectionné ni
rejeté ici ; la référence condensée reste la référence.

## Faits mesurés

**1. Première divergence à l'incrément 2.** L'incrément 1 (premier incrément
plastique : 8 systèmes actifs dès le premier pas) s'accorde à `6,7e-10` sur
la contrainte. L'écart apparaît à l'incrément 2 et les trajectoires divergent
ensuite : les glissements UMAT s'accumulent ~2 à 3× plus vite (inc. 4 :
`0,0122` contre `0,0363`), et le Newton UMAT finit par échouer (inc. 8).

**2. Les ensembles de systèmes actifs sont IDENTIQUES.** Sur toute la
trajectoire, les deux solutions activent exactement les mêmes huit systèmes
`[1, 2, 4, 5, 7, 8, 10, 11]`. La différence de branche n'est pas une
sélection différente de systèmes : ce sont les **amplitudes** qui diffèrent.

**3. Le problème 3D admet plusieurs racines au même état.** Depuis le même
état committé (incrément 1) et le même incrément de déformation (incrément 2
avec la déformation transverse UMAT imposée), la loi 3D brute
(`Fcc316LForestRubinSrix`) converge vers une racine avec
`sigma_zz = -154,7 MPa`, tandis que le Newton conjoint UMAT (21 inconnues)
converge vers une racine avec `sigma_zz = 0` — les deux Newtons convergent
(statut 0), les deux états satisfont leurs systèmes. L'équation de fermeture
`sigma_zz(eps_zz) = 0` admet donc (au moins) deux solutions en `eps_zz` :
`-0,00165` (branche de la référence) et `-0,00262` (branche UMAT).

**4. Sensibilité au départ, inversée entre les deux backends.** Dix départs
perturbés (`1e-4` sur les variables internes committées) avant l'incrément 2 :
la branche UMAT est **robuste** — `eps_zz = -0,002624` et contrainte
`[159,04 ; 8,89]` identiques sur les dix départs ; la **référence échoue** sur
neuf départs sur dix (échec du Newton de fermeture local). La robustesse
n'est donc pas du côté de la référence.

**5. Mécanisme : la rétroaction `eps_zz` ↔ `Deq`.** L'amplitude d'écoulement
`Deq = sqrt(2/3 (de|de))` croît avec `|eps_zz|` (la déformation transverse
entre dans le déviateur total). Les deux branches sont des points fixes
auto-cohérents de la boucle « fermeture → Deq → glissements → contrainte
transverse » : la branche UMAT combine une plus grande `|eps_zz|`, un `Deq`
plus grand et des glissements ~2× plus grands, tout en satisfaisant
`sigma_zz = 0`.

## Limites et interprétation

- Le comptage de racines par départs perturbés est **contaminé** : perturber
  l'état committé change le problème incrémental (les racines « à 12
  systèmes » observées viennent de problèmes perturbés). La preuve valide de
  la multiplicité est la comparaison **même état, même incrément, Newtons
  différents** (fait 3), pas le comptage par départs.
- Les deux branches sont des racines du même système discret ; aucune n'est
  « plus correcte » au niveau discret. La multiplicité est une propriété de
  la loi SRIX (crochet de Macaulay), déjà consignée au journal du 2026-08-03
  (« une autre solution »).
- Conséquence pour la référence : les campagnes archivées condensées
  reposent sur UNE branche ; un chemin de fermeture différent (ou un Newton
  conjoint) peut sélectionner l'autre. C'est un point d'identifiabilité de la
  loi, pas un artefact d'implémentation.

## Ce que ce diagnostic ne décide pas

- Il ne qualifie ni ne rejette le backend UMAT (la qualification F1 reste
  l'autorité) ;
- il n'identifie pas la branche « physique » (la limite du pas fini — l'écart
  croît quand le pas diminue, donc les branches ne se rejoignent pas à pas
  nul sur ce chemin) ;
- il ne règle pas la sélection de branche. Les voies possibles restent
  ouvertes : bornage de `eps_zz` physique, étude de l'unicité de la loi,
  ou acceptation documentée de la branche UMAT comme solution légitime.

## Enregistrement

Artefacts : `validation/_generated/performance/srix_plane_stress_branches.json`
et le script de diagnostic. Aucun chiffre de la qualification n'est modifié.
