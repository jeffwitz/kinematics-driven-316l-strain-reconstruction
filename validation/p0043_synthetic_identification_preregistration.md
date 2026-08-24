# P43-SYNTH-001 — pré-enregistrement de l’identification synthétique

## Objet

Tester la récupération de paramètres SRIX sur un jumeau synthétique construit
avec un crop P43 réel, sa carte EBSD réelle et son histoire de déplacements DIC
réparée. Cette campagne ne concerne pas le P43 expérimental et ne permet
aucune conclusion sur le matériau réel.

## Support et vérité

- crop P43 M20 : nœuds absolus `[1610:1630, 1075:1095]` ;
- EBSD : `essais/9_numerical/CP_dataset.h5`, angles Bunge co-enregistrés ;
- frontière : `validation/reference_data/dic_multistep_history_p0043_repaired_v1/repaired_history_mm.npy` ;
- états macro : 0, 5, ..., 40 ;
- génération synthétique avec SRIX et le preset enregistré ;
- aucune contrainte, contrainte interne ou variable d’état fournie à
  l’identificateur ;
- première étape sans bruit et avec observation identité, afin de tester la
  chaîne numérique avant d’introduire le transfert DIC.

## Ordre des tests

1. départ proche, quatre paramètres libres, même partition génération/inversion ;
2. départs éloignés, sans bruit ;
3. trois paramètres avec `b` fixé à la vérité, puis avec `Q` fixé ;
4. quatre paramètres libres depuis plusieurs départs, étude de la vallée `Q-b` ;
5. seulement après réussite des étapes précédentes : mismatch de partition,
   puis transfert et bruit DIC.

## Paramétrisation et optimisation

- coordonnées `eta = log(tau0, R, Q, b)` ;
- `C`, `d`, élasticité et interactions FCC fixés au preset ;
- bornes : facteurs `[1/4, 4]` autour de la vérité, fixés avant calcul ;
- `scipy.optimize.least_squares` avec Jacobienne FEMU directe et pas shadow
  `h=0.0015` adopté par `SHADOW-003C` ;
- aucune optimisation REGM et aucun calcul P43 expérimental.

## Critères du smoke test sans bruit

Le test A est considéré comme réussi si le coût final atteint le plancher
numérique et si l’erreur dans les directions SVD identifiables est faible. La
récupération coordonnée de `Q` et `b` n’est pas exigée si la SVD montre une
direction `Q-b` quasi nulle ; cette dégénérescence doit alors être rapportée,
jamais masquée par un prior.
