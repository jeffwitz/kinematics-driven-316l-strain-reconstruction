# P43-SYNTH-003 — montée M20 vers le crop P43 M100

Après `P43-SYNTH-002B`, le meilleur résultat multi-départ M20 est utilisé
comme initialisation paramétrique d'un calcul synthétique sur le crop P43
enregistré de 100×100 éléments (origine `(1580,1030)`, nœuds `[1580:1680,
1030:1130]`). La vérité synthétique reste le preset SRIX connu, avec la même
histoire DIC réparée, 32 incréments, EBSD co-enregistré et observation identité.

Le calcul M100 effectue une optimisation limitée à quatre évaluations pour
fournir un premier diagnostic de transfert d'échelle pendant la nuit. Il
archive séparément le forward de vérité, l'état initial issu de M20, le champ
identifié, la Jacobienne finale et les temps. Ce n'est pas une campagne P43
expérimentale et la récupération quatre-paramètres n'est pas revendiquée.

