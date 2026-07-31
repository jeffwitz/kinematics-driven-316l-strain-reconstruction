# ADR 0009 — Étendre l'opérateur d'observation existant plutôt qu'en créer un second

- Statut : accepté
- Date : 2026-07-31
- Portée : lot 1 du cahier des charges « outils objectifs de comparaison
  spatiale entre EVM DIC et EVM FEM »

## Contexte

Le cahier des charges demande, au §4.1, d'ajouter un mode d'observation
`synthetic_disflow` réalisant la chaîne
`déplacements FEM → image déformée → DISFlow → EVM`.

**Cette chaîne existe déjà** dans `workflows/dic_observation_replay.py`, sous la
fonction `replay_dic_observation`, et son manifeste déclare littéralement
`"mode": "synthetic_disflow"`. Elle a produit les résultats archivés du lot V3
et les campagnes de comparaison de trajets du 2026-07-30.

Créer un second opérateur produirait deux chemins de mesure concurrents, et les
résultats archivés cesseraient d'être comparables aux nouveaux — c'est-à-dire
exactement la faute que le §3.2 interdit (« même opérateur pour tous les
candidats »).

## Décision

Étendre l'opérateur existant. Quatre règles :

1. **Ne pas reconstruire.** `replay_dic_observation` reste l'unique
   implémentation de `M_DIC`.
2. **Ne renommer aucun artefact archivé.** Les noms existants
   (`fem_raw_evm.npy`, `fem_observed_evm.npy`, `report.json`) diffèrent de ceux
   du §4.2 mais sont référencés par les hashs `outputs` de rapports déjà
   commités. Les fichiers manquants sont **ajoutés** ; la correspondance de
   nommage est documentée en référence.
3. **Refuser plutôt qu'interpoler.** Voir le contrat de données ci-dessous :
   l'interpolation est aujourd'hui l'identité. Un support incompatible doit
   lever une erreur, jamais être rééchantillonné silencieusement.
4. **Ne pas créer `src/fem_inhouse/validation/`** au lot 1. L'arborescence du
   §13 est justifiée pour les lots 2 à 4 (géométrie des bandes, FSS, bootstrap,
   Pareto), qui sont du code neuf. Le lot 1 est une extension et reste où il
   est.

## Fichiers créés ou modifiés au lot 1

| Fichier | Action |
|---|---|
| `src/fem_inhouse/workflows/dic_observation_replay.py` | étendu : artefacts §4.2 manquants, métadonnées §4.3 manquantes |
| `src/fem_inhouse/measurement/warp.py` | constantes d'interpolation et de bord exposées pour le manifeste |
| `tests/unit/workflows/test_observation_operator_contract.py` | nouveau : tests synthétiques §4.4 non couverts |
| `docs/reference/observation_operator.md` | étendu : contrat de données, correspondance de nommage, limites |
| `docs/adr/0009-observed-evm-comparison-operator.md` | ce document |
| `docs/reference/architecture_decisions.md` | entrée de résumé |

Aucun fichier de résultat archivé n'est modifié. Aucun comportement MFront,
solveur ou claim n'est touché.

## Contrat de données : déplacements FEM sur la grille image

| Élément | Valeur |
|---|---|
| entrée | `U.npy`, `float64`, forme `(nx+1, ny+1, 2)` |
| axes | canoniques `(x, y)`, composantes `(ux, uy)` |
| unité | millimètres |
| support image | recadrage `[R0+sx0 : R0+sx1+1, C0+sy0 : C0+sy1+1]`, forme `(nx+1, ny+1)` |
| **interpolation** | **identité** — un nœud vaut un pixel, par construction du recadrage |
| conversion | `canonical_to_image_flow` : `[u_y, u_x] / pixel_size_mm` (échange d'axes + passage en pixels) |
| warp | `cv2.remap`, `INTER_LINEAR`, `BORDER_REFLECT101`, résolution itérative de `destination = source + u(source)`, tolérance `1e-5` px, 50 itérations max, jacobien minimal `1e-6` |
| retour | `image_flow_to_canonical`, inverse exact vérifié par test |
| EVM | `reconstruct_historical_evm` sur le champ nodal, opérateur historique en contraintes planes, `nu = 0.3`, pas `= pixel_size_mm` |

L'égalité `forme du recadrage == forme du champ nodal` est **vérifiée à
l'exécution** et lève une erreur si elle est violée. C'est ce qui rend l'étape
d'interpolation du §4.1 sans objet sur ce support, et ce qui empêche un support
futur incompatible de passer inaperçu.

## Cache et provenance

Le cache existe déjà : `DICObservationOperatorConfig.fingerprint()` produit une
empreinte stable qui inclut le mode d'observation, et un test garantit que la
configuration `synthetic_disflow` y entre bien
(`test_synthetic_observation_configuration_is_in_cache_fingerprint`).

La provenance repose sur des hashs SHA-256 : image de référence, déplacements
FEM, manifeste de campagne, cas préparé, et chaque artefact produit. Le
manifeste porte aussi le commit Git, la convention d'axes, le recadrage, le
masque, l'opérateur EVM, les paramètres DISFlow **demandés** et **relus depuis
OpenCV**, ainsi que la liste explicite des réglages laissés aux valeurs d'usine.

Complété au lot 1 : état propre ou modifié du dépôt, version d'OpenCV, mode
d'interpolation, mode de bord, paramètres de différentiation.

## Tests synthétiques prévus

Déjà couverts, conservés : translation rigide exacte, EVM nulle sous
translation rigide, inverses exacts entre conventions canonique et image,
correspondance historique `U = u_y` / `V = u_x`, récupération d'une translation
lisse par warp + DISFlow, rejet d'une carte non inversible, rejet d'une
configuration DISFlow invalide.

Ajoutés au lot 1 : déformation affine, cisaillement simple, bande gaussienne
intégrée, **orientations horizontale et verticale** de la même bande, absence de
transposition ou de miroir implicite sur un champ non carré et asymétrique,
déterminisme bit à bit sur deux exécutions, refus d'une échelle finale non
native en mode métrologique.

## Ambiguïtés détectées

Elles sont consignées ici parce que les lots suivants s'appuieront dessus.

1. **Le support image est nodal, l'EVM est centrée élément.** Le recadrage fait
   `(nx+1, ny+1)` et l'EVM `(nx, ny)`. Il existe donc un décalage d'un demi-pixel
   entre le réseau des nœuds et celui des centres d'éléments. Il est **identique
   pour la DIC et pour le FEM**, donc il s'annule dans la comparaison actuelle.
   Il ne s'annulera plus au lot 2, où les lignes centrales et sections normales
   sont des objets géométriques en coordonnées pixel : la géométrie des bandes
   devra déclarer explicitement sur quel réseau elle vit.
2. **L'interpolation est l'identité aujourd'hui.** Le §4.1 suppose une étape
   d'interpolation. Elle n'existe pas et n'est pas nécessaire sur ce support. La
   décision est de refuser un support non conforme plutôt que d'écrire un
   rééchantillonnage non testé.
3. **Le mode de bord du warp est `BORDER_REFLECT101`.** Le contenu réfléchi au
   bord du recadrage n'est pas physique. C'est sans effet aujourd'hui car la
   comparaison porte sur le cœur, padding exclu, mais un corridor de bande placé
   près du bord au lot 2 y serait exposé.
4. **Le masque est un déclaratif, pas une mesure.** `declared_all_valid_mask`
   déclare tout valide ; il n'existe aujourd'hui aucune zone invalide. Le lot 2
   ne doit pas supposer qu'un masque réel filtre quoi que ce soit.
5. **Le renommage historique `U = u_y`, `V = u_x`** est la place classique d'un
   bug de transposition. Tout code neuf touchant les tableaux historiques bruts
   doit passer par `historical_uv_to_canonical`.
6. **La version d'OpenCV n'était pas enregistrée.** Le manifeste consignait les
   paramètres demandés et relus, mais pas la version de la bibliothèque ; or les
   valeurs d'usine des réglages non fixés diffèrent entre versions. Le manifeste
   n'était donc pas suffisant pour reproduire. Corrigé au lot 1.

## Conséquences

- un seul opérateur d'observation, donc les résultats archivés restent
  comparables aux futurs ;
- les campagnes déjà commitées ne sont pas invalidées et n'ont pas à être
  relancées ;
- le lot 2 hérite de six ambiguïtés explicitement nommées plutôt que de les
  redécouvrir ;
- les noms d'artefacts s'écartent du §4.2 ; l'écart est documenté en référence
  plutôt que corrigé au prix de la traçabilité.
