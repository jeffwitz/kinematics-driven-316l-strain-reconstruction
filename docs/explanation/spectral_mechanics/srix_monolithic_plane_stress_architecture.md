# SRIX monolithique en contrainte plane généralisée

## État de faisabilité

Le comportement SRIX actuellement compilé est un comportement MFront
`@DSL Implicit` sous l’hypothèse `Tridimensional`. Son système local implicite
contient les inconnues constitutives `deel` et `dg[12]`. Les six composantes de
déformation sont des composantes du gradient imposé au comportement.

Les trois composantes transverses ne sont donc pas, dans ce comportement,
des inconnues locales indépendantes auxquelles on pourrait simplement ajouter
les équations

```text
sigma_zz = 0
sigma_yz = 0
sigma_xz = 0
```

sans modifier le générateur de code MFront.

## Vérification du point d’extension

La branche TFEL utilisée pour le prototype contient bien l’énumération et le
plomberie de l’hypothèse `GeneralisedPlaneStress`. Sa propre documentation
précise toutefois que le générateur MFront ne fournit pas encore le système
local à trois inconnues nécessaire.

Le commentaire de `AbstractBehaviourDSL::isModellingHypothesisSupported`
indique également que les hypothèses de contrainte plane nécessitent du code
spécifique dans les DSL `Implicit` et `RungeKutta`. La présence de
`GENERALISEDPLANESTRESS` dans les dimensions et dans le parseur ne constitue
donc pas une implémentation du Newton monolithique.

## Conséquence

Le backend C++ batch actuel reste explicitement une fermeture externe : il
appelle le comportement 3D pour chaque point et résout les trois contraintes
transverses autour de cet appel. Il est utile pour la qualification et pour
réduire l’orchestration Python, mais il ne satisfait pas le contrat
« un unique Newton MFront » et ne doit pas être présenté comme tel.

La réalisation monolithique nécessite une évolution de TFEL/MFront qui génère,
pour cette hypothèse :

1. les trois inconnues de déformation transverses ;
2. les trois résidus de contrainte plane ;
3. les couplages de ces résidus avec `deel` et `dg` ;
4. le tangent cohérent du système augmenté ;
5. l’interface MGIS batch correspondante.

Cette évolution doit être faite dans le générateur MFront, pas dans le
wrapper applicatif. Tant qu’elle n’est pas disponible, le backend condensé
MGIS reste la référence et le backend C++ batch reste expérimental.

## Résultats disponibles en attendant

Le backend C++ batch déjà qualifié conserve les mêmes champs que la référence
condensée à environ `1e-11` relatif sur P43 M100 EBSD. En quatre threads, il
réduit le temps du backend natif scalaire de `188,05 s` à `76,09 s`, mais reste
plus lent que la référence condensée à `56,88 s`. Ces résultats ne constituent
pas une qualification du backend monolithique.

## Décision

Ne pas sélectionner le backend batch C++ comme backend SRIX par défaut et ne
pas lancer M200 sur la base de cette optimisation. Le prochain chantier
recevable est une extension ciblée du générateur TFEL/MFront, avec un prototype
à un point et une vérification du tangent avant toute intégration spectrale.
