# Historical record. Superseded for current scientific interpretation.

Read `validation/tann_fcc_recovery_strategy.md` before using this document.

# MISSION — construire le premier modèle constitutif causal TANN-FCC piloté directement par la DIC

Tu travailles dans le dépôt :

    jeffwitz/kinematics-driven-316l-strain-reconstruction

Tu pars d'un contexte complètement vierge.

Tu NE DOIS PAS te fier à ce prompt comme substitut au dépôt : ce prompt fixe
le raisonnement scientifique et les décisions prises depuis les derniers
documents, mais l'état réel du code, des fichiers, des branches et des commits
est celui du checkout courant.

Ton objectif n'est PAS de poursuivre l'inverse Krylov.

Ton objectif est de construire le premier prototype scientifiquement propre
d'identification constitutive directe :

    DIC + EBSD
        -> loi constitutive causale TANN sur 12 systèmes FCC
        -> équilibre mécanique global
        -> déplacement prédit
        -> comparaison DIC

avec une règle fondamentale :

    l'état constitutif au pas n+1 DOIT provenir de l'état au pas n
    par une évolution temporelle / de chemin explicitement imposée
    par l'architecture.

Aucun état plastique ou latent ne peut être réidentifié librement image par image.

Le TANN doit être le cœur du modèle.

La spatialité CNN/cristallographique viendra ensuite comme enrichissement du
TANN et ne doit PAS être le premier moyen de produire la plasticité.


======================================================================
0. AVANT TOUTE MODIFICATION : REPRISE À FROID DU DÉPÔT
======================================================================

Commence obligatoirement par :

    git status
    git branch --show-current
    git log --all --decorate --oneline -40

Puis lis EN ENTIER :

    Claude.md
    validation/dic_driven_plastic_identification.md

Puis retrouve et lis les documents/résultats correspondant au minimum à :

    validation/tensor_local_inverse_results.md
    validation/tensor_local_inverse_preregistration.md
    validation/local_coefficient_inverse_results.md
    validation/full_field_operator_gate.md

ainsi que les documents de phase-space, clustering, FCC/slip et closure
constitutive ajoutés récemment.

Cherche également dans :

    validation/
    artifacts/
    results/
    scripts/
    docs/

les artefacts JSON, CSV, PNG et Markdown produits par les commits récents.

En particulier, retrouve dans `git log --all` les travaux correspondant aux
préfixes/étapes suivants si présents :

    b57914d   famille tensorielle libre / non-identifiabilité
    f3865bd   clustering de l'espace de phase
    b95014b   analyse continue de l'espace de phase
    e5b5116   shared FCC slip-law ladder
    102596f   benchmark final de fermeture temporelle multi-estimateurs

Les SHA complets ou messages ont pu évoluer : ne considère pas leur absence
sous ce préfixe comme une raison de les ignorer. Cherche par contenu, noms
d'artefacts et historique.

IMPORTANT :
le `Claude.md` actuellement visible peut être antérieur aux derniers commits.
Il faut donc confronter ce qu'il dit aux résultats plus récents.

Ne modifie PAS encore `Claude.md`.

Avant de coder, écris dans ton compte-rendu de travail une synthèse de 20 à
40 lignes de ce que le dépôt dit ACTUELLEMENT.


======================================================================
1. CONCLUSIONS SCIENTIFIQUES DÉSORMAIS FIGÉES
======================================================================

Ces points ne sont plus des hypothèses de travail. Ils résultent des campagnes
déjà réalisées. Ne les rouvre pas sans résultat nouveau qui les contredise.

----------------------------------------------------------------------
1.1 La reconstruction Krylov est un inverse cinématique, pas un état constitutif
----------------------------------------------------------------------

La chaîne étudiée était :

    DIC
      -> eigenstrain admissible expliquant la cinématique
      -> tentative de lecture d'une loi constitutive locale

Cette deuxième flèche a échoué.

Le champ reconstruit par Krylov doit désormais être appelé, suivant le contexte :

    effective inelastic eigenstrain
    ou
    effective inelastic correction

et PAS :

    true plastic strain
    ground-truth plastic strain
    constitutive state

La raison structurelle est :

    u_DIC = A eps_inel

avec un noyau non trivial de A.

Une famille tensorielle libre a montré :

    19 directions nulles sur 192
    conditionnement ~3.5e16
    récupération bloquée à ~31 % pendant la descente
    exact least squares ~80 % d'erreur de jauge
    TSVD meilleur plancher ~52 %

et une eigenstrain uniforme est rigoureusement invisible dans la formulation
Dirichlet intérieure.

Donc :

    eps_inel
    et la contrainte reconstruite à partir de eps_inel

ne sont pas uniques.

La forte qualité d'ajustement Krylov ne doit plus être confondue avec une
identification constitutive.

----------------------------------------------------------------------
1.2 Aucun apprentissage constitutif ne doit utiliser Krylov comme target
----------------------------------------------------------------------

INTERDIT :

    network -> eps_p_Krylov
    network -> gamma_Krylov
    network -> reconstructed internal variables

Les champs Krylov pourront rester des DIAGNOSTICS de ce que la DIC autorise.

Ils ne doivent jamais être utilisés comme supervision de la loi constitutive.

----------------------------------------------------------------------
1.3 L'espace FCC à 12 systèmes est en revanche une représentation utile
----------------------------------------------------------------------

Les analyses récentes ont projeté la correction effective dans les 12 systèmes
FCC.

Cette décomposition est redondante et les gamma^alpha ne sont pas uniques,
mais plusieurs résultats ont été reproductibles :

- la structure observée est relativement stable entre au moins deux jauges
  admissibles de décomposition ;
- l'écart entre jauge L2 et jauge temporelle sur la loi partagée reste faible
  (max ~0.025 sur le benchmark concerné) ;
- une même loi partagée entre les 12 systèmes atteint environ 88--90 % de la
  meilleure performance obtenue avec des lois séparées.

Ce résultat est important.

Il suggère que :

    les 12 systèmes ne doivent PAS être traités comme 12 classes arbitraires

mais comme 12 réalisations cristallographiquement équivalentes d'un même
mécanisme.

La numérotation 1..12 n'a aucune signification constitutive intrinsèque.

----------------------------------------------------------------------
1.4 Les tentatives de fermeture locale de l'espace de phase ont échoué
----------------------------------------------------------------------

Résumé des résultats déjà obtenus :

Analyse tensorielle continue :
    environ 400 000 échantillons actifs
    angle moyen correction/stress ~86 degrés
    R² conditionnel d'amplitude in-sample ~0.24--0.26

Clustering :
    l'orientation EBSD structure fortement les clusters
    mais les clusters ne prédisent quasiment pas la réponse
    aucune preuve de régimes constitutifs discrets

Projection FCC :
    tau^alpha et Delta gamma^alpha montrent une structure visuelle importante
    mais une partie de la géométrie deux-quadrants vient de la contrainte
    tau^alpha Delta gamma^alpha >= 0 utilisée dans certaines décompositions.

Shared-law ladder LOSO :

    S1 : tau seul
         R² ~0.11--0.12

    S2 : tau + Gamma propre
         R² ~0.16--0.18

    ajout des Gamma des autres systèmes :
         performance dégradée

La dégradation en grande dimension n'est PAS une preuve d'absence physique
d'écrouissage latent.

Elle dit seulement que les Gamma cumulés bruts ne sont pas les bons états
internes pour cet estimateur.

----------------------------------------------------------------------
1.5 Les résistances statiques et mémoires inventées à partir des gamma ont échoué
----------------------------------------------------------------------

Une résistance du type :

    r^alpha = r0
              + a Gamma^alpha
              + c sum_beta!=alpha Gamma^beta

n'améliore pas la fermeture.

Le fit pousse les coefficients vers zéro.

Des mémoires scalaires dynamiques simples, saturantes ou signées, ne
récupèrent pas non plus une loi transférable.

----------------------------------------------------------------------
1.6 Le passé local explicite n'a pas fermé la loi
----------------------------------------------------------------------

Des fenêtres temporelles contenant par exemple :

    tau_n
    tau_{n-1}
    Delta tau_n
    Delta gamma_{n-1}
    etc.

ont été testées.

Le premier résultat kNN montrait une performance qui se dégradait avec la
longueur de fenêtre.

Deux objections méthodologiques ont ensuite été contrôlées :

1. curse of dimensionality du kNN ;
2. mauvaise représentation du signe de tau.

Le benchmark final a utilisé plusieurs familles d'estimateurs et une
représentation corrigée.

Ordres de grandeur à retrouver dans les artefacts du dépôt :

                       kNN       ridge      boosting

    baseline            .036      .079       .068
    + 1 pas             -.010      .026       .030
    + Delta gamma prev  -.036      .018       .013
    + 2 pas             -.111     -.030      -.053

Les chiffres exacts du dépôt font autorité si une différence apparaît.

Conclusion :

    la meilleure fermeture LOSO reste extrêmement faible
    et l'ajout du passé observé ne la sauve pas.

Il ne faut PAS inventer une nouvelle cinquième mémoire à partir de Krylov.

----------------------------------------------------------------------
1.7 Conclusion de toute cette campagne
----------------------------------------------------------------------

La conclusion scientifique à conserver est :

    Krylov permet de reconstruire des corrections inélastiques compatibles
    avec une DIC déjà observée.

mais :

    Krylov ne fournit PAS un état constitutif permettant de prédire
    l'incrément suivant.

Donc :

    inverse cinématique != identification constitutive

La prochaine loi doit être jugée directement par :

    état interne causal
       -> plasticité
       -> équilibre
       -> déplacement
       -> DIC

et non par comparaison avec une pseudo-vérité plastique reconstruite.


======================================================================
2. QUESTION SCIENTIFIQUE DE CE NOUVEAU TRAVAIL
======================================================================

Nous voulons maintenant répondre à :

    Une loi locale causale, portée par les 12 systèmes FCC et possédant
    ses propres variables internes latentes, peut-elle produire,
    après équilibre mécanique, les champs DIC expérimentaux ?

L'idée centrale est :

    la continuité temporelle ne doit PAS être quelque chose que le réseau
    découvre éventuellement.

Elle doit être imposée structurellement.

Si :

    Y_n

est l'état constitutif, le seul moyen d'obtenir :

    Y_{n+1}

doit être :

    Y_{n+1} = Integrate(F_theta, Y_n, loading n->n+1)

Il est interdit d'avoir :

    Y_{n+1} = Network(DIC_{n+1})

ou :

    Y_{n+1} = free_parameter[n+1]

ou toute autre forme permettant à chaque image de posséder son état plastique
indépendant.


======================================================================
3. ARCHITECTURE CIBLE : T0 = TANN-FCC CAUSAL, SANS CNN SPATIAL
======================================================================

La première architecture doit être volontairement minimale.

Nom de travail :

    TANN-FCC-T0

Elle doit contenir :

    - la mécanique existante ;
    - la géométrie des 12 systèmes FCC issue de l'EBSD ;
    - une mémoire constitutive latente ;
    - une évolution causale ;
    - une construction thermodynamiquement admissible ;
    - aucune convolution spatiale apprise.

Le CNN cristallographique est explicitement différé.

----------------------------------------------------------------------
3.1 Discrétisation spatiale existante
----------------------------------------------------------------------

Respecter la cinématique du dépôt.

La représentation mécanique actuelle utilise notamment :

    TwoSubcellDiagnostic2D

avec deux sous-cellules triangulaires par pixel.

Les historiques matériaux doivent donc rester nativement de la forme :

    [nx, ny, 2, ...]

et non être interpolés sur une grille différente.

L'orientation EBSD est portée par le pixel et partagée par les deux
sous-cellules du pixel, sauf si le dépôt contient désormais une convention
plus précise.

Ne reconstruis PAS une nouvelle cinématique.

Réutilise les opérateurs existants.

----------------------------------------------------------------------
3.2 Géométrie FCC
----------------------------------------------------------------------

Pour chaque pixel x, l'EBSD donne une orientation R(x).

À partir de cette orientation, construire/réutiliser exactement les 12 systèmes
octaédriques FCC déjà utilisés par les lois SRIX/Méric du dépôt :

    s^alpha(x) : direction de glissement
    m^alpha(x) : normale au plan
    P^alpha(x) = 1/2 (s tensor m + m tensor s)

Ne crée PAS une deuxième convention FCC si une convention existe déjà.

Il faut au contraire établir un test de cohérence entre :

    géométrie TANN-FCC
    géométrie MFront/SRIX/Méric existante.

Sur un ensemble d'orientations et de contraintes aléatoires, vérifier que :

    tau^alpha = sigma : P^alpha

coïncide avec la convention existante à la précision numérique attendue.

Important :

    les 12 systèmes utilisent TOUS le même réseau constitutif.

Aucun identifiant `alpha` ou one-hot "system 7" n'est permis comme feature.

----------------------------------------------------------------------
3.3 EBSD invalide
----------------------------------------------------------------------

Le dépôt a déjà identifié les pixels EBSD non indexés et une stratégie de
remplissage/validity flag.

Ne réimplémente pas cela silencieusement.

Cherche notamment les briques autour de :

    schmid_channels

ou leur successeur.

Réutilise exactement le masque/fill qualifié et garde un validity flag.

----------------------------------------------------------------------
3.4 Variables d'état
----------------------------------------------------------------------

Première proposition minimale :

pour chaque point matériel et chaque système alpha :

    gamma^alpha       slip signé
    z^alpha in R^d    état latent interne

avec :

    d configurable
    d = 2 pour le premier vrai run

Donc typiquement :

    gamma shape = [nx, ny, 2, 12]
    z shape     = [nx, ny, 2, 12, 2]

Initialisation au premier état :

    gamma_0 = 0
    z_0     = 0

Ne rends PAS z_0 libre spatialement.

Un champ latent initial libre recréerait exactement le problème d'identifiabilité
que l'on cherche à éviter.

Ne donne PAS encore un nom physique aux composantes de z.

En particulier, ne les appelle pas :

    isotropic hardening
    backstress
    dislocation density

tant que les données ne le montrent pas.

Dans le code :

    latent_state
    z
    internal_latent_state

sont des noms appropriés.

----------------------------------------------------------------------
3.5 Déformation plastique
----------------------------------------------------------------------

La déformation plastique doit provenir des slips :

    eps_p = sum_alpha gamma^alpha P^alpha

et les incréments :

    d eps_p = sum_alpha d gamma^alpha P^alpha.

La plasticité incompressible est alors héritée des systèmes de glissement.

Pour la mécanique 2D en contrainte plane :

    sigma_zz = sigma_xz = sigma_yz = 0.

Le travail résolu est donc calculable à partir des composantes in-plane :

    tau^alpha = sigma : P^alpha.

Ne fais PAS passer les P^alpha dans l'ancien
`PLANE_STRESS_PLASTIC_GAUGE` comme s'ils étaient une eigenstrain 2D arbitraire.

Ce sont de vrais tenseurs de Schmid FCC.

Utilise une condensation plane-stress cohérente avec la mécanique existante.

Comme l'élasticité est linéaire, audite d'abord la condensation existante avant
d'écrire quoi que ce soit.

----------------------------------------------------------------------
3.6 Élasticité
----------------------------------------------------------------------

Pour T0 :

    ne réapprends PAS l'élasticité.

L'échelle élastique doit rester fixée.

Raison :

avec des conditions majoritairement/entièrement cinématiques et aucune
réaction mesurée, laisser le réseau modifier simultanément rigidité et
plasticité introduirait une liberté inutile et peu identifiable.

Pour la première qualification, privilégier l'élasticité déjà utilisée dans le
solveur DIC actuel.

Si le dépôt permet proprement l'élasticité cubique orientée sans modifier
l'expérience, garde-la comme comparaison T0b ultérieure.

Mais NE CHANGE PAS simultanément :

    architecture constitutive
    + élasticité isotrope -> cubique
    + spatialité

dans le premier test.

On veut isoler le rôle du TANN causal.

----------------------------------------------------------------------
3.7 Forme thermodynamique
----------------------------------------------------------------------

Le TANN ne doit pas reposer uniquement sur une pénalité :

    ReLU(-D)

ajoutée à la loss.

La thermodynamique doit être encodée dans l'architecture.

Une première construction recommandée est la suivante.

État généralisé par système :

    q^alpha = [gamma^alpha, z_1^alpha, ..., z_d^alpha]

Énergie libre :

    Psi = Psi_elastic(eps - eps_p)
          + Psi_h(z)

Pour T0, commence volontairement avec un stockage latent simple et stable :

    Psi_h = 1/2 sum_alpha ||z^alpha||^2

avant d'introduire éventuellement une énergie latente apprise.

Les forces thermodynamiques généralisées sont :

    A^alpha = - dPsi / dq^alpha.

La composante associée à gamma contient la force résolue tau^alpha.

Construire ensuite une mobilité dissipative :

    M_theta^alpha = L_theta^alpha (L_theta^alpha)^T

où le réseau partagé produit les coefficients d'une matrice triangulaire
inférieure L.

L'évolution est :

    dq^alpha/dlambda
        = loading_scale
          M_theta^alpha A^alpha

d'où automatiquement :

    D = sum_alpha
        (A^alpha)^T M_theta^alpha A^alpha
      >= 0.

Cette construction a un avantage important sur :

    tau^alpha dot_gamma^alpha >= 0

imposé système par système :

elle autorise une partie du travail à être stockée dans les variables internes
latentes.

On ne force donc pas artificiellement chaque système à être dissipatif
indépendamment de son stockage interne.

Si une construction TANN déjà présente dans le dépôt offre une garantie plus
propre, elle peut être utilisée, mais elle doit satisfaire la même propriété :

    dissipation non négative PAR CONSTRUCTION.

Documenter mathématiquement la preuve dans un ADR.

----------------------------------------------------------------------
3.8 Réseau partagé entre systèmes
----------------------------------------------------------------------

Même réseau pour les douze systèmes.

Une première architecture raisonnable :

    local embedding phi_theta(A^alpha, z^alpha, ...)
    permutation invariant pool:
        c = mean_beta phi_theta(...)
    mobility network:
        L^alpha = rho_theta(local_alpha, c)

Ainsi les systèmes peuvent interagir au même point matériel sans que leur
numérotation ait de sens.

Cette interaction est :

    intra-point / inter-slip-system

et NON :

    spatiale.

Ne mélange pas encore les deux.

Architecture initiale recommandée :

    latent_dim = 2
    hidden width = 32
    2 hidden layers
    SiLU ou activation lisse équivalente
    float64 pour les qualifications

Le but n'est pas de chercher une énorme capacité.

Le premier modèle doit rester petit et analysable.

----------------------------------------------------------------------
3.9 Symétries à respecter
----------------------------------------------------------------------

Minimum obligatoire :

1. permutation des 12 systèmes :

si l'on permute l'ordre des systèmes en entrée et inverse la permutation en
sortie, le résultat physique doit être identique.

Aucune couche ne doit dépendre du numéro alpha.

2. rotation cristallographique :

elle est déjà traitée par les P^alpha calculés depuis l'EBSD.

Ne donne PAS les angles d'Euler bruts au réseau T0.

3. changement de convention de numérotation :

doit être couvert par le test de permutation.

Le problème plus subtil :

    s^alpha -> -s^alpha
    gamma^alpha -> -gamma^alpha

est une jauge de représentation.

Si la convention FCC du dépôt est canonique et figée, cette invariance de signe
n'a pas à bloquer T0, mais elle doit être notée explicitement dans l'ADR comme
un point à traiter avant une architecture générale exportable.


======================================================================
4. CONTINUITÉ TEMPORELLE / DE CHEMIN : C'EST LE POINT CENTRAL
======================================================================

Ne considère PAS le temps comme une simple feature.

INTERDIT :

    network(state_n, frame_index)
    network(state_n, n/40)

Le modèle ne doit pas apprendre :

    image 31 -> image 32.

Il doit apprendre l'évolution d'un état matériel sous un chemin mécanique.

----------------------------------------------------------------------
4.1 Pas de temps physique artificiel
----------------------------------------------------------------------

L'essai est quasi statique.

Ne donne donc pas au réseau une "seconde" artificielle par image.

Utilise une paramétrisation par chemin de chargement.

Pour un incrément n -> n+1, définir une trajectoire locale :

    eps(s) = eps_n + s Delta eps
    s in [0,1]

avec :

    d eps/ds = Delta eps.

Le TANN évolue le long de cette trajectoire.

Pour éviter qu'un simple changement de subdivision numérique modifie la loi,
le champ de vecteurs doit être construit de façon à dépendre du chemin de
déformation et non de la durée arbitraire de l'incrément.

Une solution pratique :

    rate_scale = ||Delta eps||

et utiliser :

    dq/ds = rate_scale *
            F_theta(q, eps(s), direction_of_Delta_eps, ...).

Ainsi :

    Delta eps = 0

implique exactement :

    Delta q = 0.

La direction :

    Delta eps / ||Delta eps||

ne doit être calculée qu'avec une protection numérique propre autour de zéro.

----------------------------------------------------------------------
4.2 Intégrateur
----------------------------------------------------------------------

Pour T0, privilégier un intégrateur déterministe et différentiable à nombre
fixe de sous-pas.

Par exemple RK4.

Ne commence PAS avec un solveur ODE adaptatif dont les changements de pas
créent :

    branches numériques
    non-déterminisme
    difficulté d'adjoint
    discontinuités de gradient.

Paramètre :

    n_constitutive_substeps

Commencer avec par exemple :

    4

et qualifier :

    1
    2
    4
    8

sur les tests matériau.

----------------------------------------------------------------------
4.3 Critère essentiel : invariance au sous-pas
----------------------------------------------------------------------

Pour une trajectoire donnée :

    Delta eps

comparer :

    un incrément complet
    deux demi-incréments
    quatre quarts d'incrément

en conservant exactement le même chemin linéaire.

L'état final et la contrainte finale doivent converger vers le même résultat.

Cette propriété est centrale.

Elle constitue une preuve beaucoup plus utile de continuité constitutive que la
simple corrélation entre deux images DIC.

Aucune campagne P43 ne doit commencer avant que cette propriété soit qualifiée.


======================================================================
5. CONTRAT DU MATÉRIAU TANN-FCC
======================================================================

Le matériau doit suivre les mêmes règles transactionnelles que les autres lois.

État accepté :

    q_n

Une évaluation pendant Newton reçoit :

    q_n committed
    eps_trial

et calcule :

    sigma_trial
    q_trial
    C_alg = d sigma_trial / d eps_trial

mais NE MODIFIE PAS q_n.

Seul :

    commit()

après convergence globale peut transformer :

    q_n <- q_trial.

`revert()` doit restaurer exactement le dernier état accepté.

Une line-search, un matvec Jacobien, un calcul d'adjoint ou un essai
d'optimisation ne doit jamais modifier l'histoire constitutive.

Tests obligatoires :

    evaluate twice -> bitwise/near-bitwise identical
    evaluate + revert -> committed state unchanged
    failed Newton -> committed state unchanged
    commit once -> correct new state
    second commit without accepted increment -> forbidden or no-op explicit


======================================================================
6. TANGENTE ALGORITHMIQUE
======================================================================

Le solveur global Newton a besoin de :

    d sigma_{n+1} / d eps_{n+1}

à état committed n fixé.

Le TANN + intégrateur doit donc exposer une tangente algorithmique exacte ou
obtenue par automatic differentiation.

Ne fais PAS de finite differences de la loi à chaque point dans le vrai solveur.

Les différences finies servent uniquement à QUALIFIER la tangente.

Audit d'abord les dépendances du dépôt :

    pyproject.toml
    requirements
    environnement courant.

Si un framework AD existe déjà, réutilise-le.

Sinon, PyTorch est un choix acceptable pour les JVP/VJP locaux, mais :

    NE PORTE PAS tout le solveur spectral dans PyTorch.

La mécanique globale qualifiée reste la mécanique globale.

Le backend TANN peut utiliser AD localement et exposer des tableaux NumPy au
reste du solveur.

Si PyTorch doit être ajouté :

    le rendre optionnel
    ne pas casser l'import minimal de fem_inhouse
    documenter l'extra/dépendance
    ne pas installer silencieusement une stack GPU.

Qualification :

sur des états aléatoires physiquement raisonnables :

    C_alg_AD

contre différence finie centrée.

Erreur relative cible :

    <= 1e-5

et idéalement :

    <= 1e-6

avec étude du pas de perturbation.


======================================================================
7. INTÉGRATION AU SOLVEUR GLOBAL
======================================================================

Réutiliser le solveur Dirichlet spectral existant.

Ne reconstruis PAS un nouveau FEM.

Ne remplace PAS l'équilibre par une PINN.

La chaîne est :

    boundary DIC
        -> harmonic/full Dirichlet lifting existant
        -> Newton-GMRES
        -> TANN-FCC aux points matériau
        -> equilibrium
        -> u_sim interior.

Le TANN fournit une loi.

Le solveur fournit la mécanique.

C'est précisément cette séparation que nous voulons.


======================================================================
8. DONNÉE D'APPRENTISSAGE : UNIQUEMENT LA DIC OBSERVABLE
======================================================================

La DIC intérieure ne doit jamais être une entrée du TANN.

Entrées autorisées :

    orientation EBSD statique
    géométrie FCC
    état interne simulé précédent
    contrainte simulée
    déformation simulée
    incrément de déformation du solveur
    futur contexte spatial SIMULÉ lorsque les CNN seront ajoutés

Interdit :

    u_DIC intérieur courant en entrée du réseau
    eps_DIC intérieur courant en entrée
    eps_p_Krylov
    gamma_Krylov
    numéro absolu du pixel
    coordonnées x,y pour aider le réseau à mémoriser le champ
    numéro d'image

La DIC intervient dans :

    la condition de Dirichlet mesurée
    la fonction objectif intérieure APRES résolution mécanique.

Les conditions de bord mesurées font partie du problème mécanique.

Elles peuvent donc être utilisées à un incrément tenu hors loss.

Ce n'est pas une fuite.

En revanche, les déplacements intérieurs du même incrément ne doivent pas être
utilisés pour recalibrer l'état avant la prédiction.


======================================================================
9. LOSS DIC
======================================================================

La loss principale doit être une loss sur les déplacements, pas sur une
déformation DIC dérivée.

Utiliser préférentiellement le modèle de bruit DIC déjà qualifié.

Le dépôt documente un repeat final et une PSD de bruit.

S'il existe déjà un whitener spectral propre :

    W_D

réutilise-le.

Sinon construis/qualifie avant apprentissage :

    L_DIC =
        1/2 || W_D (u_sim - u_DIC) ||²

sur les DOF intérieurs observés.

Ne construis PAS une covariance dense.

Important :

    aucun terme de force/réaction.

Il n'existe pas de série de réaction expérimentale utilisable dans ce projet.

Ne crée pas une pseudo-force à partir de la simulation.


======================================================================
10. APPRENTISSAGE SÉQUENTIEL
======================================================================

Une trajectoire d'apprentissage doit être jouée du début à la fin :

    q_0
      -> solve state 1 -> q_1
      -> solve state 2 -> q_2
      ...
      -> solve state 40 -> q_40.

Il est interdit de réinitialiser q entre les états.

C'est précisément ce qui différencie ce modèle de Krylov.


======================================================================
11. HOLDOUT : NE PAS DEMANDER UNE EXTRAPOLATION DE PLASTICITÉ NON VUE
======================================================================

NE PAS utiliser comme test principal :

    train 21..33
    test 34..40.

L'essentiel de la plasticité se développe tardivement.

Ce split testerait principalement :

    extrapolation hors domaine d'état

et non :

    qualité de la loi dans un domaine d'état couvert par l'apprentissage.

Le test principal doit être un masked-state/interleaved holdout réparti sur
l'histoire.

Une proposition de split primaire :

    holdout = [24, 28, 32, 36, 39]

avec l'état final 40 conservé dans l'apprentissage afin que le domaine de forte
plasticité soit vu.

AVANT de fixer définitivement ces indices :

    tracer une grandeur OBSERVABLE monotone du chargement
    et/ou le défaut par rapport à l'élasticité

et vérifier que les holdouts couvrent :

    faible
    intermédiaire
    forte plasticité

sans être hors enveloppe.

Ne choisis PAS les indices à partir des performances du TANN.

Une fois le split fixé :

    l'écrire dans validation/tann_fcc_preregistration.md
    avant le premier entraînement réel.

Le split historique :

    [24, 28, 32, 36, 40]

peut être calculé secondairement pour comparaison avec les anciens travaux,
mais le state 40 tenu hors apprentissage constitue un test d'extrémité de
domaine et ne doit pas être le critère principal.


======================================================================
12. SÉMANTIQUE DU HOLDOUT DANS UN MODÈLE CAUSAL
======================================================================

Pour un état holdout h :

on utilise normalement :

    q_{h-1}
    boundary_DIC_h

on résout :

    q_{h-1}
        -> TANN
        -> equilibrium
        -> u_pred_h
        -> q_h_pred

mais :

    u_DIC_h interior

n'entre PAS dans la loss d'entraînement.

Ensuite :

    q_{h+1}

part OBLIGATOIREMENT de :

    q_h_pred

et jamais d'un état recalé avec la DIC h.

Important :

si un état ultérieur h+1 ou h+2 appartient au train, sa loss peut indirectement
contraindre les paramètres de la dynamique ayant traversé h.

C'est volontaire.

Le test principal est donc :

    temporal interpolation with missing observations

et non :

    online forecasting.

Écrire cette distinction clairement dans la documentation.


======================================================================
13. DIFFÉRENTIATION DU SOLVEUR : NE PAS UNROLLER NEWTON/GMRES
======================================================================

Ne différencie PAS naïvement à travers toutes les itérations Newton et GMRES.

Le dépôt possède déjà des opérateurs matrix-free et des adjoints.

Construire un adjoint discret de la trajectoire.

À l'incrément n :

    R_n(u_n ; q_{n-1}, theta) = 0

et le matériau fournit après convergence :

    q_n = Q_n(u_n, q_{n-1}, theta).

Loss locale éventuelle :

    ell_n(u_n).

Pour la rétropropagation dans le temps, définir un co-état matériau :

    v_n = d future_loss / d q_n.

L'adjoint mécanique lambda_n vérifie schématiquement :

    (dR/du)^T lambda_n
        =
        dell/du
        + (dQ/du)^T v_n.

Puis :

    v_{n-1}
        =
        (dQ/dq_prev)^T v_n
        - (dR/dq_prev)^T lambda_n

et la contribution paramétrique :

    dL/dtheta +=
        (dQ/dtheta)^T v_n
        - (dR/dtheta)^T lambda_n.

Vérifie soigneusement les signes à partir du Lagrangien discret avant
implémentation : les équations ci-dessus donnent la structure, pas un prétexte
pour copier des signes sans dérivation.

Les VJP locaux :

    Q_u^T v
    Q_q^T v
    Q_theta^T v
    R_theta^T lambda

peuvent être calculés par AD au niveau du batch matériau.

L'adjoint mécanique reste matrix-free.

Il faudra probablement exposer :

    tangent matvec
    transpose tangent matvec

avec les tangentes constitutives locales correspondantes.

Ne suppose pas que le Jacobien appris restera symétrique.


======================================================================
14. QUALIFICATION OBLIGATOIRE DE L'ADJOINT
======================================================================

Avant P43 :

construire un problème minuscule, par exemple :

    8x8
    ou
    16x16

sur 2--4 incréments.

Comparer :

    gradient adjoint dL/dtheta

à :

    central finite difference

pour plusieurs paramètres/rayons aléatoires de theta.

Cible :

    erreur relative <= 1e-4

et chercher <=1e-5 si les tolérances Newton le permettent.

Faire également les produits scalaires :

    <J v, w>
    vs
    <v, J^T w>

sur la mécanique TANN complète.

Cible :

    ~1e-8 ou mieux en float64

si les solves sont suffisamment serrés.

Aucun entraînement long n'est autorisé avant ce test.


======================================================================
15. QUALIFICATIONS MATÉRIAU AVANT DONNÉES RÉELLES
======================================================================

Créer un script dédié, par exemple selon les conventions du dépôt :

    qualify_tann_fcc_material.py

Il doit vérifier au minimum :

A. zéro incrément

    Delta eps = 0
        -> Delta gamma = 0
        -> Delta z = 0
        -> état identique

B. dissipation

sur plusieurs milliers d'états aléatoires :

    D >= -tol

Tolérance absolue/normalisée documentée.

Il ne doit pas y avoir "8 % de puissance négative".

L'admissibilité doit tenir à l'arrondi près.

C. permutation des systèmes

permutation aléatoire des 12 systèmes :

    résultat permuté
        == permutation du résultat original

D. subdivision

    one step
    2 half steps
    4 quarter steps
    8 eighth steps

convergence vers le même résultat.

E. tangente

AD vs finite differences.

F. transaction

evaluate / revert / commit.

G. comparaison géométrique

tau^alpha et P^alpha contre la convention FCC existante.


======================================================================
16. PREMIER RUN RÉEL : P43 100x100
======================================================================

Ne commence PAS directement par 200x200.

Le 100x100 a déjà tous les artefacts de comparaison et permet de développer
l'architecture à coût raisonnable.

Ce run répond à UNE question :

    un TANN-FCC causal local peut-il reproduire de la DIC tenue hors loss
    mieux que l'élasticité ?

Il ne doit PAS chercher :

    le meilleur réseau possible
    un CNN
    l'interaction entre grains
    l'élasticité cubique en même temps
    un énorme sweep d'hyperparamètres.


======================================================================
17. MÉTRIQUES DU RUN P43
======================================================================

Rapporter au minimum par état :

1. loss déplacement brute

2. loss déplacement blanchie par bruit

3. métrique relative au défaut élastique :

    E_n =
       ||u_model - u_DIC||
       -------------------
       ||u_elastic - u_DIC||

ou la forme strictement compatible avec la métrique existante du dépôt.

Toujours documenter si la métrique porte sur :

    déplacement
    strain
    Kelvin norm
    weighted norm.

Ne mélange pas les anciennes métriques sans l'indiquer.

4. train vs holdout

    mean
    median
    max

5. dissipation généralisée

    min D
    mean D
    total D

6. amplitude des slips

    sum |Delta gamma|
    distribution par système
    distribution spatiale

7. variables latentes

uniquement comme diagnostic :

    distributions
    évolution temporelle
    saturation éventuelle

PAS comme preuve qu'un z a une signification physique.

8. convergence mécanique

    Newton
    GMRES
    cutbacks éventuels
    temps constitutif
    temps total.


======================================================================
18. CRITÈRE DE VIABILITÉ SCIENTIFIQUE DE T0
======================================================================

Le critère minimal n'est PAS :

    faire aussi bien que Krylov.

Krylov résout un autre problème et possède une liberté incompatible avec une
loi constitutive.

Le critère minimal est :

    median(E_holdout) < 1

et idéalement :

    amélioration sur au moins 4 des 5 états holdout.

Interprétation :

    E < 1 :
        la loi causale apprend une information constitutive utile au-delà de
        l'élasticité.

    E ~ 1 :
        aucune information constitutive prédictive démontrée.

    E > 1 :
        la loi dégrade la prédiction.

On peut déclarer comme "signal fort", SANS en faire une barrière artificielle :

    median(E_holdout) ~0.7 ou moins.

Ne modifie PAS ce seuil après résultat pour annoncer un succès.

Même en cas d'échec, conserver les résultats.

Un échec T0 serait scientifiquement utile :

    il montrerait qu'une loi locale causale FCC reste insuffisante
    et fournirait une justification propre à l'introduction de spatialité.


======================================================================
19. PAS DE CNN DANS T0
======================================================================

C'est volontaire.

On veut d'abord savoir ce que produit :

    mémoire causale + FCC + thermodynamique + équilibre.

Le CNN ne doit PAS pouvoir masquer un problème du TANN.

Cependant, prépare une INTERFACE VIDE pour l'extension future.

Par exemple conceptuellement :

    SpatialContextProvider.forward(
        committed_or_predicted_state,
        crystal_geometry,
        grain_ids
    ) -> context_per_point_per_system

T0 fournit :

    context = 0

Le TANN accepte le contexte dans sa signature mais celui-ci est nul.


======================================================================
20. ARCHITECTURE FUTURE À DOCUMENTER, MAIS PAS ENCORE À IMPLÉMENTER
======================================================================

Dans l'ADR, documenter T1/T2/T3 pour éviter que la prochaine session reparte
sur un CNN générique.

T1 :
    contexte spatial statique EBSD seulement

T2 :
    convolution cristallographique intragranulaire

T3 :
    transport intergrain equivariant.

Principe du futur opérateur spatial :

pour chaque système alpha :

    t^alpha(x)
      = normal_surface cross m^alpha(x)

donne la trace du plan de glissement sur la surface.

Le futur opérateur pourra échantillonner :

    x +/- r_j t^alpha

avec des distances par exemple :

    1, 2, 4, 8 pixels

et apprendre une fonction radiale partagée.

La géométrie :

    t^alpha

vient de l'EBSD.

Les poids :

    w_theta(r)

sont appris.

IMPORTANT :

le futur CNN/opérateur spatial ne doit JAMAIS produire directement :

    Delta gamma
    eps_p
    z_{n+1}

Il doit uniquement fournir un contexte :

    c_spatial

au TANN.

La seule transition temporelle reste :

    TANN:
        Y_n -> Y_{n+1}.

C'est une contrainte architecturale permanente.

----------------------------------------------------------------------
20.1 Premier futur opérateur : intragranulaire
----------------------------------------------------------------------

Dans T2, ne traverser les joints de grain qu'après qualification.

Le kernel suit les directions du grain courant mais est masqué lorsque
l'échantillon sort du grain.

----------------------------------------------------------------------
20.2 Transport intergrain
----------------------------------------------------------------------

Ne jamais faire correspondre naïvement :

    system 3 grain A
    avec
    system 3 grain B.

La numérotation locale est arbitraire.

Le futur T3 devra utiliser un matching/transport basé sur la géométrie relative
des systèmes :

    s_A^alpha . s_B^beta
    m_A^alpha . m_B^beta
    etc.

et idéalement une formulation equivariant/gauge-like.

Mais ceci est HORS SCOPE du présent travail.


======================================================================
21. CE QU'IL NE FAUT ABSOLUMENT PAS FAIRE
======================================================================

1. Ne pas ressusciter Krylov comme target.

2. Ne pas utiliser eps_p_Krylov comme pré-entraînement.

3. Ne pas créer un z_n libre pour chaque état.

4. Ne pas réinitialiser l'état latent entre images.

5. Ne pas donner frame index au réseau.

6. Ne pas donner x,y au réseau.

7. Ne pas donner la DIC intérieure courante comme feature.

8. Ne pas utiliser de force/réaction fictive.

9. Ne pas faire train 21:33 / test 34:40 comme validation principale.

10. Ne pas commencer par un CNN/U-Net.

11. Ne pas faire un PINN qui remplace l'équilibre existant.

12. Ne pas changer simultanément TANN + anisotropie élastique + CNN.

13. Ne pas déclarer les z comme variables physiques identifiées.

14. Ne pas confondre thermodynamique de la variable effective Krylov avec
    thermodynamique du nouveau modèle.

15. Ne pas faire de loss sur les slips.

16. Ne pas tester la loi en comparant une carte gamma ou PEEQ à la DIC.

17. Ne pas cacher un run négatif.

18. Ne pas modifier des critères après avoir regardé les résultats.


======================================================================
22. STRUCTURE DE CODE SUGGÉRÉE
======================================================================

Adapte aux conventions réellement présentes.

Une organisation plausible serait :

    src/fem_inhouse/constitutive/tann_fcc.py
        architecture matériau et énergie

    src/fem_inhouse/constitutive/tann_fcc_geometry.py
        interface vers orientations et systèmes FCC

    src/fem_inhouse/constitutive/tann_fcc_batch.py
        évaluation batch + transactions + tangentes

    src/fem_inhouse/identification/tann_fcc_sequence.py
        trajectoire complète DIC

    src/fem_inhouse/identification/tann_fcc_adjoint.py
        adjoint temporel/global

    src/fem_inhouse/identification/spatial_context.py
        interface future, ZeroSpatialContext pour T0

    scripts/qualify_tann_fcc_material.py

    scripts/qualify_tann_fcc_adjoint.py

    scripts/train_tann_fcc_p43.py

Mais NE crée pas ces chemins aveuglément si le dépôt possède déjà une meilleure
hiérarchie.


======================================================================
23. TESTS UNITAIRES / INTÉGRATION À AJOUTER
======================================================================

Tests rapides :

    test_tann_zero_increment
    test_tann_dissipation_nonnegative
    test_tann_system_permutation_equivariance
    test_tann_substepping
    test_tann_algorithmic_tangent_fd
    test_tann_transaction_revert
    test_fcc_geometry_matches_existing_convention
    test_zero_spatial_context_no_effect

Tests solveur :

    test_tann_global_jacobian_transpose
    test_tann_small_dirichlet_converges
    test_tann_state_advances_only_on_commit

Tests identification :

    test_tann_sequence_masked_state_no_dic_leak
    test_tann_sequence_state_not_reset_at_holdout
    test_tann_sequence_adjoint_fd

Pour le test de fuite DIC :

modifier arbitrairement `u_DIC_interior` d'un état holdout SANS changer son bord.

La prédiction forward de cet état doit être strictement identique.

Seule la métrique d'évaluation doit changer.


======================================================================
24. DONNÉES / PERFORMANCE
======================================================================

Ne lance pas un gros entraînement avant de profiler.

Mesure séparément :

    material forward
    material tangent
    Newton
    GMRES
    adjoint
    VJP theta
    stockage histoire

Sur 100x100 :

    100 x 100 x 2 = 20 000 points matériau.

Avec :

    12 systèmes
    gamma + latent_dim=2

cela représente déjà :

    20 000 x 12 x 3

scalaires d'état par incrément.

Vérifier le budget mémoire sur 40 états.

Si nécessaire :

    checkpointing temporel
    recomputation contrôlée

mais ne passe pas en float32 pour la mécanique avant qualification.


======================================================================
25. 200x200
======================================================================

P43 200x200 est le prochain banc naturel, mais seulement APRÈS T0 qualifié sur
100x100.

Le passage 100 -> 200 quadruple le nombre de points et augmente surtout
l'information spatiale/microstructurale.

Pour T0 local, il sert à vérifier la reproductibilité avec davantage de grains.

Pour T2 spatial, il deviendra particulièrement important.

Ne le lance que si :

    - matériau qualifié
    - adjoint qualifié
    - run 100x100 reproductible
    - coût mesuré acceptable.


======================================================================
26. DOCUMENTATION À PRODUIRE
======================================================================

AVANT le premier entraînement P43 :

créer :

    validation/tann_fcc_preregistration.md

qui contient :

    question scientifique
    architecture exacte
    latent_dim
    réseau
    intégrateur
    subdivision
    normalisations
    loss
    split holdout
    métriques
    critères
    seed
    backend
    version logicielle

Créer également un ADR, suivant la convention du dépôt, par exemple :

    docs/.../ADR-xxxx-tann-fcc-causal-identification.md

Le titre doit refléter :

    "Causal TANN-FCC constitutive identification from DIC"

L'ADR doit expliquer :

    pourquoi Krylov n'est plus une cible
    pourquoi le temps est une transition d'état et non une feature
    pourquoi les 12 systèmes partagent la même loi
    pourquoi la thermodynamique est architecturale
    pourquoi la spatialité est différée
    comment les futurs CNN pourront entrer sans bypasser le TANN.


======================================================================
27. ARTEFACTS À ARCHIVER
======================================================================

Chaque vrai run P43 doit produire un JSON autosuffisant contenant :

    git SHA
    dirty status
    date
    machine/backend
    seed
    crop
    states
    holdout
    network architecture
    latent_dim
    normalisations
    integrator/substeps
    optimizer
    losses
    metrics per state
    train aggregate
    holdout aggregate
    Newton iterations
    Krylov iterations
    timings
    dissipation metrics
    slip metrics
    state continuity metrics

Sauvegarder aussi :

    training history
    model checkpoint
    configuration exacte

Les figures doivent être générées depuis cet artefact, pas par valeurs copiées
manuellement.


======================================================================
28. FIGURES MINIMALES
======================================================================

Pour le premier rapport T0 :

Figure A :
    E_n vs state
    elastic = 1
    distinction train / holdout

Figure B :
    u/strain DIC vs elastic vs TANN
    pour quelques états faible/moyen/fort

Figure C :
    residual spatial DIC
    elastic vs TANN

Figure D :
    total slip activity
    sum_alpha |Delta gamma^alpha|
    comme diagnostic seulement

Figure E :
    dissipation généralisée
    distribution / évolution

Figure F :
    évolution de quelques composantes latentes
    sans interprétation physique

Figure G :
    test de subdivision
    stress/state error vs substep count.

Pas de figure présentant :

    gamma_TANN vs gamma_Krylov

comme validation.


======================================================================
29. COMPARAISONS À CONSERVER
======================================================================

Comparer au minimum à :

    elastic baseline

Puis, si trivial à rejouer sans développement :

    SRIX actuel
    Méric actuel

sur exactement la même DIC Dirichlet et les mêmes états.

Krylov peut apparaître dans un tableau séparé intitulé par exemple :

    "kinematic inverse reference -- not constitutive"

pour rappeler son niveau d'ajustement.

Mais jamais dans la colonne "ground truth".


======================================================================
30. SI T0 ÉCHOUE
======================================================================

Ne corrige pas immédiatement l'échec en ajoutant un CNN.

D'abord vérifier :

    gradient adjoint
    transaction
    tangent
    subdivision
    thermodynamique
    capacité minimale du réseau
    fuite / absence de fuite
    split
    optimisation

Si tout est qualifié et que :

    E_holdout >= 1

alors le résultat scientifique est :

    une loi locale causale TANN-FCC, sous cette capacité et cette représentation,
    ne suffit pas à expliquer la DIC.

Ce sera précisément le point de départ légitime pour T1/T2 :

    spatial conditioning.

Documenter l'échec avant de changer d'architecture.


======================================================================
31. SI T0 FONCTIONNE
======================================================================

Si le TANN local donne un gain robuste :

    E_holdout < 1

alors :

1. figer le checkpoint / artefact T0 ;
2. rerun avec plusieurs seeds ;
3. tester latent_dim 1 / 2 / 4 seulement ensuite ;
4. vérifier la stabilité des prédictions, pas la stabilité brute des z ;
5. seulement après, ouvrir T1/T2.

Le prochain test scientifique sera alors :

    TANN local
       vs
    TANN + contexte cristallographique spatial

avec le MÊME cœur TANN.

Le gain mesurera directement la valeur de l'information spatiale.


======================================================================
32. MISE À JOUR DE Claude.md
======================================================================

À LA FIN seulement, après résultats réellement obtenus :

mettre à jour `Claude.md`.

Il doit devenir utilisable pour une reprise à froid.

La nouvelle section "état courant" doit dire clairement :

1. Krylov est fermé comme piste constitutive.

2. Les raisons quantitatives :
       non-identifiabilité
       échec phase-space LOSO
       échec mémoire
       échec fermeture multi-estimateurs.

3. Les 12 systèmes FCC restent une représentation robuste.

4. La nouvelle architecture est :
       causal TANN-FCC
       equilibrium
       DIC loss.

5. Si T0 a été exécuté :
       résultats exacts
       artefact
       commit
       verdict.

6. CNN spatial :
       futur enrichissement
       pas encore un producteur indépendant de plasticité.

Ne supprime pas l'historique de ce qui a été réfuté.

Marque les sections anciennes comme superseded lorsque nécessaire.


======================================================================
33. QUALITÉ LOGICIELLE
======================================================================

À la fin :

    ruff
    mypy selon la couverture du dépôt
    pytest tests ciblés
    puis suite complète raisonnable
    sphinx-build -W si la documentation le requiert.

Ne transforme pas les warnings antérieurs sans rapport en side quest.

Si un test préexistant échoue :

    déterminer s'il était déjà rouge
    documenter
    ne pas masquer.


======================================================================
34. COMMITS
======================================================================

Faire des commits scientifiques cohérents et lisibles.

Séquence suggérée :

1.
    docs: preregister causal TANN-FCC identification

2.
    feat: add thermodynamic causal FCC material core

3.
    test: qualify TANN-FCC tangent state and dissipation

4.
    feat: couple TANN-FCC to the Dirichlet spectral solver

5.
    feat: add sequential discrete adjoint for TANN-FCC

6.
    test: qualify TANN-FCC sequence adjoint

7.
    experiment: run causal TANN-FCC on P43

8.
    docs: record TANN-FCC identification verdict

N'invente pas un commit vide pour respecter cette liste.

Adapte-la au travail réel.


======================================================================
35. FORMAT DU RAPPORT FINAL À L'UTILISATEUR
======================================================================

À la fin de ton travail, ne donne pas une simple liste de fichiers.

Réponds dans cet ordre :

A. Ce que tu as trouvé en reprenant le dépôt
    notamment éventuel décalage entre Claude.md et les commits récents.

B. Architecture réellement implémentée.

C. Vérifications numériques :
    thermodynamique
    subdivision
    tangente
    adjoint
    transactions.

D. Résultat P43 si exécuté :
    train
    holdout
    métriques
    temps
    figures.

E. Verdict scientifique :
    T0 apporte-t-il une information constitutive prédictive ?

F. Ce que cela autorise ou interdit pour T1/T2.

G. SHA final.

Ne dis jamais "cela marche" uniquement parce que la loss d'entraînement descend.


======================================================================
36. PRINCIPE DIRECTEUR À NE JAMAIS PERDRE
======================================================================

Tout le travail précédent peut être résumé par :

    DIC + équilibre
        peut sélectionner une correction cinématiquement admissible

mais :

    cette correction ne devient pas spontanément une histoire constitutive.

Le nouveau modèle doit donc imposer :

    CAUSALITÉ TEMPORELLE
        par le TANN

    MÉCANISMES CRISTALLOGRAPHIQUES
        par les 12 systèmes FCC/EBSD

    THERMODYNAMIQUE
        par l'architecture

    ÉQUILIBRE
        par le solveur global existant

    OBSERVATION
        par la DIC uniquement dans la loss.

La spatialité future viendra comme :

    CONTEXTE

et jamais comme un raccourci permettant de réidentifier indépendamment
la plasticité de chaque image.

En une équation :

    Y_0
      --TANN/equilibrium--> Y_1, u_1
      --TANN/equilibrium--> Y_2, u_2
      ...
      --TANN/equilibrium--> Y_40, u_40

et l'optimisation cherche theta tel que :

    u_n(theta) ~= u_DIC,n

sur les états observés,

tout en demandant aux états masqués :

    u_h(theta)

d'être corrects SANS jamais avoir utilisé leur DIC intérieure pour reconstruire
Y_h.

C'est cela que nous voulons démontrer ou réfuter.
