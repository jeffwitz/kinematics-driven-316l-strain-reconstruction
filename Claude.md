# Plan de mise à niveau de `fem_inhouse`

Dernière mise à jour : 2026-08-23
Objectif de maturité : **au moins 4/5 sur tous les axes**

## REPRISE À FROID PRIORITAIRE — SRIX-REGM

**Commencer par lire `validation/srix_regm_worklog.md`.** Ce fichier est le
journal autoritatif du nouveau chantier d'identification SRIX par défaut
d'équilibre faible reconditionné. Il contient les gates, les contrats
mécaniques réutilisés, les artefacts et la décision GO/NO-GO. Aucun calcul P43
ni apprentissage TANN n'est autorisé avant validation du jumeau numérique et
du classement REGM/FEMU.

Le chemin historique REGM était : histoire de déplacement connue depuis l'état 0,
cinématique `TwoSubcellDiagnostic2D`, replay SRIX causal sans tangente de sortie,
résidu intérieur faible `B^T sigma`, correction `-K0^-1 f`, observation et
whitening DIC. `K0` est assemblé et factorisé une seule fois par problème. La
SVD de la jacobienne résiduelle en coordonnées logarithmiques est un résultat
scientifique central, pas un diagnostic facultatif.

Le jumeau exact M8 est maintenant positif et documenté dans
`validation/srix_regm_twin_results.md` : résidu RMS vrai `1.474e-13 mm`, résidu
initial `3.143e-8 mm`, résidu identifié `1.412e-13 mm`, erreur logarithmique
projetée `0.248 %`. Une évaluation REGM coûte `2.90 s` contre `124.48 s` pour
la trajectoire directe (`43.0 x`). La quatrième direction, essentiellement le
contraste `Q/b`, est toutefois très faible (`conditionnement 2.15e4`). Les
diagnostics REGM sont désormais clos : les prochains développements doivent
différencier le vrai résidu/tangent du solveur FEMU, sans réutiliser les
opérateurs REGM ; P43 reste interdit.

Le benchmark 32 états (`validation/srix_regm_scaling_results.md`) donne
`1.270 s` sur M20 et `19.708 s` sur M100. À M100, `18.693 s` sont dans le replay
matériau contre `0.447 s` dans `K0^-1` : ne pas développer un nouvel inverse
FFT. La condensation 3D externe demande encore des tangentes localement pour
fermer la contrainte plane ; comparer le backend GPS déjà qualifié avant toute
optimisation supplémentaire.

---

## REPRISE À FROID PRIORITAIRE — TANN-FCC corrigé, apprentissage suspendu

**Commencer par lire `validation/tann_fcc_recovery_strategy.md`.** C'est la
source autoritative sur le TANN-FCC. Les anciens fichiers
`tann_fcc_preregistration.md`, `tann_fcc_primary_run_results.md` et
`tann_fcc_amended_run_status.md` sont conservés comme preuves historiques,
mais leurs conclusions opérationnelles sont supersédées.

### Verdict actuel

Il n'existe pas encore de TANN-FCC entraîné et scientifiquement qualifié. Le
run primaire était presque élastique ; le run amendé 100 x 100 n'a pas terminé
une trajectoire d'apprentissage. En outre, l'audit du 23 août a trouvé quatre
défauts invalidant l'interprétation des anciens artefacts : état constitutif
non remis au départ entre pas Adam, observation de la DIC une seconde fois,
interface engineering/Kelvin incorrecte en cisaillement et fermeture transverse
hybride. La figure EVM mélangeait également DIC absolue et FEM incrémentale.

Ces défauts sont corrigés dans le code courant et couverts par des tests :

- chaque rollout repart du même état exact et chaque record porte son vrai
  état précédent ;
- la perte est `O(u_FEM) - u_DIC` et utilise l'adjoint exact du transfert
  affine-préservant ;
- le profil différentiable provient de `legacy_script_2021`, pas du profil V4 ;
- les conversions engineering/Kelvin portent déformation, contrainte, tangente
  et cotangentes de l'adjoint ; la tangente acceptée est archivée dans la
  convention engineering attendue par l'adjoint mécanique ;
- la fermeture plane-stress élimine les cisaillements élastiques transverses et
  est testée contre Hooke 3D et la dérivée de l'énergie condensée ;
- l'histoire mécanique est rejouée depuis l'état 0 ; 1--20 sont warm-up et les
  états interpolés 31--32 ne sont jamais scorés ;
- poids et état Adam sont checkpointés après chaque étape ; un restart partiel
  de trajectoire est interdit pour l'apprentissage car il perdrait la
  sensibilité de l'état initial aux paramètres.

### Prochaine action autorisée

**Gate B : jumeau numérique constitutif**, sur petit domaine, avec SRIX/Méric
comme vérité connue, plusieurs orientations et chemins incluant du cisaillement,
observation DIC et bruit qualifiés, puis holdout d'un chemin/orientation. Aucun
nouveau run P43 long n'est autorisé avant ce gate.

La stratégie nominale n'est plus d'apprendre une mobilité entièrement libre
depuis une seule histoire de déplacements. Elle est : loi FCC physique
qualifiée + correction TANN bornée et sans dimension ; paramètres d'échelle
fixés indépendamment ; contexte spatial seulement si le modèle local passe les
gates d'histoire/amplitude mais échoue sur une morphologie observable. P43 ne
fournissant pas de force, l'échelle absolue de contrainte/mobilité n'y est pas
identifiable seule. `sigma_ref` est donc un paramètre de dynamique constitutive,
pas une simple normalisation numérique.

### Fichiers à lire, dans cet ordre

1. `validation/tann_fcc_recovery_strategy.md` ;
2. `src/fem_inhouse/constitutive/tann_fcc.py` ;
3. `src/fem_inhouse/identification/tann_fcc_sequence.py` ;
4. `src/fem_inhouse/identification/tann_fcc_adjoint.py` ;
5. `src/fem_inhouse/identification/dic_whitening.py` ;
6. `scripts/train_tann_fcc_p43.py` ;
7. les trois tests `test_tann_fcc*.py` sous `tests/unit/constitutive/` ;
8. `validation/reference_data/dic_multistep_history_p0043_repaired_v1/report.json`.

### Environnement local

Utiliser directement `.venv/bin/python` et `.venv/bin/fem-inhouse`. Pour les
tests MFront/MGIS, sourcer `/home/jeff/.local/share/tfel/env/env.sh`; l'installation
est sous `/home/jeff/.local`. Définir ensuite
`MFRONT_BEHAVIOUR_LIBRARY=$PWD/build/mfront/src/libBehaviour.so` et
`SRIX_GENERIC_MFRONT_BEHAVIOUR_LIBRARY=$PWD/build/srix-generic/src/libBehaviour.so`.
Ces bibliothèques sont distinctes : la bibliothèque principale ne contient pas
le symbole `Fcc316LForestRubinSrixGeneric3D`. La section détaillée
« environnement scientifique installé » plus bas reste autoritative.

---

## [HISTORIQUE SUPERSEDÉ] ÉTAT TANN-FCC causal au 2026-08-18

**Point d'entrée unique : `validation/tann_fcc_preregistration.md`**
(architecture figée + amendements 1-4, barres, holdout, marge de bruit)
et `validation/archive/tann_fcc_initial_mission.md`. Les résultats : `validation/tann_fcc_primary_run_results.md`
(verdict du run enregistré), `validation/tann_fcc_amended_run_status.md`
(état du run amendé), artefact `validation/_generated/shared_tensor_generator/tann_fcc_p43_run.json`
+ figures `validation/figures/tann_fcc_p43/`.

### La voie en cours

Krylov est clos comme piste constitutive (inverse cinématique ≠ état
constitutif : noyau non trivial, 19 directions nulles sur 192, échec
phase-space/mémoire/fermeture multi-estimateurs). Les 12 systèmes FCC
restent la représentation. L'architecture est le **TANN-FCC causal** :
état interne `q = [gamma; z]` par point et système, évolution
`Y_{n+1} = Integrate(F_theta, Y_n, Delta eps)` (GENERIC : `M = L L^T`,
`D >= 0` par construction), équilibre par le solveur spectral existant,
DIC uniquement dans la perte. Chaîne complète construite, qualifiée et
committée : matériau (portes A-G vertes), séquence masquée (fuite DIC
testée bitwise), adjoint discret (vérifié FD 1e-8..1e-4 aux rayons
5x/20x/50x), figures A-G.

### Verdict T0 enregistré (run primaire, seed 20260817)

`median(E_holdout) = 1.052` — barres 1 et 2 **échouées**. Diagnostic
structurel quantifié : à `sigma_ref = 2 mu` (Amendement 1) la force
normalisée `A/sigma_ref ~ 8e-4` colle la mobilité softplus à son
plancher 0.693, le slip par incrément est `~8e-7` contre `~1e-3`
requis, et le gradient adjoint exact (`3.8e-9`) colle à l'estimation
par règle de chaîne : la loi est inerte quelle que soit la capacité.
Amendement 3 inscrit et testé : `sigma_ref = 200 MPa` (échelle
plastique) — les portes restent vertes, la loi bouge.

### État du run amendé — campagne arrêtée (2026-08-19)

25x25 complet converge (E_holdout = 1.34 à l'init — la plasticité forte
non entraînée fait pire que l'élastique, mais la loi répond). Le
100x100 converge 1-17 (limiteur + tolérance 1e-8) puis bute au 18 :
problème d'équilibre du solveur global, PAS de raideur de l'intégrateur
(benchmark Radau vs RK4 vs Euler implicite : hypothèse raide réfutée,
`validation/tann_fcc_integrator_benchmark.md`). Checkpointing par
incrément disponible (`--resume-increment`, reprise approximative
documentée). Figure EVM expé/calcul (`validation/figures/tann_fcc_p43/EVM_exp_vs_calc.png`) :
accord 1 % à l'état 40 (normalisation pixel corrigée), états faibles
dominés par le plancher métrologique — l'élastique est indépendant du
module en Dirichlet, l'écart EST la signature plastique à expliquer.
Prochain levier ouvert pour une session future : convergence des
incréments tardifs du solveur (line search, GMRES, trust region).

### Architecture du code (commits a1284ab → 91d4caa)

`src/fem_inhouse/constitutive/tann_fcc.py` (matériau, tangente AD
chunkée, VJP d'incrément), `tann_fcc_geometry.py` (EBSD → systèmes,
remplissage sentinelle), `identification/tann_fcc_sequence.py`
(trajectoire masquée), `tann_fcc_adjoint.py` (adjoint discret),
`identification/spatial_context.py` (extension T1/T2, contexte nul en
T0). Scripts : `qualify_tann_fcc_material.py`,
`train_tann_fcc_p43.py`, `figure_tann_fcc_p43.py`. Tests :
`tests/unit/constitutive/` (14 tests verts).

---

## [SUPERSEDÉ le 2026-08-18 — voir l'état courant ci-dessus] identification plastique pilotée par la DIC (2026-08-16)

**Point d'entrée unique pour reprendre à froid :
`validation/dic_driven_plastic_identification.md`.** Ce document porte le
problème, la chaîne mécanique, les conventions et leurs pièges, les
emplacements de données, ce qui est établi, ce qui est réfuté, et la suite. La
présente section n'en est que le résumé.

### Où nous en sommes

L'objet que l'on cherchait à réduire était le mauvais. Toutes les tentatives de
représentation réduite **fixe** ont échoué — POD globale, POD par bande de
Laplace, autoencodeur convolutif, champ neuronal implicite, inpainting
morphologique — et c'est la prémisse commune qui est en cause, non les
méthodes : rien n'oblige le champ plastique à vivre sur une variété globale de
faible dimension. Ce qui doit être de faible dimension est **l'espace des
corrections plastiques admissibles autour de l'état mécanique courant**.

```text
réfuté      eps_p_n = Phi a_n              un Phi(x) fixe pour tous les états
en cours    eps_p_n = Phi_theta(S_n) a_n   une base que l'état engendre
```

Un générateur convolutif reçoit l'état du **prédicteur** — contrainte,
déformation, plastique accumulée, chemin plastique, en Kelvin — et produit `r`
directions plastiques plein champ ; l'équilibre en choisit les coefficients ; la
DIC n'intervient que dans la perte. Ni DIC intérieure ni coordonnée en entrée.

### Résultats, P43 100x100, holdout temporel

Métrique : part du défaut élastique qui subsiste, `1.0` = pas mieux que
l'élasticité.

| base | r=4 | r=8 | r=16 | puissance négative | chi |
|---|---|---|---|---|---|
| krylov fixe | 0.602 | 0.392 | 0.245 | 42-44 % | +0.016 à +0.033 |
| J2 imposé à la main | 0.895 | 0.876 | 0.855 | 26-28 % | +0.39 à +0.41 |
| direction apprise, libre | 0.608 | 0.547 | 0.506 (partiel) | 37-43 % | non mesuré |
| apprise, dissipative par construction | 0.621 | 0.651 | 0.587 | **8-11 %** | +0.31 à +0.36 |

Trois faits structurent la lecture. **Qualité d'ajustement et plausibilité
physique sont fortement anticorrélées** : Krylov gagne toutes les colonnes
d'ajustement et est l'objet le moins physique du tableau. **Le rang de Krylov
n'achète pas de travail plastique** — sa dissipation nette reste à 2.0e3 aux
trois rangs pendant que la puissance absolue double, donc les modes
supplémentaires ajoutent de la plasticité positive et négative qui s'annule.
**Apprendre la direction bat largement la prescrire**, 0.62 contre 0.89 à rang
égal.

Le cône `{C a >= 0}` sur modes non projetés est **exactement {0}**, mesuré par
LP de faisabilité. Cela clôt la question ouverte depuis la campagne Krylov —
ces QP renvoyaient `a = 0` parce que le cône l'était, pas par défaut de
solveur. En revanche cela ne montre **pas** que projeter chaque mode soit
nécessaire : seulement qu'une combinaison linéaire globale de ces modes bruts
ne rencontre le cône qu'à l'origine. La condition peut aussi porter sur le
champ assemblé, `d eps_p = P_H(Phi a)`, avec modes bruts et coefficients
libres — c'est le dernier contrôle purement géométrique avant la
cristallographie.

### Changement de cap — P43 redevient un banc de qualification

Le démonstrateur 100x100 a servi son objet et **ne prépare pas le plein champ**.
Deux arithmétiques l'imposent.

La mécanique ne passe pas à l'échelle telle qu'implémentée : le solveur direct
creux coûte 12 ms à 19 602 degrés de liberté et environ `N^1.5`, donc **~460 s
par résolution** à 22,3 millions, avec une factorisation qui ne tient pas en
mémoire. Une Green par FFT est l'alternative : un aller-retour 3600x3100 mesure
**616 ms**, donc quelques secondes par application de `A`.

Et les coefficients réduits ne passent pas à l'échelle **physiquement** : seize
nombres globaux décrivent plausiblement quelques grains, pas des milliers. Le
mur de calcul et le défaut de modélisation sont le même mur.

La première estimation — 32 min par pas de gradient, 4,4 ans par campagne —
était un artefact de la construction explicite de `A Phi` colonne par colonne.
En assemblant le champ d'abord, `v = Σ a_jk w_j φ_jk`, puis `q = P_H(v)`, le
gradient mécanique coûte **un `A` et un `A^T`**, quel que soit le nombre de
coefficients. Deux réserves honnêtes : le problème intérieur en `a` devient
itératif dès qu'on ne peut plus former les équations normales, chaque itération
coûtant `A + A^T` — d'où l'importance du warm-start et d'un active-set
semi-lisse exploitant que `P_H` est affine par morceaux. Et le gradient
extérieur n'a probablement **pas** besoin d'un second adjoint : la partie DIC de
la perte extérieure *est* le terme de données intérieur, la pénalité
d'orthogonalité ne dépend pas de `a`, et seuls le ridge (`O(1e-6)`) et la
pénalité de dissipation (poids `1e-2`) brisent l'identité. Cette dernière est
**redondante dans les bras contraints**, où la dissipation tient par
construction ; en la retirant le théorème de l'enveloppe s'applique et le budget
est divisé par deux.

### Ce qui tourne, et ce qui est acquis

Rien en permanence. Les jalons 0, 1.0, 1A et 1B sont **franchis et mesurés** ;
la suite est scientifique, plus numérique.

| jalon | verdict |
|---|---|
| **0** — opérateur plein champ | adjoint **4,4e-17** sur 22,3 M d'inconnues, 29 itérations, `T_A = 52 s`, 1,8 Go |
| **1.0** — préconditionneur sous tangent plastique | sature à 79 itérations même à contraste 10⁴ ; le verrou redouté n'existe pas |
| **1A** — vrai critère de plasticité | 14 → 44 GMRES/Newton de l'élastique au tout-plastique, ×3 et saturant |
| **1B** — coefficients locaux | 64 ou 4096 coefficients : mêmes 8 Newton, mêmes ~170 Krylov |
| **1B chaud** | une perturbation coûte **1 Newton et 19 Krylov**, contre 8 à froid |

Le goulot a changé quatre fois, désigné à chaque fois par la mesure et non par
l'intuition : cinématique FEM générique (réglé par le stencil, ×345), puis la
transformée (padding et plan mesuré, ×3,7), puis le nombre d'itérations
(warm start et tolérance, ×2,4), et enfin **l'intégration constitutive, 76 %
du temps**.

### Le constitutif : MFront threadé, décidé par la mesure

MFront mono-thread est deux à six fois plus lent que le batch NumPy vectorisé ;
à **huit threads il est 1,9× plus rapide** sur la branche plastique, et perd
encore d'un facteur deux sur la branche élastique où le travail par point est
trivial. Bout en bout à 256² sur huit incréments : **73,6 s contre 160,6**, soit
**×2,18 sur le run complet** et ×3,5 sur le terme constitutif, dont la part
tombe de 71 % à 45 %. Le point de bascule est le coût du problème local — d'où son évidence
en plasticité cristalline et sa marginalité en J2 élastique. Les bancs sont
basculés sur `--backend mfront --mfront-threads 8`.

Et pour mémoire : aucune campagne antérieure n'a comparé une loi Python à
MFront. Dans `p43_m100_backend_comparison_latest.json`, le run `python_condensed`
a pour backend `mfront-3d-condensed-plane-stress` — « python » y désigne où
tourne la boucle de condensation hôte, pas la loi.

### L'hyper-réduction RID : construite, non adoptée, et pourquoi

Voir **`validation/reduced_integration_domain_rationale.md`**, qui explique en
détail le principe, son arithmétique et la condition de son intérêt.

En bref : un domaine d'intégration réduit évalue la loi coûteuse sur une
fraction des points et la reconstruit ailleurs, l'équilibre restant plein champ.
Son plafond est fixé par la part que la loi occupe dans le run. Pour J2 c'est
76 %, ce qui borne la méthode vers **×4** — et MFront threadé en rend déjà 2,18
pour le prix d'un paramètre, ramenant la part à 45 % et le plafond restant à
**×1,68**, trop peu pour justifier la machinerie.
Pour la plasticité cristalline, où un point coûte trois ordres de grandeur de
plus, la part approche 95 % et le plafond approche `1/r`. **La méthode est
juste ; J2 est le mauvais problème sur lequel la dépenser.**

Le split exact `sigma = sigma_n + C_0:d eps + h` est implémenté et testé, prêt
pour ce jour-là.

### Deux erreurs à ne pas refaire

Les **fenêtres avec DIC imposée sur leur contour** sont un excellent matériau
d'apprentissage et jamais une preuve : la cinématique du bord contient déjà
l'effet de tout l'extérieur, donc une plasticité juste au-delà peut être
réattribuée à l'intérieur, et dix mille fenêtres résolues indépendamment ne
garantissent rien sur `B^T sigma = 0` global. L'architecture est CNN par tuiles,
champ plastique assemblé globalement, équilibre global.

L'**investissement FFT n'est pas jetable** avec la cristallographie : avec
`C(x) = C_0 + ΔC(x)`, l'inverse homogène reste le préconditionneur spectral
naturel du problème hétérogène. Et l'anisotropie élastique n'a pas à entrer en
même temps que l'EBSD — élasticité isotrope plus géométrie cristallographique
dans le générateur plastique sépare proprement les deux effets.

### Documents référencés

| fichier | contenu |
|---|---|
| `validation/dic_driven_plastic_identification.md` | **le document de reprise à froid** |
| `validation/tensor_local_inverse_results.md` | **jalon 3 : la famille tensorielle libre ajuste 5,19 décades mieux et n'est pas identifiable** — eigenstrain uniforme rigoureusement invisible, plancher 52-80 % |
| `validation/tensor_local_inverse_preregistration.md` | seuils du jalon 3, et la correction inline sur l'ordre de la branche D |
| `validation/local_coefficient_inverse_results.md` | **jalon 2 : la représentation locale est identifiable** — 0,021 % au jumeau, conditionnement 200, et la base enrichie déficiente en rang |
| `validation/shared_tensor_generator_preregistration.md` | **jalon 4 : le générateur tensoriel partagé contre le noyau mesuré, seuils figés avant exécution** |
| `validation/krylov_projected_control_results.md` | **contrôle Krylov projeté P_H^{G_p} (2026-08-17) : bat le réseau dissipatif de 0,19–0,25 à rang égal, utile mais pas « strategy-changing » (0,402 vs 0,386) ; f_0 ≈ 0,47 — la moitié de la plasticité active est déposée exactement sur la frontière de travail nul : D≥0 est nécessaire mais insuffisant, la tâche du réseau devient la sélection de direction dans le demi-espace admissible** |
| `validation/phase_space_local_law_results.md` | **analyse d'espace de phase (2026-08-17) : sur la partie observable (noyau exclu), la direction inélastique est isotrope par rapport à s (std circulaire 42,6°, R²_cond 0,24) — pas de loi locale minimale F(s, p_eq) dans cette reconstruction ; la suite enregistrée est la décomposition Δε_D/Δε_0 avant tout ajustement de loi** |
| `validation/phase_space_cluster_results.md` | **clustering espace de phase (2026-08-17) : l'orientation (Euler+Schmid) donne les clusters les plus stables (AMI 0,98) mais AUCUNE famille ne conditionne la réponse (gain 1,00, R²≤0,14) ; p_eq déstabilise (AMI 0,12) — la loi par régimes sur le champ effectif brut est fermée, la décomposition Δε_D/Δε_0 devient l'étape décisive** |
| `validation/phase_space_conditioning_results.md` | **conditionnement continu (2026-08-17) : kNN LOSO — l'amplitude porte une structure continue faible et spécifique à l'état (R²=0,13 vs 0,25 in-sample : la variable d'histoire manquante est temporelle), la direction n'est conditionnée par aucun état testé (R²_circ≤0,12, MAE 90°) ; géométrie continue non triviale confirmée, pas de régimes discrets** |
| `validation/path_memory_results.md` | **mémoire de chemin local (2026-08-17) : AUCUNE fenêtre du passé observable (1–4 pas de τ et Δγ) n'améliore — déclin monotone jusqu'à −0,19. La piste « découverte d'une loi locale dans l'espace de phase reconstruit » est fermée avec preuve ; le champ effectif est cinématiquement exact mais constitutivement vide au-delà de la corrélation in-sample ; la voie directe (générateur constitutif dans l'équilibre + validation DIC) devient la voie convaincante ; la question SRIX/Méric (structure connue, dynamique fermée) reste ouverte et distincte** |
| `docs/explanation/femu_identification.md` | **entrée opérationnelle de la voie directe : boucle FEMU-U (SRIX dans l'équilibre, 2 paramètres libres, gradients FD, L-BFGS) — opérationnelle, observable dégénéré (Dirichlet champ complet imposé par le workflow de partition, misfit ~1e-9 pour toute loi) ; la correction est l'imposition bord-seul, intérieur libre** |
| `validation/local_coefficient_inverse_preregistration.md` | seuils du jalon 2, et **la condition enregistrée de réouverture du RID** |
| `validation/full_field_operator_gate.md` | jalons 0, 1.0, 1A, 1B : opérateur, préconditionneur, non linéaire, coefficients locaux |
| `validation/reduced_integration_domain_rationale.md` | **le RID : principe, arithmétique, et pourquoi il attend la cristallographie** |
| `validation/constitutive_hyperreduction_preregistration.md` | seuils préenregistrés de l'hyper-réduction, trois régimes de certification |
| `validation/adaptive_reduced_basis_learned_flow.md` | direction d'écoulement apprise, tableau complet |
| `validation/adaptive_reduced_basis_preregistration.md` | seuils préenregistrés, dont le critère déclaré inatteignable |
| `validation/adaptive_reduced_basis_first_rung.md` | bases construites à la main, plafond du champ libre |
| `docs/explanation/spectral_mechanics/plastic_inverse_reuse.md` | la page-pont : ce que le solveur spectral fournit déjà |
| `docs/how-to/choose_mfront_backend.md` | **choisir *et appeler* MFront** : le piège du `thread_count`, le croisement mesuré, l'étiquette `python_condensed` trompeuse, les conventions génie/Kelvin, un extrait qui marche |
| `validation/morphology_reduction_findings.md` | la ligne base fixe, réfutée |
| `validation/ludwik_on_the_measured_p43_history.md` | le verdict Ludwik |

Scripts vivants : `qualify_full_field_plastic_operator.py` (le portique),
`stencil_kernel.py`, `qualify_preconditioner_under_plasticity.py`,
`bench_ludwik_plastic_fraction.py`, `bench_local_coefficients_nonlinear.py`,
`bench_warm_start_coefficients.py`, `qualify_local_coefficient_inverse.py`,
`qualify_tensor_local_inverse.py`,
`learn_flow_direction_p43.py`.
Paquets : `src/fem_inhouse/hyperreduction/`,
`src/fem_inhouse/identification/local_coefficient_inverse.py`,
`src/fem_inhouse/identification/tensor_local_inverse.py`.

### Ce qui reste ouvert, par ordre

1. **Le noyau ne fait plus l'objet d'une campagne d'énumération ; la décision
   d'architecture est prise et pré-enregistrée le 2026-08-17** dans
   `validation/shared_tensor_generator_preregistration.md` : la structure
   apprise partagée est le candidat, testé contre le noyau mesuré par des
   portes figées (transversalité, jumeau à part invisible contrôlée, stabilité
   de la composante noyau entre graines). Une eigenstrain uniforme produit un
   déplacement rigoureusement nul, et plus généralement toute contrainte propre
   auto-équilibrée est invisible : 19 directions sur 192, conditionnement 3,5e16,
   plancher de reconstruction 52 à 80 % quelle que soit la méthode. Ce n'est pas
   un défaut d'optimisation. La décision d'architecture vit désormais là.
2. **Reconstruire le relèvement élastique** avec conversion assertée, puis la
   comparaison des familles sur la **vraie** DIC. Aucun chiffre des
   `scripts/*_p43.py` historiques n'est réutilisable pour cela.
3. **Le générateur**, conçu *contre* le noyau mesuré et non autour. Le jalon 3
   change son statut : la structure apprise partagée n'est pas un dispositif
   d'efficacité, c'est ce qui rend la représentation locale identifiable.
   **Jalon 4 pré-enregistré le 2026-08-17** : quatre bras (A1 J2 imposé,
   A2 appris libre, A3 appris dissipatif, A4 tenseur libre), holdout temporel
   {24, 28, 32, 36, 40}, huit portes figées dont la stabilité de la composante
   noyau entre graines.
   L'orthogonalisation de la base enrichie `q > 1` est abandonnée — elle
   enrichirait l'amplitude, pas la direction tensorielle.
4. **Le calcul global plein champ**, volontairement différé : `T_A = 52 s` et
   ~78 min pour trois incréments.
5. **La cristallographie**, qui déclenchera le RID — dont la condition de
   réouverture est enregistrée et chiffrée, et qu'aucune mesure du jalon 2
   n'a modifiée.

### Deux défauts consignés, non réparés

* **Le relèvement élastique d'une douzaine de `scripts/*_p43.py` n'équilibre
  pas.** Raideur Kelvin, déformation de génie et divergence Voigt enchaînées
  sans conversion, ce qui double la contrainte de cisaillement : 32 % du résidu
  intérieur subsiste là où la forme convertie atteint 4e-16. Tout résidu mesuré
  contre cette référence est affecté, dont le « défaut élastique à 0.29 » et les
  sous-espaces de Krylov. Les deux scripts vivants convertissent et l'assertent.
* Le bras libre au rang 16 n'a jamais fini et précède la sauvegarde des poids,
  donc il n'est pas re-notable.

### Garde-fous permanents

* `A` est **surjectif** : ajuster la DIC ne prouve rien, seules les contraintes
  portent de l'information.
* Une erreur retenue proche de zéro est une **fuite**, pas un succès.
* Ne jamais déplacer un seuil préenregistré après avoir vu les résultats ; si
  l'instrument est incapable de l'atteindre, le démontrer par un témoin
  indépendant et le consigner comme tel.
* Les conditions aux limites mesurées sont l'énoncé du problème : jamais
  renormalisées vers un chargement idéalisé.
* Une borne ne s'énonce jamais sans ses hypothèses.
* `p_eq` est `sqrt(z^T G z)` avec `PLANE_STRESS_PLASTIC_GAUGE`, jamais
  `norm(z)`.

---

## [SUPERSEDÉ le 2026-08-16 — voir l'état courant ci-dessus] Priorité scientifique — oracle mécanique compatible avec la DIC

### Constat expérimental à ne plus masquer

Les solveurs directs J2, SRIX et Méric sont désormais suffisamment qualifiés
numériquement pour que leur désaccord avec l'expérience ne puisse plus être
attribué par défaut à un simple défaut de Newton, de tangent, de transaction ou
de couplage non local. Les figures P43 M100 actuelles montrent que le non-local
agit bien sur les variables plastiques, mais qu'il ne résout pas le problème
modèle-expérience :

- en J2, le maximum de PEEQ diminue d'environ 44 % entre le local et le
  non-local et les zones plastiques restent très corrélées entre elles ;
- en SRIX, le maximum de `Gamma` diminue d'environ 50 % et la localisation est
  redistribuée ;
- malgré cela, la déformation totale équivalente J2 reste faiblement corrélée
  à la DIC P43 sur le cas tracé (`Spearman` voisin de `0.15`, recouvrement des
  5 % de hotspots voisin de `0.02`).

La DIC est une mesure de déplacement/déformation totale. Elle n'est ni une
PEEQ ni un glissement cumulé. Une carte PEEQ ou `Gamma` ne doit donc jamais
être soustraite directement à une carte DIC. Le constat pertinent est que le
champ de déplacement équilibré produit par les lois actuelles ne reproduit pas
correctement la géographie expérimentale observée.

**Conséquence : la priorité n'est plus d'ajouter une nouvelle loi directe.** Il
faut construire une plasticité J2 associée dont l'incrément plastique scalaire
est libéré, afin d'obtenir un oracle mécanique expérimental qui projette la DIC
vers un état simultanément proche des mesures et compatible avec l'équilibre.
Cet oracle servira ensuite à recaler toute loi MFront, puis à mesurer honnêtement
jusqu'où la loi recalée reproduit l'équilibre et les champs expérimentaux.

### But du P0

Pour chaque incrément expérimental, rechercher simultanément :

```text
u*          champ de déplacement corrigé, proche de la DIC
Delta p*    incrément plastique J2 scalaire, positif
sigma*      contrainte associée à (u*, Delta p*)
```

sous les contraintes :

```text
epsilon* = B_h u*
R_u(u*, Delta p*) = B_h^T sigma* = 0
Delta p* >= 0
```

Le résultat attendu est un jeu d'états mécaniques propres :

```text
(epsilon*, sigma*, Delta p*, p*, u*)
```

Il ne constitue pas encore une loi constitutive identifiée. Il constitue
l'**oracle expérimental** contre lequel une loi MFront peut être recalée et
ensuite rejouée dans le solveur direct.

### Inventaire des briques

| Brique | État | Décision P0 |
|---|---|---|
| Histoire DIC plein champ | disponible | réutiliser l'histoire `(steps+1,nx,ny,2)` |
| Cinématique `u -> epsilon=B_h u` | qualifiée | utiliser `TwoSubcellDiagnostic2D`, sans reconstruction parallèle |
| Équilibre faible `sigma -> B_h^T sigma` | qualifié | utiliser l'adjoint discret existant |
| Solveur/préconditionneur spectral matrix-free | qualifié | réutiliser, sans nouveau solveur EF |
| J2 contraintes planes et écoulement associé | disponible | séparer `Delta p` de Ludwik |
| Transactions `evaluate/revert/commit` | qualifiées | imposer le même contrat au Driven J2 |
| Bruit DIC spatial | avancé | construire un whitener spectral `C_D^{-1/2}` |
| Pénalité pilotée par l'incertitude | disponible au bord | généraliser au plein champ |
| Identification de petits vecteurs | disponible | ne pas la confondre avec l'optimisation de champs |
| Optimiseur `(u,Delta p)` / KKT | absent | nouvelle brique principale |
| Force expérimentale synchronisée | absente | non bloquante pour P0, requise pour l'échelle absolue |

### Contrat du backend `Driven J2`

Le backend ne doit pas imposer Ludwik. L'optimiseur fournit `Delta p >= 0` et
la loi locale conserve strictement la direction d'écoulement J2 associée en
résolvant :

```text
F(sigma; epsilon, Delta p)
  = sigma - C:[epsilon - epsilon_p^n - Delta p n(sigma)] = 0

n(sigma) = d sigma_eq / d sigma
```

Après acceptation uniquement :

```text
epsilon_p^(n+1) = epsilon_p^n + Delta p n(sigma)
p_(n+1)         = p_n + Delta p
```

Une évaluation, une line-search ou un essai d'optimisation ne commit jamais
l'état matériau. `revert()` doit restaurer exactement l'incrément accepté
précédent. `commit()` n'est permis qu'après convergence de l'incrément global.

En posant :

```text
A = I + Delta p C : dn/dsigma
```

le backend doit exposer sans inverse explicite :

```text
dsigma/depsilon =  A^{-1} C
dsigma/dDelta_p = -A^{-1} (C:n)
```

`implicit_sensitivities.py` doit être réutilisé si son contrat couvre ce petit
système implicite. Les produits globaux restent matrix-free :

```text
delta epsilon = B_h delta u
delta sigma   = sigma_,epsilon delta epsilon
              + sigma_,Delta_p delta p
delta R_u     = B_h^T delta sigma
```

### Fonction objectif DIC

Ne pas construire de covariance dense. À partir de la densité spectrale du
bruit DIC mesuré `S_D(k)`, définir un opérateur de blanchiment spectral :

```text
FFT(W_D v)(k) = FFT(v)(k) / sqrt(S_D(k) + epsilon)
```

et :

```text
J_DIC(u) = 1/2 ||W_D (u-u_DIC)||^2.
```

Le whitener et son adjoint doivent être matrix-free, testés par produit
scalaire et compatibles avec le masque/support de l'opérateur d'observation
DIC existant. Le niveau de régularisation spectral doit venir du bruit mesuré,
pas d'une constante choisie uniquement pour faire converger.

### Problème inverse P0

Formulation cible par incrément :

```text
min_(u,Delta p) J_DIC(u) + J_prior(Delta p) + J_reg(Delta p)
sous R_u(u,Delta p)=0 et Delta p>=0.
```

Le premier solveur doit être un Lagrangien augmenté :

```text
L_rho = J_DIC + J_prior + J_reg
      + lambda^T R_u + rho/2 R_u^T P R_u,
```

où `P` réutilise autant que possible le préconditionneur mécanique spectral.
Le solveur doit conserver séparément : norme DIC blanchie, norme d'équilibre,
violation de positivité, prior, régularisation et évolution du multiplicateur.
Une diminution de la fonction totale ne suffit pas à déclarer l'équilibre.

### Prior et régularisation de `Delta p`

Ludwik ne disparaît pas : il devient un prior explicite :

```text
J_prior = lambda_J ||Delta p - Delta p_Ludwik||^2.
```

La qualification doit utiliser une continuation `lambda_J` forte vers faible,
afin d'identifier où la donnée et l'équilibre exigent de quitter Ludwik. Le
champ `Delta p* - Delta p_Ludwik` est un livrable scientifique.

Première régularisation :

```text
J_reg = lambda_s ||grad Delta p||^2
      + lambda_t ||Delta p_n-Delta p_(n-1)||^2.
```

Le Laplacien/DCT et Helmholtz existants doivent être réutilisés. Une variante
Huber/TV sera comparée ensuite, car une régularisation quadratique peut effacer
les bandes physiques de localisation. Aucun choix ne sera qualifié sans test
synthétique avec le bruit P43 réel.

### Force expérimentale

L'absence actuelle de force synchronisée n'empêche pas le P0 méthodologique.
Elle interdit en revanche de présenter l'oracle comme une identification
absolue de contrainte ou de loi. Dès que `F_exp(t_n)` est disponible, ajouter :

```text
F_calc(t_n) ~= F_exp(t_n)
```

avec sa propre incertitude. Tous les artefacts P0 doivent enregistrer :

```text
physical_force_history_available: false|true
absolute_constitutive_identification: false|true
```

### Architecture minimale

Ne pas créer quinze modules ni un solveur par loi. Le P0 ajoute trois briques :

```text
src/fem_inhouse/core/driven_j2.py
    réponse locale pilotée par Delta p
    tangentes dsigma/depsilon et dsigma/dDelta_p
    transactions

src/fem_inhouse/identification/dic_whitening.py
    PSD/covariance du bruit mesuré
    W_D et W_D^T spectraux matrix-free

src/fem_inhouse/workflows/experimental_mechanical_oracle.py
    problème incrémental (u, Delta p)
    équilibre, prior, régularisation, optimisation
```

L'oracle est indépendant de MFront. Les lois recalées restent des lois MFront
utilisées par l'infrastructure directe commune :

```text
oracle expérimental J2 piloté
        -> états (epsilon*,sigma*,Delta p*)
        -> recalage loi MFront
        -> simulation directe avec cette loi
        -> comparaison équilibre/champs/DIC
```

### Ordre d'implémentation impératif

1. Figer le constat DIC actuel et les métriques de comparaison.
2. Implémenter `DrivenJ2MaterialBatch` au point matériau.
3. Qualifier `evaluate/revert/commit`, positivité et répétabilité.
4. Vérifier `dsigma/depsilon` et `dsigma/dDelta_p` par différences finies.
5. Vérifier le produit global `B^T delta sigma` sur petit domaine.
6. Construire le whitener à partir du bruit DIC mesuré et tester son adjoint.
7. Construire un problème synthétique J2 connu.
8. Ajouter une réalisation du bruit P43 mesuré au déplacement synthétique.
9. Retrouver `u_truth`, `Delta p_truth` et `sigma_truth` sans effacer la localisation.
10. Implémenter le Lagrangien augmenté séquentiel dans le temps.
11. Exécuter P43 petit crop, puis M20, M50 et M100.
12. Produire l'oracle P43 et seulement ensuite recaler J2 Ludwik, SRIX, Méric ou
    toute autre loi MFront.
13. Rejouer chaque loi recalée en simulation directe et mesurer ce qu'elle
    reproduit ou non de l'équilibre et de la DIC.

À chaque niveau :

```text
matériau -> tangent -> opérateur global -> synthétique bruité -> P43
```

Ne jamais corriger simultanément plusieurs couches lorsqu'un niveau échoue.

### État d'implémentation du P0 au 2026-08-14

Le premier socle exécutable est maintenant intégré :

| Brique | État |
|---|---|
| `DrivenJ2PlaneStressBatch` piloté par `Delta p` | implémenté et testé |
| `dsigma/depsilon`, `dsigma/dDelta_p` implicites | qualifiés par FD |
| transactions `evaluate/revert/commit` | testées |
| résidu global `B^T sigma` et `Jv` matrix-free | implémentés et testés |
| adjoint global `J^T v` | implémenté et testé par produit scalaire |
| whitener DIC spectral | implémenté, auto-adjoint testé |
| objectif DIC + prior Ludwik + régularisations | implémenté |
| positivité de `Delta p` | imposée par bornes |
| Lagrangien augmenté séquentiel par incrément | première version implémentée |
| récupération synthétique affine sans bruit | testée |
| récupération avec bruit P43 mesuré et bandes localisées | première qualification M5 réussie |
| application P43 réelle | M20, états DIC réparés 0 à 8, première qualification réussie |

La première boucle utilise L-BFGS-B comme solveur intérieur borné. Elle ne doit
pas être considérée comme le KKT final haute performance : elle sert à valider
la formulation, les gradients, la positivité et les transactions avant le test
synthétique avec le bruit P43 réel. Le préconditionneur mécanique spectral n'est
pas encore introduit dans la métrique d'équilibre du Lagrangien augmenté.

Le premier synthétique localisé utilise une fenêtre réelle du résidu DISFlow
entre images répétées P43. Le support libre est appliqué de manière identique à
la PSD, au whitener et à son adjoint. Une mise à jour sélective de la pénalité
du Lagrangien augmenté est indispensable : augmenter systématiquement `rho`
rendait le solveur intérieur mal conditionné et éloignait la solution du truth.

Résultat M5 déterministe archivé dans
`validation/_generated/performance/experimental_oracle_synthetic_p43_noise/` :

```text
equilibrium RMS                    3.80e-5
displacement error recovered/noisy 0.533
Delta p relative error prior       0.527
Delta p relative error recovered   0.370
Delta p Spearman prior             0.652
Delta p Spearman recovered         0.886
whitened discrepancy / target      1.017
projected gradient infinity norm   3.41e-5
```

Ce résultat qualifie la formulation et ses gradients sur petit domaine. Il ne
qualifie encore ni les hyperparamètres, ni l'échelle absolue des contraintes,
ni P43 réel. L'étape suivante est un sweep de discrepancy/prior sur plusieurs
réalisations P43, puis le passage M20 avant toute application M100. Sur ce
premier cas, le sweep `1, 1e-2, 1e-4, 7e-5, 3e-5` confirme qu'un poids DIC trop
fort recopie le bruit et qu'un poids trop faible dépasse le niveau statistique
admissible. `7e-5` atteint ici la cible `J_DIC/non-weighted ~= 0.5` ; ce nombre
est propre à ce synthétique et ne doit pas devenir un défaut universel.

Trois fenêtres P43 distinctes `(100,120)`, `(20,30)` et `(180,200)` passent
ensuite les mêmes critères avec le même réglage, sans retuning :

```text
equilibrium RMS                    1.32e-5 .. 3.80e-5
displacement error recovered/noisy 0.468    .. 0.844
Delta p relative error recovered   0.370    .. 0.392
Delta p Spearman recovered         0.886    .. 0.891
discrepancy / target               1.017    .. 1.462
projected gradient infinity norm   3.41e-5 .. 2.21e-4
```

Le transfert d'échelle synthétique M10 puis M20 a ensuite été exécuté en
conservant la géométrie relative de la bande et exactement les mêmes poids :

```text
case  u_error/noise  Delta_p rel-L2 prior->oracle  Spearman prior->oracle
M5       0.533                 0.527 -> 0.370             0.652 -> 0.886
M10      0.536                 0.598 -> 0.277             0.671 -> 0.934
M20      0.695                 0.598 -> 0.201             0.673 -> 0.946
```

Les trois cas satisfont simultanément équilibre, discrepancy, réduction du
bruit, amélioration L2 de `Delta p`, amélioration de sa localisation et gradient
projeté inférieur à `1e-3`. Le prochain verrou n'est donc plus la formulation
locale, le gradient ou le passage M20. C'est le branchement de l'histoire DIC
P43 réelle incrément par incrément, puis le coût à M50/M100. Le solveur intérieur
atteint encore fréquemment sa limite d'itérations ; son optimisation vient après
la première histoire P43 convergée.

Ce branchement réel est maintenant effectué sur l'histoire P43 réparée et non
plus sur une interpolation du seul champ final. Le pilote lit directement :

```text
validation/reference_data/dic_multistep_history_p0043_repaired_v1/
    repaired_history_mm.npy
```

Le premier Lagrangien augmenté simultané reste disponible et qualifié sur les
synthétiques, mais il est trop mal conditionné sur P43 réel : il peut atteindre
l'équilibre sans atteindre la stationnarité après des milliers d'itérations
L-BFGS-B. Une seconde voie cohérente a donc été ajoutée pour la qualification :
elle élimine `u` en résolvant exactement l'équilibre à chaque essai de `Delta p`
et calcule le gradient réduit par un adjoint matrix-free. Elle résout le même
problème contraint ; elle est robuste mais n'est pas encore la cible de
performance.

Résultat archivé, crop P43 M20 `(1610:1630, 1075:1095)`, histoire DIC réparée
complète de 40 pas mesurés consécutifs :

```text
status                              completed
accepted DIC states                 1 .. 40
elapsed                             56.99 s
equilibrium RMS                     1.20e-13 .. 1.48e-9
projected reduced gradient          2.95e-11 .. 9.30e-8
maximum Delta p / increment         0 .. 1.03e-3
whitened discrepancy / target       1.16e-4 .. 3.54e-2
cutbacks                            0
constitutive trial rejections       5, all recovered at state 40
```

Le premier calcul à `356.62 s` et discrepancy gigantesque était invalide : la
PSD avait été estimée après retrait de la moyenne spatiale, puis utilisée pour
pondérer un écart de déplacement conservant sa composante constante. Le mode DC
avait ainsi une variance presque nulle et dominait artificiellement plus de
`99.9 %` du misfit. Pour l'assimilation P43 réelle, le whitener conserve
désormais explicitement la moyenne du champ de répétabilité
(`remove_spatial_mean=False`) : le biais global mesuré fait partie de
l'incertitude expérimentale. Les tests synthétiques, dont le bruit est
explicitement recentré, gardent l'option inverse.

Avec ce contrat statistique cohérent, la projection J2 pilotée est équilibrée,
stationnaire et reste très largement dans l'incertitude DIC mesurée. Le conflit
M20 précédemment annoncé n'existe donc pas. À l'état final, la correction de
déplacement maximale vaut `1.15e-5 mm`, soit `12.2 %` de l'incertitude P43
documentée (`9.4e-5 mm`). La déformation équivalente totale DIC/oracle conserve
une corrélation de Spearman `0.903` ; le PEEQ cumulé oracle/prior Ludwik reste
presque identique (`0.20 %` d'écart L2 relatif, Spearman `0.99992`). Ce résultat
qualifie l'histoire M20 complète comme premier oracle mécaniquement admissible,
mais pas encore une loi constitutive identifiée.

Artefacts :

```text
validation/_generated/performance/experimental_oracle_p43_m20/report.json
validation/_generated/performance/experimental_oracle_p43_m20/fields.npz
validation/_generated/performance/experimental_oracle_p43_m20/fields.png
```

Première sensibilité au poids du prior Ludwik, sur les mêmes 40 états :

```text
prior weight  elapsed  max discrepancy  PEEQ rel-L2 vs prior  Spearman PEEQ
3e-2           56.99 s       0.0354              0.0020           0.99992
3e-3           69.18 s       0.0379              0.0154           0.99692
3e-4          149.24 s       0.0525              0.0551           0.96571
```

Affaiblir le prior autorise bien le champ piloté à s'en écarter, mais n'améliore
pas l'accord DIC sur ce crop : à `3e-4`, la régularisation spatiale devient
dominante, le coût double et les rejets constitutifs passent de `5` à `33`.
`3e-3` est le candidat de continuation intermédiaire ; `3e-2` reste la référence
stable tant que la sensibilité n'a pas été répétée sur plusieurs crops et
échelles. Artefacts complémentaires :

```text
validation/_generated/performance/experimental_oracle_p43_m20_prior_003/
validation/_generated/performance/experimental_oracle_p43_m20_prior_0003/
```

Le test limite `prior_weight=0` apporte la distinction attendue : le champ Δp
devient effectivement non identifiable/stable avec les seules contraintes
actuelles. Sur les 40 états, il converge encore mais coûte `315.74 s`, avec
`103` rejets constitutifs ; au dernier état, la corrélation PEEQ oracle/Ludwik
tombe à `0.512` et l'écart L2 relatif à `29.97 %`. La corrélation de la
déformation équivalente oracle/DIC tombe à `0.474`, contre `0.903` avec le
prior `3e-2`. Le déplacement reste pourtant très proche de la DIC (`14.5 %`
de l'incertitude au maximum) et l'équilibre reste résolu (`9.55e-9` RMS).

Le baseline Ludwik, évalué directement sur la DIC, n'est pas équilibré : son
résidu mécanique final vaut `4.79e-2` RMS et `2.45e-1` en norme infinie. Le
prior ne doit donc pas être interprété comme une solution mécanique ; il sert
à sélectionner une branche Δp parmi des champs dont l'effet sur u est faible.

Le diagnostic matrix-free de sensibilité `Δp → u`, obtenu par
`K δu = -Bᵀ σ_,Δp δp` sur 12 directions aléatoires, donne une norme DIC blanchie
de `184 .. 326` par unité de norme L2 de Δp. Pour une perturbation de taille
typique `||Δp||₂≈8.0e-3`, cela représente seulement `1.47 .. 2.61` unités
blanchies : la DIC contraint donc faiblement les détails de Δp. L'artefact
reproductible est :

```text
validation/_generated/performance/experimental_oracle_p43_m20/identifiability.json
scripts/diagnose_experimental_oracle_identifiability.py
```

La suite prioritaire est : qualifier le compromis discrepancy/prior sur cette
histoire complète, ajouter le contrôleur de reprise/cutback avant les tailles où
il devient nécessaire, puis passer à M50/M100 et au recalage des lois MFront.

### Critères de qualification du P0

Le P0 n'est pas qualifié par la seule baisse d'une loss. Il faut démontrer :

- exactitude transactionnelle du Driven J2 ;
- plateau FD des deux tangentes locales ;
- adjoint discret `B/B^T` inchangé ;
- adjoint du whitener ;
- récupération synthétique de `u`, `Delta p` et `sigma` ;
- réduction du bruit conforme au discrepancy principle ;
- conservation des bandes de localisation synthétiques ;
- décroissance séparée du résidu d'équilibre et du misfit DIC ;
- indépendance raisonnable à l'initialisation et à la continuation du prior ;
- provenance complète des PSD, masques, hyperparamètres et transactions.

Sur P43 sans force synchronisée, le livrable doit être nommé :

```text
mechanically admissible DIC-compatible oracle
```

et jamais `identified constitutive law`.

### Ce qui est explicitement hors P0

- PINN, Transformer, réseau neuronal ou opérateur neuronal ;
- optimisation simultanée de tous les temps avant validation séquentielle ;
- covariance DIC dense ;
- remplacement des opérateurs spectraux `B` et `B^T` ;
- identification absolue sans force synchronisée ;
- champ plastique libre non régularisé pixel par pixel ;
- modification opportuniste des lois MFront pour améliorer directement les
  figures avant que l'oracle soit qualifié.

Le P0 doit d'abord produire des données mécaniques propres. Les méthodes
d'apprentissage ou de sparse regression pourront être évaluées ensuite sur
`(epsilon*, sigma*, Delta p*)`, et non sur les champs directs actuellement en
désaccord avec la DIC.

## État courant prioritaire — 2026-08-08, après reprise des agents

La branche active de référence est désormais :

```text
main
HEAD = origin/main = 1377d47
worktree = fichiers suivis propres ; journaux temporaires non suivis conservés
```

### Correction de la régression BLAS dans Krylov — 2026-08-09

La régression de performance observée sur les solveurs spectraux venait de la
sur-souscription BLAS dans SciPy LGMRES/LGMRES/GCROT, pas de MFront ni de la
condensation. À Newton, les produits scalaires et opérations vectorielles de
Krylov utilisaient plusieurs threads BLAS alors que les appels étaient déjà
répétés et que MFront disposait de ses propres threads.

Le commit `cb96390` ajoute la dépendance `threadpoolctl` et borne par défaut
BLAS à un thread uniquement pendant l'appel au solveur Krylov. Le paramètre
`EBISpectralSolverConfig.krylov_blas_threads` est réglable : `1` par défaut,
`None` pour désactiver la limitation. Les threads MFront et FFTW ne sont pas
modifiés. La provenance archive désormais le backend BLAS chargé, ses threads
natifs, le réglage demandé et l'application du plafond runtime.

Confirmation reproductible sur P43 M100 EBSD, 8 incréments, 4 threads MFront,
sans variables BLAS exportées dans le shell :

| configuration | temps | Newton | overhead Krylov |
|---|---:|---:|---:|
| avant correction runtime | `119,48 s` | `57` | `56,13 s` |
| BLAS limité dans le shell | `61,24 s` | `57` | `1,31 s` |
| limitation intégrée au solveur | `62,38 s` | `57` | `1,57 s` |

Le test GPS+FD a ensuite été relancé avec cette limitation intégrée, sur le
même crop : `58,38 s`, `58` Newton, incréments `[6,6,7,7,7,8,8,9]`, résidu
`5,34e-9`. Le FD a concerné `192` points et `1152` trajectoires, pour
`2,15 s` de coût propre. Il devient donc légèrement plus rapide que la
condensation Python sur ce cas, tout en conservant les écarts de champs déjà
qualifiés (`<3e-8` pour les observables de glissement).

Artefacts versionnés :

```text
validation/_generated/performance/
  srix_p43_m100_condensed_blas1.json
  srix_p43_m100_condensed_blas1.fields.npz
  srix_p43_m100_condensed_runtime_blas1.json
  srix_p43_m100_condensed_runtime_blas1.fields.npz
  gps_fd_m100_runtime_blas1.json
```

Ces résultats ne modifient aucune loi constitutive. Les journaux et fichiers
`.progress.jsonl` temporaires restent volontairement hors commit.

La branche `codex/native-generalised-plane-stress` et son ancien HEAD
`6cf51b8` sont historiques. Ne pas reprendre leur état comme état courant.

### Résultat principal des travaux GPS

Le backend produit actuellement est un backend **UMAT de contrainte plane
généralisée**, et non un comportement MFront monolithique. Il est intégré dans
le dépôt applicatif et documenté dans :

```text
mfront/Fcc316LForestRubinSrixGps.mfront
src/fem_inhouse/core/mfront.py
src/fem_inhouse/core/mfront_behaviours.py
src/fem_inhouse/core/plane_stress_material.py
docs/explanation/spectral_mechanics/umat_gps_handoff_2026-08-07.md
```

La voie UMAT a été retenue pour éviter de modifier le générateur TFEL. Elle
est fonctionnellement qualifiée sur les cas M20/M100 avec fermeture plane,
tangente par différences finies et champs proches de la référence. Elle reste
toutefois pénalisée à M100 : environ `85` Newton contre `57` pour la référence,
alors que la comparaison locale et les champs sont corrects. Le travail de
diagnostic a établi que cette pénalité suit la tangente GPS, pas le résidu :
la différence locale de formulation est d'environ `3e-3` au point matériel
responsable. Le backend UMAT ne doit donc pas être déclaré équivalent à la
condensation de référence ni sélectionné sans tenir compte de ce surcoût.

Le fork TFEL `jeffwitz/tfel-generalised-plane-stress` reste un prototype
extérieur, non utilisé par la production. Le TFEL de production est
`5.1.0` non modifié. Le fork explore uniquement le support générateur d'une
hypothèse à trois inconnues transverses ; il n'a pas encore de comportement
SRIX monolithique 3D qualifié.

### Sweep de condensation par blocs M100

Le sweep est désormais versionné dans :

```text
validation/_generated/performance/
  srix_p43_m100_ebsd_condensation_blocks.json
  srix_p43_m100_ebsd_condensation_blocks.csv
  srix_p43_m100_ebsd_condensation_blocks/
scripts/benchmark_srix_ebsd_condensation_blocks.py
```

Configuration : P43 M100 EBSD, 8 incréments, SRIX, 4 threads MFront,
Eisenstat--Walker, LGMRES recyclé, prédicteur transverse tangent.

Résultats mesurés :

| variante | temps | Newton | erreur de champs vs monolithique |
|---|---:|---:|---:|
| monolithique | `54,45 s` | `57` | référence |
| blocs 10000 | `79,24 s` | `57` | `7,32e-11` |
| blocs 5000 | `84,32 s` | `57` | `7,32e-11` |
| blocs 2500 | `82,49 s` | `57` | `7,32e-11` |
| blocs 1250 | `83,57 s` | `57` | `7,32e-11` |
| blocs 625 | `94,01 s` | `57` | `7,32e-11` |

Conclusion : la gestion par blocs est fidèle mais plus lente que le batch
monolithique sur M100. Aucun bloc ne doit être choisi comme optimisation de
production. La télémétrie montre notamment l'augmentation des appels MGIS
avec la diminution de la taille des blocs.

### Diagnostic final de la pénalité GPS 85/57

Les résultats récents sont à lire dans l'ordre des commits `8f7b415` à
`b009364` et dans les rapports associés. Les points établis sont :

- le trial GPS est pur et transactionnel ;
- `accept_global_trial` est neutre pour le GPS ;
- le chemin sous-pas a été acquitté avec un wrapper corrigé ;
- le forcing Krylov serré réduit seulement `52` à `50` Newton sur M20 ;
- le spectre de `BᵀCB` est pratiquement identique entre GPS et référence ;
- un seul point matériel EBSD porte l'essentiel de la différence d'action ;
- la sensibilité directe reproduit exactement le jacobien DSL GPS ;
- le Schur de la référence diffère de la projection GPS d'environ `3e-3` au
  même état complet.

La conclusion actuelle est donc une différence de **formulation de tangent**
entre GPS et condensation Schur, et non un défaut de cache, de rollback, de
Krylov, de rotation ou de convergence locale. L'option `CondensedTangent`
ajoutée expérimentalement dans `Fcc316LForestRubinSrixGps.mfront` est
conservée mais son résultat est négatif (`46 %` d'erreur FD au point 96 et
échec du run M20) ; elle reste désactivée par défaut.

### Test Python du Schur brut — 2026-08-08

Le test proposé après `aa3c33a` a été ajouté à
`scripts/diagnose_gps_direct_sensitivity.py` sans modifier le `.mfront` : les
trois lignes de fermeture GPS sont remplacées en Python par les six lignes
cinématiques brutes, six seconds membres sont résolus, puis le tangent 3D et
son Schur sont reconstruits.

Artefact :

```text
validation/_generated/performance/gps_direct_sensitivity_raw_star.json
```

Résultat au checkpoint de l'incrément 6 :

| point | `C_sens` vs Schur référence | `C_raw*` vs Schur référence | `C_raw*` vs `C_sens` |
|---:|---:|---:|---:|
| 96 | `3,125e-3` | `3,125e-3` | `9,756e-16` |
| 95 | `3,745e-3` | `3,745e-3` | `7,304e-16` |
| 59 | `1,420e-4` | `1,420e-4` | `3,695e-16` |

Le premier câblage du test utilisait seulement trois seconds membres et
calculait une réponse de type déformation plane ; il a été corrigé avant la
mesure archivée. Le test corrigé montre donc que le remplacement algébrique
des lignes GPS ne rapproche pas le résultat du Schur de référence : il
reproduit `C_sens` à la précision machine. La piste « les trois lignes
cinématiques suffisent à récupérer le Schur » n'est pas confirmée par cet
artefact.

Les valeurs de branche Macaulay calculées pour GPS et pour le backend de
référence aux points 96, 95 et 59 ont les mêmes signes/masques actifs dans ce
checkpoint. Elles ne démontrent donc pas une inversion de branche à cet
endroit. Elles restent enregistrées par système dans l'artefact ; il faudra
étendre l'inspection à un état strictement transplanté si la piste de
non-différentiabilité doit être poursuivie.

### Correction same-state et oracle Schur live — 2026-08-08

La conclusion ci-dessus est désormais historique et doit être remplacée par
le diagnostic corrigé. L'ancien transplant mutait `manager.s0`, puis les
helpers restauraient le snapshot original ; les écarts de l'ordre de `3e-3`
étaient donc contaminés. Le script
`scripts/diagnose_gps_tangent_blocks.py` construit maintenant un snapshot cible
immutable, convertit explicitement les gradients global/cristal, copie les
ISV par nom et conserve le `committed_global_strain`.

Le script `scripts/diagnose_gps_direct_sensitivity.py` ne lit plus
`gps_tangent_blocks.json` comme oracle : il reconstruit à chaque exécution le
raw 3D de la référence sur le snapshot GPS transplanté, puis son Schur.
Artefacts courants :

```text
validation/_generated/performance/gps_tangent_blocks_same_state_v2.json
validation/_generated/performance/gps_direct_sensitivity_same_state_v2.json
validation/gps_same_state_transplant_diagnostic.md
```

Résultats live au checkpoint, pour les points 96/95/59 :

```text
|C_sens-C_raw,Schur|/|C_raw,Schur| = 1.36e-13, 1.82e-11, 3.54e-14
|C_raw*-C_sens|/|C_sens|         = 9.76e-16, 7.30e-16, 3.70e-16
```

La formulation GPS et le Schur brut sont donc algébriquement cohérents au
même état physique committé, au même incrément et avec la même orientation.
Cela invalide l'interprétation des anciens écarts comme différence
intrinsèque de formulation. Aucune loi `.mfront` n'a été modifiée et aucune
campagne M20/M100 n'a été lancée pour cette étape.

### Diagnostic shadow/sous-pas — 2026-08-08

Le shadow runtime n'est pas un oracle same-state : `_shadow_condensed_tangent`
repart de l'état committé GPS et réintègre le comportement raw en un pas avec
la transverse finale GPS. Il peut donc suivre une trajectoire différente de la
composition des sous-pas GPS.

Une instrumentation ajoutée à `MFrontNativeGeneralisedPlaneStressBatch` archive
le masque de sous-pas, les divisions, les différences d'ISV et l'écart de
tangent par point. Les campagnes M20 donnent :

```text
GPS sans shadow                         52 Newton
shadow sur points non sous-pasés       52 Newton
shadow sur points sous-pasés            47 Newton
shadow sur tous les points              47 Newton
```

La différence causale est donc portée par la classe de points sous-pasés,
même si certains points non marqués sous-pasés suivent aussi une trajectoire
raw full-step différente. L'oracle FD de l'application complète donne au point
96, pour `h=1e-7`, `FD/GPS=1.35e-1` et `FD/shadow=6.35e-2`; aux points 95 et 59,
les deux tangentes coïncident avec FD à environ `1e-9`. Le shadow est ainsi un
quasi-Newton potentiellement utile, pas une correction d'une erreur algébrique
GPS. Voir `validation/gps_substep_tangent_diagnostic.md`.

### Tangente FD composite sélective — 2026-08-08

Une option expérimentale `gps_composite_fd_tangent=True` est maintenant
implémentée dans `MFrontNativeGeneralisedPlaneStressBatch`. Elle ne calcule la
différence finie centrale que pour les points effectivement sous-pasés ; les
autres conservent le tangent GPS MFront. Sur M20 :

```text
GPS                  52 Newton
FD composite ciblée  45 Newton
référence raw        46 Newton
19 points FD, 114 trajectoires, 0 changement de partition
écart déplacement 4,24e-13, contrainte 2,65e-9, glissements 4,00e-9
```

L'option reste désactivée par défaut et n'est pas qualifiée sur M100. Elle
utilise des évaluateurs GPS mono-point cachés ; il faut mesurer le coût réel,
la stabilité de la partition et la reproductibilité avant toute activation de
production. Artefact :
`validation/_generated/performance/gps_composite_fd_vs_gps_m20.json`.

## État de reprise — 2026-08-07

Cette section est prioritaire pour toute nouvelle IA qui reprend le dépôt.
Elle décrit les travaux récents qui n'étaient pas encore reportés dans ce
fichier.

### État historique de la branche expérimentale

La branche décrite ci-dessous n'est plus la branche active ; elle est conservée
comme historique de conception :

```text
codex/native-generalised-plane-stress
HEAD historique: 6cf51b8 docs(mfront): document monolithic plane-stress blocker
```

À l'époque, le dépôt applicatif contenait des changements non commités
appartenant au travail de condensation par blocs. Ils ont depuis été intégrés
et archivés sur `main`; la liste ci-dessous est conservée uniquement comme
trace historique :

```text
docs/explanation/spectral_mechanics/srix_p43_performance_and_step_control.md
tests/unit/core/test_mfront.py
scripts/benchmark_srix_ebsd_condensation_blocks.py
validation/_generated/performance/srix_p43_m100_ebsd_condensation_blocks.csv
validation/_generated/performance/srix_p43_m100_ebsd_condensation_blocks.json
validation/_generated/performance/srix_p43_m100_ebsd_condensation_blocks/
```

Sur l'état courant `main`, ces fichiers sont versionnés. Ne pas utiliser cette
ancienne liste pour déduire l'état du worktree.

### Résultats applicatifs déjà réalisés

- La divergence non linéaire TRI2 a été vectorisée et les essais constitutifs
  acceptés sont réutilisés au Newton suivant. Le commit correspondant est
  `22143ec`.
- Le backend SRIX avec condensation externe en contraintes planes reste le
  backend de référence et doit rester disponible.
- Le cas SRIX P43 M100 EBSD avec 8 incréments, Eisenstat--Walker, LGMRES
  recyclé, prédicteur transverse tangent et quatre threads MFront avait une
  référence à `56,88 s`. Le dernier artefact de la campagne par blocs mesure
  `54,45 s` pour le batch monolithique dans son environnement précis. Ces
  chiffres ne doivent être comparés qu'avec les manifestes et l'environnement
  exacts.
- La carte EBSD est bien prise en compte dans les campagnes EBSD ; elle ne doit
  pas être remplacée par l'orientation homogène `[35,20,15]` sans le signaler.
- Les blocs de condensation MGIS ont été qualifiés en fidélité sur M100, mais
  aucune taille de bloc n'est une optimisation : `79,24–94,01 s` contre
  `54,45 s` pour le batch monolithique, avec `57` Newton dans tous les cas.
  Le surcoût vient de la multiplication des appels MGIS.
- Le contrôleur adaptatif par doublement de pas a été étudié sur M20. Le
  contrôleur ne doit pas être déclaré qualifié : près de la première activation
  plastique, le critère relatif sur les glissements était dominé par une
  amplitude de l'ordre de `1e-6`. Les seuils d'erreur doivent rester
  réglables et distinguer contrôle par nombre de Newton et contrôle par erreur
  constitutive.

### Comparaison SRIX/Méric à retenir

La comparaison P43 16 incréments des cartes de glissement est documentée. Les
trois systèmes dominants sont les mêmes et dans le même ordre (`01`, `07`,
`11`), avec une hiérarchie dominante commune mais une redistribution des
amplitudes entre systèmes et dans l'espace. La distance de variation totale
est d'environ `0,2565`, le recouvrement `S95` est `0,80`, et la corrélation de
rang corrigée est `0,9385`.

Le champ global comparé est la **somme des glissements cumulés des douze
systèmes**, notée `total_accumulated_system_slip`; il ne s'agit pas d'une PEEQ.
La comparaison des signes est par système et sépare désormais : activité dans
les deux lois, même signe, signe opposé, activité Méric seule et activité SRIX
seule. Un zéro dans une loi ne doit jamais être appelé « signe opposé ».

Limites obligatoires : `R` SRIX est une transposition analytique de paramètres
Méric à une vitesse de référence, non une identification directe du matériau
P43 ; Méric dépend du pseudo-temps ; l'orientation actuelle est homogène dans
ces cartes de comparaison ; les champs Méric à 16 incréments sont convergés
numériquement mais ne constituent pas une étude de convergence temporelle.

### Prototype TFEL externe — ne pas le confondre avec la production

Le support monolithique de la contrainte plane généralisée est développé dans
un fork TFEL séparé, pas dans le dépôt applicatif :

```text
repository : jeffwitz/tfel-generalised-plane-stress
local path : /tmp/tfel-generalised-plane-stress
branch     : agent/generalised-plane-stress
HEAD       : ffcdcb3 docs(mfront): describe generalized plane stress prototype
```

Commits importants du fork :

```text
108891f  accepte GENERALISEDPLANESTRESS comme hypothèse de propriété MFront
4b6b741  génère un système local prototype à trois inconnues transverses
ffcdcb3  documente le statut et la limite du prototype
```

Vérification effectuée : avec le TFEL construit dans `/tmp/tfel-gps-build` et
installé dans `/tmp/tfel-gps-install`, un comportement implicite minimal
compile avec un système local de dimension 6 : trois composantes de
déformation imposées et trois inconnues scalaires (`ezz`, `eyz`, `exz`).

Cette avancée **ne constitue pas encore le comportement SRIX monolithique**.
Le prototype MFront expose encore une représentation réduite de type tenseur
symétrique ; il ne fournit pas à SRIX un tenseur de contrainte et une tangente
3D internes permettant d'assembler correctement les trois équations
`sigma_zz = sigma_yz = sigma_xz = 0`. Il ne faut donc pas sélectionner ce
backend dans l'application, ni annoncer un gain M100.

La prochaine étape technique est de concevoir cette séparation entre :

1. gradient externe généralisé à trois composantes dans le plan ;
2. état interne 3D complet pour l'élasticité, les rotations, les glissements
   et les contraintes ;
3. trois résidus transverses dans le même Newton local ;
4. tangent condensé cohérent retourné au solveur global.

Une simple enveloppe appelant SRIX 3D plusieurs fois depuis Python ou C++ ne
respecterait pas l'objectif monolithique et ne doit pas être introduite sous
ce nom.

### Commandes et précautions de reprise

Pour vérifier le prototype TFEL sans toucher au dépôt applicatif :

```bash
cd /tmp/tfel-generalised-plane-stress
git log -3 --oneline
cmake --build /tmp/tfel-gps-build -j4 --target TFELMFront mfront
cmake --install /tmp/tfel-gps-build --prefix /tmp/tfel-gps-install
```

Le probe minimal utilisé pendant la vérification était temporaire dans
`/tmp/GeneralisedPlaneStressProbe.mfront`; il n'est pas un comportement SRIX
et n'est pas un artefact scientifique du dépôt.

Avant toute modification applicative, lire notamment :

```text
docs/explanation/spectral_mechanics/srix_monolithic_plane_stress_architecture.md
mfront/Fcc316LForestRubinSrix.mfront
src/fem_inhouse/core/plane_stress_material.py
src/fem_inhouse/core/mfront.py
```

Ne pas relancer M100/M200 pendant une phase de conception du générateur sans
avoir d'abord validé un point matériel et M20. Le backend condensé externe
reste la référence numérique.

### 2026-08-08 — UMAT GPS : de bloqué à plus rapide que la référence, et le §5.1 est retiré

Vingt-sept commits fusionnés dans `main` (`76e6959`). La substance tient en
trois défauts et leurs conséquences.

**Le pont appliquait la déformation TOTALE comme un incrément.** `evaluate`
reçoit le total — c'est le contrat de tous les backends de contrainte plane du
module — et le pont écrivait `s0.gradients + gradient`. La charge imposée
s'accumulait donc en `1+2+3+…` au lieu de `1,2,3` : trace en plan de `3,0e-3`
là où `2,0e-3` était demandé à l'incrément 2. L'incrément 1 n'était pas touché
(total et incrément y coïncident), **et c'est exactement pourquoi toutes les
comparaisons contre la référence s'accordaient à l'incrément 1 et divergeaient
au 2** — signature lue trois jours durant comme un changement de branche.

**Il n'y a jamais eu deux branches.** Le F1 de la préinscription, la
« multiplicité de racines » du diagnostic, le mur de robustesse du §5.2 (la
déformation croissait quadratiquement) et la tangente fausse découlent tous de
cette ligne. La décision « on force la racine de contrainte plane » n'a plus
d'objet, et **les campagnes condensées archivées n'ont pas à être refaites** :
la référence avait raison depuis le début. Accord après correction : `1,1e-11`.

**Les rotations par point étaient lues à travers les points.**
`rotations[:, row, col]` est une vue à pas de neuf doubles, passée à MGIS en
`ExternalStorage` — qui la lit comme contiguë. Chaque point lisait les `Q` d'un
autre. Les tampons étaient de surcroît des temporaires libérés aussitôt. **Un
seul point matériel masque les deux**, d'où le silence de toutes les
qualifications mono-orientation pendant que les 400 points EBSD démarraient
leur premier résidu à `0,835` contre `0,178`.

**Le pont intégrait sur un seul fil** quand la référence en utilisait quatre —
l'essentiel du « facteur 2,5 inexpliqué » du coût par intégration. Intuition de
l'utilisateur, vérifiée en un grep.

**Trois optimisations ensuite.** La contrainte plane portée par trois lignes de
résidu au lieu de trois inconnues (21 → 18, plus de point-selle, jacobien
élastique constant) ; le sous-pas appliqué **aux points fautifs** et non au lot
entier, ce qui gagne **deux ordres de grandeur** sur l'accord avec la référence
(contrainte `3,2e-3 → 3,8e-5`) ; et un cache des indices fautifs **prouvé
complet par ré-intégration groupée** — avancer le `s0` d'un point du cache lui
fait voir un incrément nul, donc un seul appel groupé prouve qu'aucun point hors
cache n'a échoué, sans rien lire dans l'état.

**Résultat.** Qualification **ACCEPTÉE** sur les trois cas : fermeture
`2 – 4e-14 MPa`, tangente FD `1,2 – 1,6e-07`, écart à la référence `1e-11`.
P43 20x20 tourne à `1,2 – 1,7×` la référence sur le temps matériau ; **100x100
est à parité**, et la pénalité de `1,49×` sur les itérations globales qui l'y
retient est mesurée mais **inexpliquée** — ce n'est pas le sous-pas, qui ne
touche que `0,071 %` du lot (400 points sous-passés sur 1,72 M intégrations,
cache à 91 % de succès).

**Cinq résultats négatifs conservés et reproductibles** : la route 1 n'a jamais
tiré (`|ΔK| = 0` bit à bit) ; la route 2 est réfutée (imposer la déformation ne
sélectionne pas la branche) ; le GIL est retenu (`0,73×` en threads Python
contre `2,82×` au pool MGIS) ; le détecteur par résidu de fermeture est aveugle
(MFront laisse `s1` à l'état committé sur échec) ; et deux de mes propres
lectures sont rétractées — le §8.7 (violation cinématique, en fait un artefact
de ma reconstruction) et le §8.17 (ensemble fautif plus grand, en fait quatre
fois plus petit).

**Tout tourne sur TFEL/MFront 5.1.0 NON MODIFIÉ.** Vérifié par reconstruction
depuis zéro, par les en-têtes installés identiques octet pour octet à l'état
d'avant le fork, et par le `mfront` installé qui rejette l'hypothèse
`GeneralisedPlaneStress` du fork. La voie UMAT a précisément été choisie pour
éviter de patcher le générateur ; le fork `jeffwitz/tfel-generalised-plane-stress`
reste hors de la chaîne de production.

**Piège rencontré deux fois, à retenir** : un incrément de déformation nul
n'interroge pas la loi qu'on croit interroger — la branche élastique gardée
répond. Il a rendu la vérification A6 vacue (succès rapporté sur une tangente
fausse d'un facteur cinq) puis cassé la convergence de P43 à l'incrément 5
quand la ré-intégration de preuve a renvoyé la tangente élastique aux points les
plus plastiques.

**Reste ouvert** : la pénalité d'itérations globales à 100x100, et le `2,1×` de
coût par appel — ni la taille du système local, ni le nombre d'itérations
locales, ni le recalcul de constantes ne l'expliquent. Détail complet dans
`docs/explanation/spectral_mechanics/umat_gps_handoff_2026-08-07.md`, §8.8 à
§8.18.

### 2026-08-08 — Pénalité 85 vs 57 à M100 : les hypothèses locales sont toutes éliminées

Cinq commits (`f18e89a` → `f83a19b`) contre la pénalité d'itérations globales
qui retient le backend GPS UMAT à parité de temps mais pas de Newton. **Chaque
hypothèse locale a été réfutée ou acquittée ; le mécanisme restant est un écart
de taux de convergence aux états profonds.**

- **Tangente réfutée** (`f18e89a`) : sur 40 états réels de la campagne M100
  archivée (déformation par élément du champ de déplacement, orientations EBSD
  du crop), les tangentes UMAT et référence sont identiques à `1e-16`. L'écart
  FD de 6–30 % mesuré précédemment est un artefact du sous-pas : la réponse
  sous-passée n'est pas lisse à l'échelle de la sonde (rapport 0,05 à 0,21
  entre perturbations `1e-6` et `1e-4`).
- **Chemin de sous-pas acquitté** (`ba6c861`) : nouveau backend
  `mfront-3d-condensed-plane-stress-halved` qui force la référence condensée
  dans le même chemin deux-moitiés que le sous-pas GPS (loi et tangente
  intactes). 20×20 : référence 46 | halved 46 | GPS 52. M100 : référence 57 |
  halved 57 | GPS 85.
- **Divergence dès la première évaluation** (`ae1064e`) : sur 20×20, évaluation
  1 de l'incrément 1 — avant tout sous-pas — les contraintes diffèrent déjà de
  `4,4e-7 MPa` (`2,7e-9` relatif), états internes à `1e-12`. Cas A du cadre de
  divergence : différence d'évaluation à états committés identiques. Suspect :
  l'écart de tolérance de fermeture (référence `1e-8 MPa`, GPS ~`1e-14`).
- **Tolérance de fermeture et epsilon éliminés** (`1df14a6`) : référence
  resserrée à `1e-10` et `1e-12` → 57 Newton inchangé ; loi GPS remise à
  `@Epsilon 1.e-14` (comme le SRIX brut) → 85 inchangé. Les historiques de
  résidu montrent le mécanisme : aux incréments profonds (5–8), les quatre
  premières itérations sont identiques, puis le GPS converge **linéairement**
  (résidu ×0,2 par itération, 11 itérations pour `3e-9`) contre **quadratique**
  pour la référence (×0,05, 7 itérations pour `7,6e-11`). La pénalité est un
  écart de taux de convergence aux états profonds, pas une différence de
  formulation locale.
- **Le wrapper halved était cassé, le test corrigé acquitte** (`f83a19b`) : le
  `UniformlyHalvedReference` appliquait `fraction × déformation totale` — avec
  le contrat en absolu, cela renvoyait le matériau **en arrière** (à
  l'incrément 5 de 8 : `5/16` au lieu de `9/16` de `eps_f`) et restaurait
  l'état committé avant le commit du solveur, jetant l'évolution sous-passée.
  Le 57 = 57 de `ba6c861` était donc **vacant**. Corrigé : interpolation
  `eps0 + α(eps1 − eps0)` et ré-instauration de l'état final sous-passé au
  commit. M100 corrigé : référence directe 57 | référence vraiment sous-passée
  57 | GPS 85. Le chemin de sous-pas — et la structure de tangente du dernier
  sous-pas qu'il partage avec la référence halvée — sont acquittés par un test
  valide.

**Suspects restants** : la gestion trial/état du GPS (cas C du cadre de
divergence) et les différences de solution locale aux points sous-passés —
seules hypothèses qui survivent à l'échelon local. Artefacts de campagne dans
`validation/_generated/performance/` (résumés commités, répertoires de champs
commités le 2026-08-08).

### 2026-08-08 — Cinq tests du programme de falsification : le GPS est pur, le Jv est exact, et la pénalité suit le TANGENT

Programme proposé par l'utilisateur après l'acquittement du chemin de
sous-pas. Cinq tests sur 20×20 EBSD, GPS vs référence condensée, tous
implémentés dans `scripts/diagnose_gps_trial_purity.py`,
`scripts/diagnose_jv_global_fd.py`, `scripts/diagnose_crossed_stress_tangent.py`
et un run de forcing. Résultats (52 vs 46 Newton à 20×20) :

- **Test 1 — pureté `evaluate(A)-evaluate(B)-evaluate(A)` sans commit :
  GPS PUR bit à bit** (`0.00e+00` sur stress, tangent, g/p/a[12], tenseurs
  complets, transverses, décision de substepping, subdivisions, cache).
  La variante snapshot/restore est aussi bit à bit. Le GPS est strictement
  une fonction de `(s0, eps_trial)` : aucune fuite de trial/state/cache.
  Contre-témoin surprenant : la RÉFÉRENCE est légèrement impure après
  restore (`5.99e-05` sur le tangent, `5.01e-08` sur le stress) et son
  `accept_global_trial` est transformatif (`4.63e-05`) — l'ordre de sa
  tolérance de fermeture locale (`1e-8 MPa` absolu), pas un artefact de
  structure : le warm-start local converge dans sa boule de tolérance.
- **Test 2 — `accept_global_trial()` sans commit : NEUTRE sur le GPS**
  (`0.00e+00`) : le prédicteur accepté n'a aucun effet sur la réponse.
- **Test 3 — Jv global FD sur un vrai checkpoint d'incrément profond** :
  l'opérateur que le solveur applique (`div(C·Bv)` avec la tangente du
  premier itéré) contre `(R(u+hv)-R(u-hv))/2h` réintégré depuis le même
  snapshot committé, h = 1e-4..1e-7, v = direction réelle de deux itérés
  Newton consécutifs du run. GPS : `9.8e-3 → 1.9e-7 → 2.9e-8 → 1.1e-9` ;
  référence : `9.8e-3 → 8.2e-6 → 5.7e-8 → 5.7e-8`. **Le Jv assemblé du
  GPS est aussi exact que celui de la référence** — le contraste
  `~1e-5` vs `~1e-1` attendu n'existe pas. Le plateau à `1e-2` pour
  `h=1e-4` est la non-linéarité de la direction (`|Δε|=8e-3`), identique
  aux deux backends.
- **Test 4 — forcing Krylov serré** (GPS M20, `--linear-mode fixed`
  `--gmres-rtol 1e-8` contre Eisenstat–Walker) : **52 → 50 Newton**.
  L'interaction GPS/forcing inexact coûte 2 itérations sur 6 de pénalité,
  pas le mécanisme.
- **Test 5 — échange croisé stress/tangent, le décisif** :
  GPS-stress + tangent REF = **47** Newton (≈ référence 46) ;
  REF-stress + tangent GPS = **54** (≈ GPS 52). **Le nombre d'itérations
  suit le TANGENT, pas le stress** : avec le bon tangent, le résidu GPS
  converge comme la référence ; avec le tangent GPS, même le résidu
  référence converge comme le GPS. La fonction résiduelle GPS est
  innocentée ; c'est la matrice GPS qui porte la pénalité.

**Lecture cohérente de l'ensemble** : la dérivée GPS est exacte (test 3,
et tangentes identiques à `1e-16` à états appariés) mais la matrice GPS
est un itérateur moins efficace aux états profonds — convergence linéaire
à `×0.2` par itération contre `×0.05` pour la référence (mesuré dans
`1df14a6`), et c'est ce taux qui décide du compte de Newton.

### 2026-08-08 — Analyse spectrale de `BᵀCB` : le spectre n'explique pas la pénalité

Suite de la piste « matrice moins efficace » : assemblage de la matrice
PLEINE `J = BᵀCB` (722×722 sur 20×20) pour les deux backends, aux
incréments 5–8, avec les tangentes réellement appliquées par le solveur
(enregistrées à chaque appel, sous-pas GPS inclus), spectre brut et
préconditionné `J·P` (le green spectral, identique pour les deux).
`scripts/diagnose_spectral_conditioning.py`.

**Résultat : les spectres sont identiques au 3ᵉ chiffre significatif à
chaque itération de chaque incrément profond.** Conditionnement brut
`6.6e2–7.6e2`, préconditionné `3.8e1–5.4e1`, mêmes bornes pour le GPS et
la référence ; et ce malgré le sous-pas GPS actif à chaque appel après le
premier (`substeps [0,1,1,…]` contre `[0,0,0,…]` pour la référence). La
cinématique `BᵀB` domine le spectre et la tangente ne le différencie pas.
**Le mauvais conditionnement intrinsèque du GPS est réfuté comme
explication du taux `×0.2` vs `×0.05`.**

Bilan de l'investigation complète (85 vs 57) : pureté du trial bit à bit,
`accept_global_trial` neutre, Jv global exact à `1e-9`, forcing serré
quasi neutre (52→50), compte suivant le tangent (47/54 croisés), spectre
de `BᵀCB` identique. **Toute hypothèse structurelle locale est réfutée ;
il ne reste que le cas A du cadre de divergence** (`ae1064e`) : les deux
backends diffèrent dès l'évaluation 1 de l'incrément 1 — stress à
`4.4e-7 MPa` (`2.7e-9` relatif), états internes à `1e-12` — et Newton
amplifie cette différence d'évaluation en un écart de taux de convergence
puis en un compte d'itérations. La cause racine est une différence
d'évaluation constitutive de l'ordre de `1e-9` relatif, pas un défaut de
structure (ni formulation, ni dérivée, ni sous-pas, ni cache, ni
assemblage, ni conditionnement).

### 2026-08-08 — Localisation causale 52 vs 46 : UN point matériel porte la pénalité

CdC de l'utilisateur : arrêter les analyses globales, localiser
spatialement. `scripts/diagnose_gps_tangent_localisation.py`, checkpoint
incrément 6 (premier appel Newton) du run M20 EBSD, tests A–E
(rapport complet : `validation/gps_tangent_localisation_diagnostic.md`).

- **Test A — directions Newton au checkpoint quasi identiques** :
  `|δu_G − δu_R|/|δu_R| = 8,4e-5`, `cos θ = 1 − 3e-9`, et les deux
  corrections réduisent le résidu GPS du même facteur
  (`ρ_GPS = ρ_REF = 0,219`, qui reproduit exactement la réduction réelle du
  solveur `4,86e-2 → 1,07e-2`). La pénalité n'est pas une mauvaise direction
  locale au checkpoint.
- **Test B — action très concentrée** : 6 points (sur 800) font 50 % de
  l'action `(J_G − J_R)δu` ; le point 96 (pixel (8,2), subcell 0) fait 32 %
  seul. CSV : `gps_tangent_localisation_points.csv`.
- **Test C — le test central : top-1 suffit.** Substitution chirurgicale de
  la tangente référence (stress/état/loi/substepping GPS conservés) sur les
  top-k points : 0 → 52, **1 → 47**, 5..800 → 47. Un seul point matériel
  porte toute la pénalité ; le critère du CdC (« 52 → 46–48 ») est atteint
  avec k = 1.
- **Test D** : le point 96 a le système de glissement 11 dominant
  (`−1,1e-2`), 6 systèmes actifs, stress au checkpoint
  GPS `[−152,6, 119,8, −40,6]` vs réf `[−152,6, 119,9, −40,5]` MPa — états
  proches, tangentes non.
- **Test E — à même état, les tangentes diffèrent encore** : transplant des
  ISV par nom (elastic strain, g, p, back strain) sur les top-5 points,
  évaluation au même strain imposé : `1,9e-3` (point 96), `3,6e-3` (95),
  `1,9e-4` (59), `2,0e-4` (20), `8,7e-5` (21), symétrique S_G/S_R. Le seuil
  `1e-10` de formulation identique est manqué de 7 ordres de grandeur.
  Limite documentée : le transplant est partiel (le gradient committé de la
  référence est en repère cristal, celui du GPS en global ; imposer le
  gradient global à la référence fait échouer son Newton local — vérifié).
  La différence à même ISV peut venir des transverses committées ou d'une
  différence algébrique réelle de la tangente.

**Chaîne causale établie avec localisation** : `ΔC_i` (point 96, `1,9e-3` à
même ISV) → `ΔJ` (32 % de l'action sur ce point) → `Δδu` par itération
(`8,4e-5`, petite mais systématique) → amplification le long de la
trajectoire → 52 vs 46. La mesure globale (Jv exact, spectres identiques)
et le compte d'itérations sont réconciliés : la différence est locale au
point 96 et s'amplifie au fil des itérations.

**Prochaine étape (CdC §12)** : comparer bloc par bloc le Jacobien local
MFront GPS et la condensation référence sur le point 96 (et voisins 95, 59),
aux états committés des deux trajectoires au checkpoint de l'incrément 6.
Ne rien modifier avant cette démonstration.

### 2026-08-08 — Sensibilité directe = FD : la tangente OFF est exacte, le shadow est un autre système

La vérification FD du stress (transverses libres, fermeture GPS convergée
dans chaque évaluation) au point 96 donne `41115` — identique à C_sens et à
la tangente OFF. **La tangente DSL projetée est la dérivée exacte du système
GPS** ; le shadow (`41400`, `3e-3`) est la dérivée d'un autre système (loi
brute avec transverses imposées). La pénalité 85-vs-57 n'est pas une
tangente fausse : le gain du shadow (52→47) est une substitution de matrice
d'un système différent, pas une correction de dérivée.

### 2026-08-08 — Option MFront-native du Schur : échec documenté

CdC étape 3 implémentée : `@TangentOperator` custom dans
`Fcc316LForestRubinSrixGps.mfront` (paramètre `CondensedTangent`, défaut 0)
qui reconstruit la matrice de la loi brute, résout `A X = [I6;0]`, tourne
en global et retourne le Schur — sans shadow. Résultat : tangente fausse
(`46 %` d'erreur FD au point 96, `42526` vs C_sens `41115`) et run M20 non
convergent (incrément 1). Le Schur du jacobian GPS régénéré ne reproduit
pas le Schur de la loi brute (le shadow évalue la loi brute avec le strain
complet, ce que le jacobian GPS ne fait pas). **L'option reste dans la loi
(paramètre inactif par défaut, chemin OFF vérifié à 52 Newton M20), mais la
voie du CdC pour la production n'est pas praticable par cette construction.**
Le shadow reste l'oracle de dérivée le plus proche de la référence, au prix
de son intégration 3D.

### 2026-08-08 — Bloc par bloc : la formulation GPS diffère du Schur de `3e-3` au même état

Étape §12 du CdC, `scripts/diagnose_gps_tangent_blocks.py`. Transplant
COMPLET (ISV par nom + gradient tourné dans le repère du receveur +
transverse convergé du GPS imposé) sur les points 96/95/59 au checkpoint
inc 6. Contrôles méthodologiques passés : la tangente 3D de la référence
reconstruite à sa fermeture convergée redonne son Schur à `0.000e+00` ;
la tangente GPS est la tangente DSL globale réévaluée au même état.

**Résultat : à même état complet, Schur(référence) vs tangente projetée(GPS)
= `3,1e-3` (point 96), `3,8e-3` (95), `1,4e-4` (59)** — l'écart de
formulation croît avec le score de responsabilité (test B). La comparaison
6×6 brute n'est pas pertinente : la tangente DSL GPS a les lignes
transverses nulles par construction (Cbb = 0, documenté dans `mfront.py`) ;
c'est la projection in-plane qui décide du Newton, et elle diffère.

**Conclusion causale finale** : la tangente GPS projetée (DSL + projecteur
in-plane P) diffère du Schur de la condensation référence de `~3e-3` à
l'état réel du point 96 — une différence de FORMULATION (dérivée), pas
d'état. Le solveur GPS applique cette matrice le long de la trajectoire ;
la direction diffère de `8,4e-5` par itération (test A) et s'amplifie en
écart de compte (52 vs 46). Localisation, causalité et échelle établies.

**Piste de correction (non modifiée — démonstration d'abord)** : la
post-multiplication par le projecteur in-plane `P` est l'endroit où la
tangente DSL GPS devrait coïncider avec le Schur — elle n'y arrive qu'à
`3e-3`. La route 2 du handoff (`shadow_condensed_tangent`, Schur de la loi
brute via le shadow) produit exactement le Schur ; désactivée par défaut
car « le shadow ne suit pas la branche » — à réexaminer à la lumière de
cette mesure.

### 2026-08-08 — Sensibilité directe 18×18 : le DSL est exact, le SYSTÈME diffère du Schur

Étapes 1-2 du CdC, `scripts/diagnose_gps_direct_sensitivity.py`.
Réimplémentation numpy complète des 18 équations GPS (résidu + Jacobien
déclaré), validée : **`|F(x*)| = 1e-15`** à l'état convergé, et
**`C_sens = C_DSL` à `1e-15`** — la sensibilité directe
`(∂σ_a/∂x)(−A⁻¹B)` avec `B = [−I₃; 0; 0]` reproduit exactement la tangente
retournée par le bridge. **Le DSL ne fait pas d'erreur de calcul.**

Mais **`C_sens ≠ C_shadow` à `3,1e-3`** (point 96) : le jalon du CdC
(`≤ 1e-10`) est réfuté par la mesure. Interprétation : c'est le SYSTÈME,
pas la dérivée — la fermeture GPS (`σ_b = 0` lignes 2,4,5 de `feel`) ne
voit que `deel` (∂σ_b/∂dg = 0), alors que le Schur de la référence élimine
les transverses par la cinématique complète (dépend de `dg`). Mêmes
valeurs (`1e-11`), dérivées différentes (`3e-3`). Le shadow (52 → 47) ne
corrige pas une dérivée mal calculée : il substitue la matrice d'un AUTRE
système. Améliorer la convergence GPS = rapprocher la formulation de la
condensation, pas recalculer la dérivée (déjà exacte). Vérification FD
cohérente : l'écart `A_an vs A_FD ≈ 2,5e-3` constant en h, concentré sur
les lignes `fg` des systèmes inactifs (non-différentiabilité du crochet de
Macaulay).

### 2026-08-08 — Convention du wrapper halvéd corrigée ; le verdict 57 = 57 tient

Le `UniformlyHalvedReference` interpolait `eps0` depuis
`manager.s0.gradients[:, [0, 1, 3]]` — du **Kelvin dans le repère cristal**
(le pont 3D tourne les déformations globales vers le cristal avant
l'intégration, `commit()` = `mgis.update` s1→s0) — avec `eps1 =
in_plane_strain` en **ingénieur dans le repère global**. Le mélange
corrompait le cisaillement (facteur `√2` entre Kelvin et ingénieur) dès
l'incrément 2, et toutes les composantes sous orientation EBSD quelconque.
Le wrapper tenait donc sa propre comptabilité depuis le premier commit, la
correction est triviale : `_committed_in_plane_engineering` (ingénieur,
global, zéro initial, mise à jour au `commit()` avec la dernière demande,
incluse dans snapshot/revert), sans plus jamais lire le manager.

Vérification M100 (mêmes flags que `reference_halved_fixed_m100`, env TFEL
sourcé) : **57 Newton — identique au run à la convention mélangée.** La
pénalité 85 vs 57 n'est donc pas un artefact du wrapper : référence directe
57 | référence vraiment sous-passée 57 | GPS 85, acquittement du chemin de
sous-pas confirmé sur une base saine. La leçon, déjà rencontrée ailleurs :
ne jamais mélanger les conventions de stockage (Kelvin vs ingénieur, global
vs cristal) dans une interpolation — le wrapper de test doit parler la
convention de son interface, pas celle du manager MGIS.

### 2026-08-07 — UMAT GPS : la fermeture marche, la qualification échoue sur les racines multiples de la loi

La voie « fermeture dans l'UMat » (Q en 9 propriétés matériau, loi
`Fcc316LForestRubinSrixGps`, pont passif `MFrontNativeGeneralisedPlaneStressBatch`)
a été implémentée, compilée et qualifiée. **La fermeture est correcte et
vérifiée** (σ_zz = σ_xz = σ_yz = 0 à `1e-14 MPa` dans le repère structural,
accord parfait avec la référence à l'incrément élastique : contrainte
`7e-15`, tangente `4e-11`). **Le F1 de la préinscription se déclenche** :
dès le premier incrément plastique, la solution UMAT diverge de la référence
(18 MPa à 12 incréments, 160 MPa à 96, avec échec du Newton local).

**Diagnostic, établi par sonde C++ directe** : les deux états sont des racines
du même système discret — la loi SRIX admet **plusieurs racines** au premier
incrément plastique (ensembles de systèmes actifs différents sous le crochet
de Macaulay) ; le Newton imbriqué Python et le Newton conjoint 21 inconnues
sélectionnent des branches différentes. C'est le comportement « une autre
solution » déjà consigné au journal 2026-08-03. La stratégie UMAT n'est donc
**pas réfutée** (elle impose bien la fermeture structurale en un seul
Newton) ; c'est la sélection de branche de la loi qui reste ouverte, et le
backend condensé externe reste la référence. Résultat négatif archivé :
`validation/srix_umat_gps_closure_results.md`, préinscription
`validation/srix_umat_gps_closure_preregistration.md`, script
`scripts/qualify_srix_umat_gps_closure.py`. La loi et le pont sont conservés,
expérimentaux, non sélectionnés par défaut.

**Diagnostic exploratoire des branches (option 2), même jour.** Le F1 est
expliqué : le problème 3D admet plusieurs racines au premier incrément
plastique (depuis le même état committé et le même incrément, le Newton 3D
brut converge vers `sigma_zz = -154,7 MPa` et le Newton conjoint UMAT vers
`sigma_zz = 0` — les deux convergés). Les ensembles de systèmes actifs sont
identiques (`[1,2,4,5,7,8,10,11]`) ; les branches diffèrent par les
amplitudes, via la rétroaction `eps_zz ↔ Deq`. La branche UMAT est robuste
aux départs perturbés (10/10 identiques), la référence échoue 9/10 — la
fragilité n'est pas du côté attendu. Rapport :
`validation/srix_plane_stress_branch_diagnostic.md`, script
`scripts/diagnose_srix_plane_stress_branches.py`. Aucune décision n'en est
tirée : la référence condensée reste la référence.

**Décision de l'utilisateur (même jour) : la racine de contrainte plane est
la seule valide — on la FORCE.** La racine « naturelle » à `sigma_zz =
-154,7 MPa` viole la fermeture : ce n'est pas une solution de contrainte
plane. Le Newton conjoint UMAT est l'implémentation de ce principe (la
fermeture est une équation du Newton, chaque itéré se dirige vers la racine
qui satisfait `sigma_transverse = 0`). Conséquences actées : (1) la
préinscription est amendée — A1' = la solution UMAT est une racine du système
fermé (Newton convergé + fermeture `<= 1e-6 MPa` + tangente FD), l'écart à
la référence devient un écart de branche rapporté ; (2) à terme, les
campagnes condensées archivées (sur la branche naturelle) devront être
refaites ; pour l'instant on documente et on implémente.

**Deux bugs de convention de stockage trouvés et corrigés (commit
`6e9a423`)** — les cas identité étaient aveugles aux deux :
1. la formule de rotation de la loi (`gpsRotate`) utilisait le stockage
   ingénieur alors que MGIS stocke en Kelvin (cisaillement `gamma/sqrt(2)`) :
   le mélange diagonale↔cisaillement porte `sqrt(2)`. L'ordonnancement
   `[11,22,33,12,13,23]` est standard (confirmé par `kelvin_3d_to_tensor`) ;
   le facteur `sqrt(2)` était l'erreur (l'intuition de l'utilisateur sur
   l'ordonnancement a déclenché la vérification) ;
2. l'opérateur dans le plan du pont était **transposé** (la sortie MGIS a
   les vecteurs unités rotés en lignes, l'opérateur dérivé les veut en
   colonnes).

Après correction, au point tourné `[35,20,15]` plastique : fermeture
`1,3e-14 MPa`, tangente FD `1,3e-7`. **Performance : ~6,7× plus rapide** que
la condensation Python (`0,42 ms` contre `2,81 ms` par évaluation matériau).

**Mur de robustesse, ouvert** : le Newton conjoint 21 inconnues diverge aux
états plastiques profonds (incrément 8 de l'historique gelé à 12 incréments ;
le P43 20×20 à 8 incréments échoue aux points profonds). Indépendant du
départ (échoue même depuis l'état de la référence), du Jacobien (analytique
ou FD), de `@IterMax` (200) et de la normalisation de la fermeture (module
`1e6`). La référence imbriquée converge là. Limite de la structure du Newton
conjoint dans le DSL Implicit, pas de la formulation de la fermeture.
Benchmark : `scripts/benchmark_srix_umat_gps_p43.py`. La suite possible :
acter le mur (backend qualifié sur la plage modérée) ou investiguer le bassin
du Newton à l'inc 8 (la sonde C++ peut imprimer la trajectoire du résidu).

**Reprise du 2026-08-07 (modèle suivant, commit `df59103`) — le mur est
déplacé, le vrai blocage est la tangente.** Trois résultats, dans l'ordre de
mesure :

1. **Pas de repli de branche** : `scripts/diagnose_srix_closure_root_sweep.py`
   balaie `sigma_zz(eps_zz)` avec la loi 3D brute à chaque état committé —
   exactement une racine aux douze incréments, marchant à `-1,0e-3` par
   incrément = `-(eps_xx + eps_yy)` (incompressibilité plastique). La racine
   de contrainte plane existe, est unique et bien séparée où le Newton
   conjoint meurt : le mur est une limite de l'itération, pas du problème.
   (Nuance : le balayage suit la branche naturelle — la racine de fermeture
   de l'UMat lui est invisible ; la multivaluation du §5.1 n'est pas réfutée.)
2. **Le sous-pas franchit le mur de convergence** : le pont divise
   l'incrément par moitiés jusqu'à 1/256 (s0 avancé puis restauré ;
   `commit()`/`revert()` préservés). Sans sous-pas la qualification ne passe
   pas l'incrément 3 ; avec, les trois cas parcourent les douze incréments,
   fermeture à `4e-14 MPa`. Un `@Predictor` transverse (élastique puis
   isochore) a été ajouté à la loi — il ne change pas le comptage de sous-pas
   (l'échec n'est pas un problème de point de départ).
3. **A6 était vacu (défaut de mon test)** : la vérification FD tournait après
   l'historique, à incrément nul → branche élastique gardée → succès factice.
   Corrigé (contrôle à chaque incrément, critère = le pire des douze) : le
   vrai A6 est C1 = `7,4e-01`, C2 = `5,2e+00`, C3 = `3,2e-07` — C1 et C2
   rejetés. Par incrément, SANS sous-pas, la tangente est excellente
   (`7e-8`..`1,8e-6`) : la formulation est juste, c'est le **sous-pas qui
   détruit la tangente** (matrice du dernier sous-pas ≠ matrice de l'incrément
   entier). Le gain de `6,7×` n'est plus acquis (9-10/12 incréments
   sous-pasés) — à remesurer.

**Routes identifiées pour la tangente** (ni l'une ni l'autre implémentée) :
route 1 (la bonne) — sous-pas pour **localiser** `Δeps_zz`, puis Newton
complet sur l'incrément entier depuis la racine localisée, injectée au
`@Predictor` via une variable externe posée par le pont — tangente exacte et
Newton unique préservés ; route 2 — condenser avec la loi 3D brute sur
l'incrément entier (`C^ps = Caa − Cab Cbb⁻¹ Cba`), exact mais une intégration
3D de plus par évaluation. Autres réparations du même commit (préexistantes) :
le `KeyError plastic_strain_2d` du solveur FEM (observables lues depuis le
trial accepté), 1479 tests verts. Détails : §8 de
`docs/explanation/spectral_mechanics/umat_gps_handoff_2026-08-07.md`.

**Résolu le 2026-08-08 — cause unique, trois défauts, qualification ACCEPTÉE.**
La cause racine était un bug de bookkeeping de déformation : le pont GPS
appliquait la déformation TOTALE comme un INCRÉMENT (la référence écrit son
gradient en absolu). La charge imposée croissait en `1+2+3+...` ; l'incrément
1 n'était pas affecté (total = incrément) — d'où la signature « accord à
l'incrément 1, divergence à partir du 2 » lue comme un changement de branche
pendant trois jours. **Il n'y a jamais eu deux branches** : la multivaluation,
le mur de robustesse et la tangente fausse étaient les conséquences de cette
ligne (fix `6bfaf86`). Les campagnes condensées archivées ne doivent PAS être
refaites — la référence avait raison. Deux autres défauts corrigés dans la
foulée : les rotations EBSD passées en vues stridées avec durée de vie
incorrecte (invisible à un point — le champ 400 points échouait), et le
backend qui tournait sur un seul thread.

La formulation finale est plus élégante que la 21 inconnues initiale : le
système garde **18 inconnues** et les trois rangées transverses du résidu
cinématique (écrit en repère global) portent la condition de contrainte
plane ; `ezz/eyz/exz` sont des sorties. Qualification ACCEPTÉE sur les trois
cas : fermeture `2-4e-14 MPa`, tangente FD `1,2-1,6e-7`, accord référence
`1e-11`. Performance : P43 20×20 matériau `1,2-1,7×` la référence ; **100×100
à parité** (`1,02×`) avec une **pénalité d'itérations globales (`85` vs
`57`) mesurée mais non expliquée** — le terme ouvert, et le backend condensé
reste le backend de production. Le sous-pas est localisé aux points fautifs
(0,071 % du lot à 100×100) avec cache prouvé complet. Voir le handoff §8 et
`validation/srix_umat_gps_closure_preregistration.md` (amendement 1
rétracté). La route 1 a été implémentée (3
variables externes `GpsPredictorEzz/Eyz/Exz` lues par le `@Predictor`, re-run
de l'incrément entier depuis la racine localisée). Mesuré : le re-run
**réussit** mais converge vers le même point que le sous-pas (stress et
tangente identiques) — le sous-pas accumulé est déjà une racine du problème
complet. La **tangente cohérente du DSL à l'état plastique profond est fausse**
(1,5-2,2× la différence finie, même à l'identité où la correction est un
no-op) : le blocage n'est pas le chemin du Newton, c'est la mécanique
`D_tdt·Je` du DSL. **Route 2 identifiée** : condenser la tangente qualifiée de
la loi 3D brute sur l'incrément entier (`C^ps = Caa − Cab Cbb⁻¹ Cba`) — le
Schur de la tangente de la référence, exact. Convention des variables de
fermeture à trancher empiriquement avant (ingénieur vs Kelvin).

## Frontières d'extension — fusionné le 2026-08-02

Le diff `kinematics_extension_v1.diff` de GPT Work est intégré : registres de
plugins constitutifs, catalogue déclaratif des lois MFront, critères non locaux
interchangeables. Objectif : accueillir une plasticité cristalline sans
réécrire Newton. Travail fait sur `crystal-plasticity-boundaries`, fusionné
dans `main` sans fast-forward pour que la séquence reste lisible.

**Effet sur les campagnes archivées** : cinq champs de configuration
s'ajoutent, donc tous les manifestes changent. Ce n'est pas une régression
propre à ce travail — `_manifest_data()` hache déjà `_source_fingerprint()` sur
tous les `.py`, donc le manifeste change à chaque commit — mais une campagne
archivée qu'on tenterait de reprendre refusera avec
`existing run manifest does not match`. Les résultats déjà écrits restent
lisibles ; c'est la reprise et l'ajout de partitions qui sont concernés.

**Le diff ne s'appliquait pas** — les six fichiers modifiés échouaient tous et
la fusion à trois points était impossible, aucun blob de base n'étant dans la
base d'objets. Tout a été porté à la main, un commit par fichier.

### Deux corrections apportées au diff

- **Un défaut réel.** Le diff route la boucle du point fixe à travers le
  critère mais laisse **l'évaluation finale** lire
  `observables["equivalent_plastic_strain"]` en direct. Un critère personnalisé
  aurait piloté la boucle puis été silencieusement contourné pour les champs
  retournés. Verrouillé par
  `test_custom_signed_criterion_passes_through_fixed_point_without_clipping`.
- **Un trou fonctionnel.** `cli.py` n'était pas touché, et ses `choices=(...)`
  rendaient tout le registre inatteignable en ligne de commande — la seule
  façon dont ce dépôt lance des campagnes. L'option valide désormais contre le
  registre, et `partition` a gagné `--mfront-behaviour-id`,
  `--constitutive-options`, `--nonlocal-criterion`,
  `--nonlocal-criterion-options`.

J'ai aussi conservé la validation précoce du nom de backend, que le diff
dégradait en simple test de non-vacuité.

### Non-régression : conforme, après correction du critère

Le critère bit à bit que j'avais annoncé **était inatteignable** — voir
`validation/solver_reproducibility_note.md`. Mesuré : la même source relancée
deux fois donne `5,55e-17` d'écart, à cause de `mfront_threads: 8` et de
PyPardiso.

| | écart relatif | verdict |
|---|---:|---|
| bruit run-à-run, source identique | `7,80e-16` | référence |
| local P43, portage | `9,75e-16` | conforme |
| **couplé a200, portage** | **`3,90e-16`** | **conforme** |

Le couplé est le seul à exercer le point fixe refactorisé, et sa déviation est
**plus faible que le bruit run-à-run**. 673 tests avec MFront, zéro skip.

### Ce qui n'est pas fait

La plasticité cristalline **n'est pas implémentée** : le diff ne pose que les
frontières. Une loi au profil d'état différent de J2 doit fournir son propre
plugin, l'adaptateur MGIS reste à écrire, et l'affectation des orientations
EBSD aux points de Gauss est entièrement à définir.

## Hygiène du dépôt — à connaître

Le dépôt vit dans un **dossier synchronisé** qui fabrique des « conflicted
copy ». 18 artefacts ont été supprimés le 2026-08-02, dont **5 copies de
`.git/index`** et deux refs dupliquées ; sauvegardes dans
`.git/conflicted-ref-backups/cleanup-2026-08-02/`.

C'est ce mécanisme qui avait sorti le commit `553b806` du reflog. Il est
préservé par le tag **`recovered/553b806`**, poussé sur `origin`. Son contenu
est identique octet pour octet à `main` : rien n'était perdu comme travail,
seulement le nœud d'historique.

**Le risque reviendra** tant que `.git/` reste synchronisé.

## Où en est le projet

Cette section se lit en deux minutes. Le détail chronologique est en section 14.

**Le résultat principal du projet.** Le couplage micromorphique est **le seul
levier testé qui rapproche mesurablement l'EF de la DIC**. Sous observation
symétrique, `alpha=4` améliore la L2 relative de `0,194`, soit **9,6× la marge
de bruit** de cette métrique, et la corrélation de `0,059`, soit **3,2×** sa
marge. Surtout, l'aire active q90 passe de **`+61,3 %`** au-dessus de la
référence DIC pour le modèle local à **`+2,6 %`** pour `alpha=4`.

Le contraste avec tout le reste est ce qui donne sa force au résultat :

| Levier | Effet sur l'accord DIC |
|---|---|
| couplage micromorphique | L2 relative améliorée de `9,6` marges |
| histoire temporelle mesurée | aucune métrique au-delà de sa marge |
| filtrage modal du bord | aucune métrique au-delà de sa marge |
| carte matériau homogène | meilleur L2 global, mais en supprimant les bandes |

Limite à citer systématiquement : le couplage améliore amplitude, corrélation et
aire active **en dégradant la position** de la localisation — IoU top-10 % en
baisse de `0,046` et IoU q90 de `0,054`, soit `2,4` et `2,5` marges. Le
classement dépend donc de l'objectif, et c'est pour cela qu'aucun `alpha` unique
n'est retenu.

**Ce qui fonctionne et est vérifié.** Le pipeline autonome va des quatre
tableaux DIC bruts aux entrées canoniques puis au calcul partitionné, validé sur
une partition à l'échelle de l'article (234 600 éléments). Le backend MFront/MGIS
est branché dans Newton. Les tenseurs 3D complets sont reconstruits en
post-traitement du solveur 2D. Une loi J2 3D est condensée en contraintes planes
et validée contre le backend natif. Deux comportements micromorphiques existent,
avec point fixe `p ↔ chi` transactionnel dans chaque Newton.

**L'histoire temporelle mesurée s'exécute désormais de bout en bout.** Elle a
longtemps échoué sur la transition état 3 → état 4 ; la cause était un **défaut
logiciel** (prédicteur élastique résolu sur un buffer CSR écrasé), pas une limite
mécanique. Corrigé le 2026-07-30. Aucun résultat archivé n'était affecté.

**Ce que l'on sait du chargement.** À point final identique, le trajet change
PEEQ de `15,8 %` sur le cœur, concentré dans les bandes — ce n'est ni de la
discrétisation (`0,20 %`) ni du rochet de bruit. Mais sous observation
symétrique, les deux trajets sont **indiscernables** face à la DIC. C'est un
résultat d'identifiabilité : cet observable ne peut ni valider ni réfuter
l'histoire mesurée.

**Ce qui reste ouvert et bloquant scientifiquement.**

- `Hchi` et `ell` ne sont pas identifiés séparément ; aucune longueur matérielle
  transférable n'est revendiquée ;
- l'opérateur d'observation V3 a montré que DISFlow change amplitude, morphologie
  et classement : toute nouvelle identification micromorphique sur l'ancien
  objectif reste suspendue ;
- pas de branche de décharge ni de force synchronisée (lot V0), donc pas
  d'observable sensible à la plasticité cumulée ;
- exécution et raccordement des 100 partitions du ROI complet à planifier.

**Systématique connu à porter.** Toutes les campagnes micromorphiques archivées
utilisent le trajet proportionnel ; le systématique de `16 %` sur PEEQ ne
renverse pas leurs classements (marges bien plus larges) mais doit être cité.

**Profil DISFlow : 4/1 primaire, mais par provenance documentée seulement.** Le
test de reproduction du champ archivé (`000294 -> 000334`) ne discrimine pas :
`1,673 %` pour `legacy_script_2021` (4/1) contre `1,738 %` pour
`declared_medium_v4` (8/3), rapport `1,04` pour un facteur pré-enregistré de
`1,5`. Sur le sous-support P43, 8/3 est même très légèrement devant.

Le résidu commun de `~1,6–1,7 %` n'est **pas** dû au patch ni au stride : il est
dominé par le **raffinement variationnel à l'échelle la plus fine**, identique
dans les deux profils (`alpha=100`, `delta=1`, `gamma=0`, `epsilon=0,002`,
30 itérations, échelle native 0). Ce stade travaille sur l'image pleine
résolution et recouvre largement l'appariement grossier, sur lequel agissent
patch et stride. Le résultat nul est donc **attendu**, signature d'une chaîne
dominée par le raffinement, et `1,7 %` d'écart à l'archive est proche.

Confondant retiré le 2026-07-31 : j'avais listé un masquage avant corrélation
dans le pipeline historique. C'est faux, le masquage se fait **après** ; avec
ces méthodes de flux dense, masquer avant est inefficace car le solveur propage
l'information à travers la zone masquée. Il ne manque donc aucune étape de
masquage à la reproduction.

Conséquence importante : les deux profils sont quasi indiscernables sur la
**donnée**, mais diffèrent de `1,8 marge` sur l'accord EF/DIC — plus que le
trajet de chargement ou le filtrage modal. Choisir un profil sur son score
reviendrait donc à le choisir sur presque rien de mesurable dans les données.
C'est la circularité que la règle du dépôt interdit, désormais chiffrée.

Artefacts : `validation/dic_profile_endpoint_reproduction_preregistration.md` et
`validation/dic_profile_endpoint_reproduction_results.md`.

## Prochaine action

**La porte micromorphique est OUVERTE** — décision du 2026-07-30. Les lots V2 et
V3 qui la conditionnaient sont terminés, et le résultat ci-dessus est le plus
solide du projet. Une nouvelle campagne d'identification couplée est autorisée
aux trois conditions suivantes :

1. l'objectif est évalué **sous observation symétrique** ; l'objectif brut de
   l'ancienne surface paramétrique n'est pas réutilisable ;
2. la campagne est pré-enregistrée, avec des marges de significativité prises
   dans les intervalles de sensibilité au bruit DIC déjà mesurés ;
3. le systématique de `16 %` lié au trajet est cité, sans être appliqué comme
   correction.

La campagne autorisée par cette décision est **spécifiée, chiffrée et prête à
déléguer** : voir la section « Campagne à lancer » ci-dessous et le mode
opératoire cluster `docs/how-to/run_micromorphic_identification.md`.

**Normalisation de `k` réglée le 2026-07-31.** Le ressort n'est plus un
paramètre libre : il est fixé par le **principe de l'écart**, le solveur ayant
droit de s'écarter de la mesure d'exactement le bruit mesuré
`sigma = 9,40e-5 mm`. La calibration converge en 7 itérations, à `97,3 %` de la
cible, et donne `k/K_ref = 2,7` — un ressort du même ordre que le matériau,
donc bien conditionné, contre `~1e7` qu'exigerait une imitation de Dirichlet dur.

Vérification décisive : à la raideur calibrée, l'écart à la solution par
élimination **égale le misfit** à `5 %` près sur trois décades. La pénalisation
perturbe donc la mécanique d'exactement l'incertitude qu'elle représente, pas
davantage. `BOUNDARY_MISFIT` devient une carte des endroits où mécanique et
mesure sont incompatibles **au niveau du bruit** ; elle localise le désaccord
sans l'attribuer.

Reste à faire : la calibration n'a pas été exécutée sur l'opérateur élastique de
P43, donc la valeur de production de `k` pour cette ROI n'est pas connue. Il
faut assembler la raideur élastique hors de la boucle Newton.

Artefacts : `validation/boundary_penalty_calibration_preregistration.md` et
`validation/boundary_penalty_calibration_results.md`.

## Outils de comparaison objective EVM DIC/FEM — lots 1 et 2 livrés

Cahier des charges « outils objectifs de comparaison spatiale entre EVM DIC et
EVM FEM ». Décision d'architecture : `docs/adr/0009-observed-evm-comparison-operator.md`.
Référence : `docs/reference/band_comparison.md` et
`docs/reference/observation_operator.md`.

**Lot 1 — opérateur d'observation symétrique.** Constat central : la chaîne
`déplacements FEM → image déformée → DISFlow → EVM` **existait déjà** dans
`replay_dic_observation`, dont le manifeste déclarait littéralement
`"mode": "synthetic_disflow"`. En créer un second aurait rendu les résultats
archivés incomparables. Étendu en place : 5 artefacts d'audit ajoutés, version
d'OpenCV et état du dépôt enregistrés (sans la version, le manifeste n'était
**pas** reproductible, ce profil laissant 4 réglages aux valeurs d'usine),
contrat de grille et de résampling explicités, et **garde-fou métrologique**
refusant une échelle finale non native — le défaut qui avait donné un MTF-50 à
`127 px` au lieu de `49` et fait invalider la première campagne.

**Lot 2 — géométrie des bandes et métriques.** Nouveau paquet
`src/fem_inhouse/validation/` : `band_geometry.py`, `band_profiles.py`,
`falsification_cases.py`. La géométrie est construite depuis la **seule** DIC
puis gelée, pour qu'aucun candidat ne puisse déplacer les objets qui le jugent.

Deux décisions prises sans arbitrage, à revoir si besoin :

- **pas de scikit-image**. Amincissement Zhang-Suen implémenté ici, ~40 lignes
  déterministes et testées, plutôt qu'une dépendance lourde ;
- **ligne centrale par plus court chemin euclidien**, pas par nombre de sauts.
  Piège trouvé en test : le pruning laisse un résidu d'un pixel à côté du tronc
  (en 8-connexité un pixel au-dessus d'une ligne en touche trois, donc son degré
  vaut 3 et la marche s'arrête avant), et ce résidu offrait un détour diagonal
  **au même nombre de nœuds**, coudant la ligne centrale. La pondération
  euclidienne le règle ; un test le verrouille.

Les trois définitions de largeur (FWHM, intégrale, second moment) sont rapportées
côte à côte, aucune n'est déclarée supérieure, et `measure_width` renvoie un
statut explicite (`no_crossing`, `too_weak`, `peak_at_edge`, `multimodal`,
`empty`) — une section sans croisement à mi-hauteur est un échec différent d'une
bande absente, et les moyenner serait faux.

Ambiguïtés consignées pour les lots suivants, dans l'ADR : support image nodal
contre EVM centrée élément (décalage d'un demi-pixel, qui s'annule aujourd'hui
mais pas pour une ligne centrale en coordonnées pixel), interpolation identité,
mode de bord `BORDER_REFLECT101`, masque purement déclaratif, renommage
historique `U = u_y`.

**Lot 3 — FSS multiscalaire et structure du résidu.**
`fractions_skill_score.py` répond à « à quelle échelle spatiale l'aire active du
candidat devient-elle compatible avec la DIC », ce qu'un recouvrement pixel à
pixel ne peut pas : une bande légèrement déplacée y est punie deux fois. Seuils
calculés sur la DIC et appliqués **sans recalage** au candidat ; échelles
`1, 2, 4, 8, 16, 24, 32, 48, 64, 96` px pré-enregistrées ; fraction normalisée
par les pixels **valides** et non par l'aire de fenêtre, ce qui garde le sens au
bord du support. Deux champs vides donnent `nan`, pas `1.0` — annoncer une
compétence parfaite à un candidat qui ne prédit rien serait trompeur.

`residual_structure.py` fixe la convention `R = EVM_DIC − EVM_FEM,obs`, donc un
résidu **positif est de la déformation manquante**. Partition d'énergie
corridor/fond, spectre radial, variogrammes directionnels, associations avec la
DIC, et typologie heuristique (§8.3) qui renvoie ses chiffres et **se déclare
diagnostic, pas résultat démontré**. Les autocorrélations et longueurs de
cohérence réutilisent `postprocessing.spatial_correlation`, déjà en place.

Défaut de conception trouvé en test, à retenir : **le module du gradient est
aveugle au déplacement**. Un résidu de décalage est antisymétrique en travers de
la bande et le module est symétrique positif, donc leur corrélation s'annule
(mesuré `2,9e-14` pour un décalage de 2 px). Le §8.2 demandait « corrélation
avec le gradient » ; seules les dérivées **signées** détectent un défaut de
placement. Les deux sont désormais rapportées et un test verrouille la cécité
du module.

**Lot 4 — bootstrap spatial apparié et décision Pareto.**
`spatial_bootstrap.py` rééchantillonne des **blocs de sections consécutives**,
jamais des pixels : les pixels voisins d'une bande sont fortement corrélés et
les traiter comme indépendants donnerait des intervalles assez étroits pour
déclarer significative n'importe quelle différence. Le tirage est **apparié** —
un réplicat choisit un jeu de sections et tous les candidats y sont notés — car
tirer indépendamment par candidat comparerait des réalisations de bruit
distinctes. Les bandes sont rééchantillonnées séparément puis moyennées à
**poids égal**, pour qu'une bande longue ne domine pas une courte. Blocs
circulaires, `seed` et nombre de tirages enregistrés.

`pareto_decision.py` élimine d'abord, domine ensuite, et n'impose aucun
vainqueur : « un seul non dominé », « plusieurs non dominés », « aucun candidat
ne passe » sont les trois conclusions permises. Un critère obligatoire non
mesuré **élimine** — ne pas mesurer n'est pas réussir. `worst_band_vector`
renvoie un **vecteur, pas une somme** : sommer laisserait une bande très bien
reproduite compenser une bande perdue, et la pire bande peut différer d'un
critère à l'autre.

Bug trouvé en test, qui aurait biaisé **toutes** les comparaisons : avec une
inégalité stricte, deux candidats identiques donnent une différence nulle à
chaque tirage, donc une probabilité de `0,0`, donc le verdict `robustly_worse`.
Les ex æquo comptent désormais pour **une demi-victoire** ; deux tests
verrouillent la règle et le cas entièrement ex æquo.

**Piège d'outillage résolu le 2026-07-31, à connaître.** `mypy` donnait par
intermittence 2 puis 4 erreurs, que j'avais d'abord attribuées à tort à un
artefact de cache. La vraie cause est **l'environnement** : une fois
`env.sh` de TFEL sourcé, `PYTHONPATH` expose
`/home/jeff/.local/lib/python3.12/site-packages`, donc un second numpy, et la
résolution des surcharges change — `np.gradient` et `array.shape` s'infèrent
différemment.

Cela compte parce que la procédure du §1.0 impose de sourcer `env.sh` avant les
tests : un développeur qui la suit voyait 4 erreurs, un autre zéro. Les quatre
étaient réelles et sont corrigées en fixant explicitement le rang et les
indices de tuple. **Vérifier mypy dans les deux environnements**, pas seulement
dans le shell nu.

**Lot 5 — pré-enregistrement scientifique rédigé, EN ATTENTE DE TA VALIDATION.**
`validation/observed_evm_candidate_comparison_preregistration.md`. Le §21 exige
que la campagne ne soit lancée qu'après validation de ce document.

Point d'intégrité placé en tête du document plutôt que caché : les scores
**globaux** des quatre candidats sont **déjà archivés et connus** (`local`
`0,4858`, `alpha=1` `0,3542`, `alpha=2` `0,3197`, `alpha=4` `0,2917` en L2
relative). Une borne choisie juste au-dessus ou en dessous de ces valeurs serait
du théâtre. **Aucune borne d'élimination n'en est tirée** : elles viennent
toutes de la chaîne de mesure — plancher de bruit, MTF-50 à `49 px`, longueur de
cohérence `38,2 px` — ou d'une propriété géométrique des bandes. Ce qui est
réellement aveugle, et où le pré-enregistrement a force, c'est tout le
par-bande, le multiscalaire et le bootstrap, jamais calculés.

Six candidats, dont les deux références négatives (`homogeneous` pour le fond,
`translated` pour la localisation), toutes converged à 20 incréments sur le
chemin proportionnel. Les runs multipas du 2026-07-30 sont **exclus
volontairement** : ils diffèrent par le trajet et le nombre d'incréments, les
mélanger confondrait la question constitutive et la question de trajet.

Bornes et leur origine : aire minimale d'objet `256 px` (plus étroite que le
MTF-50 dans une direction), élagage à `16 px`, bloc bootstrap de 8 sections
(`32 px`, sous la longueur de cohérence pour que les blocs ne soient pas
artificiellement indépendants), 10 000 tirages, graine `20260731`.

**Amendé le 2026-07-31 après trois arbitrages.** Un trou trouvé en relecture :
le document ne disait pas **quel seuil définit la géométrie des bandes**, ce qui
décide combien de bandes existent. Réglé par une mesure sur la DIC seule :

| Seuil | Objets ≥256 px | Seconde bande |
|---|---:|---|
| q80 | 3 | complète, `5 639 px` |
| q90 | 2 | tronquée, `1 666 px` |
| q95 | **1** | **absente** |

À q95 la prémisse « deux bandes » du §3.4 s'effondre ; à q90 la seconde n'est
qu'un fragment sur lequel reposerait toute l'analyse « pire bande ». **q80
retenu**, seul seuil où les deux bandes existent entières. Le seuil de géométrie
et les seuils d'activité du FSS sont deux choses distinctes, ce que la première
version confondait implicitement.

Second constat structurel : l'élagage sature à `~26 %` du squelette sur le
chemin principal et ne bouge plus au-delà de `32 px` (testé à `128`). Ce ne sont
pas des barbules mais des **boucles**, que l'élagage par extrémités ne peut pas
retirer. Les objets sont des réseaux, pas des rubans ; la ligne centrale reste
l'axe mais ne résume pas la topologie.

Deux autres arbitrages : **E5 rétrogradé** en « rapporté seulement » — les
fractions actives sont déjà connues, un critère non aveugle ne doit pas pouvoir
éliminer ; et le critère FSS du Pareto devient **l'échelle minimale atteignant
0,7**, le `16 px` précédent étant un choix injustifié.

**Sélection automatique, décision du 2026-07-31 : plus aucune étape manuelle.**
Un objet est une bande si sa ligne centrale atteint le MTF-50 de la chaîne
(`49 px`) — une région dont l'axe principal est plus court que la longueur de
résolution ne peut pas être affirmée comme bande. Mesuré sur la DIC seule, la
règle sépare par un facteur **> 6** à chaque seuil (`175` contre `26 px` à q80).
Fixée par l'instrument, pas par inspection, et reproductible sans humain.

**Lecture « réseau » corrigée, c'était mon bug.** J'avais annoncé des centaines
de boucles et conclu à des réseaux. Le compte se faisait par `E − V + C` sur le
graphe de pixels en 8-connexité, où trois pixels en coin forment un triangle :
chaque coin d'un chemin large d'un pixel était compté comme une boucle, d'où
`661` pour la bande 1. Compté correctement sur la **région** (fond 4-connexe
dans un avant-plan 8-connexe) : **63 trous, le plus grand `32 px`**, aucun
n'atteignant la borne résolvable de `256 px`. **Aucune structure cellulaire.**

Même écueil sur les branches : `1472` branches dont **96 % font ≤ 2,4 px**, du
bruit d'axe médian sur un bord dentelé, et seulement `28` atteignent `16 px`.
Les modes d'orientation non filtrés donnaient `7, 52, 97, 142°`, soit exactement
les quatre directions du réseau de pixels. L'orientation n'est donc lue que sur
les branches résolvables, et la bande 2 n'en a aucune.

Mesures de forme qui survivent : `main_path_share` (`0,13` et `0,15` — larges et
dentelées, pas des rubans fins), nombre de trous avec borne de résolvabilité,
nombre de branches résolvables, aire et longueur d'axe.

**Approche de segmentation remplacée le 2026-07-31 : Otsu + `regionprops`.**
Les seuils quantiles étaient arbitraires et mon analyse de squelette n'a rien
conclu ; les deux sont abandonnés. Otsu est déterminé par les données, donne
`4,535e-3` sur la DIC (= q74,5) et exactement **deux objets** avec un écart de
facteur **34** au fragment suivant, contre 15 à q80. Le seuil est calculé **une
fois sur la DIC** et appliqué inchangé : le recalculer par champ laisserait
chaque candidat redéfinir « actif » et masquerait toute perte d'amplitude.

**Le résultat le plus fort de la session, mesuré sur les seules références
négatives** — aucun candidat scientifique n'a été consommé :

| Champ | actif | objets ≥256 px | excentricité | petit axe | orientation |
|---|---:|---:|---:|---:|---:|
| DIC | `26,2 %` | 2 | `0,94`, `0,93` | `104`, `72 px` | `−58`, `−46°` |
| homogène | `0,0 %` | **0** | — | — | — |
| translaté | `27,0 %` | **1** | `0,65` | `269 px` | `−15°` |

Le contrôle translaté a **la même aire active que la DIC à un point près** et
une morphologie sans rapport : un objet cellulaire au lieu de deux bandes
allongées. **Une métrique d'aire ne peut pas les séparer, la morphologie les
sépare immédiatement.** C'est la prémisse du cahier des charges, désormais
mesurée et non plus affirmée.

`scikit-image` ajouté aux dépendances, déclaré dans `pyproject.toml`.

**Lot 6 exécuté le 2026-07-31 sur les résultats archivés.** Résultats :
`validation/observed_evm_candidate_comparison_results.md`.

**Condition d'échec 2 déclenchée** : cinq candidats sur six sont non dominés,
dont le contrôle négatif translaté. Les critères pré-enregistrés **ne
discriminent pas**, aucun classement n'est publié.

**Résultat scientifique, distinct de cet échec** : **aucun candidat ne
reproduit la morphologie à deux bandes de la DIC**. Les quatre produisent un
objet fusionné unique là où la DIC en a deux, avec un petit axe environ double
(`198–217 px` contre `104` et `72`) et une excentricité de `0,79` contre `0,94`.
De `alpha=1` à `alpha=4` le petit axe tombe de `213` à `198 px` : la tendance
est dans le bon sens et très loin de suffire.

**L'aire active est inutilisable ici** : tous les candidats et le contrôle
négatif sont à moins de trois points de la DIC.

**Le contrôle translaté n'est pas rejeté par les critères enregistrés.** Sur
erreur de ligne centrale, de largeur et de masse il est *dans* la plage des
candidats couplés, et son erreur de masse est statistiquement indiscernable de
`alpha=1` (`P=0,391`) et `alpha=4` (`P=0,610`). Or c'est un contrôle construit
en déplaçant les cartes matériau. La morphologie le sépare immédiatement
(excentricité `0,645`, petit axe `269 px`) — **mais la morphologie n'est pas
dans les critères Pareto enregistrés**. L'ensemble de critères est donc
insuffisant, et c'est le contrôle négatif qui l'a prouvé. **Non réparé après
coup** : ajouter la morphologie au Pareto maintenant, en sachant que ça
changerait la réponse, serait exactement l'ajustement post-hoc que le protocole
interdit. Un ensemble révisé demande un nouveau pré-enregistrement.

Le contrôle homogène est correctement **éliminé** par E2 (`42,9 %` de sections
portant une bande, sous la borne de `50 %`).

**Deux défauts trouvés en exécutant, corrigés avant ces chiffres :** le critère
de détection était vide de sens (l'excès est positif sur presque tout profil,
le fond étant une médiane des queues) ; et validité de section et détection
étaient confondues, si bien que **la DIC obtenait `0,46` sur sa propre bande** —
band1 traverse le cœur en diagonale et seules `43` de ses `94` sections restent
dans le support à ±40 px. Contrôle de correction retenu : la DIC doit détecter
ses propres bandes à `100 %` des sections valides, ce qu'elle fait maintenant.

Le bootstrap confirme par ailleurs le classement global archivé : le couplage
est **robustement meilleur** que le modèle local sur l'erreur de masse
(`P = 1,000`).

Aucun calcul mécanique lancé, aucun paramètre micromorphique sélectionné,
aucun claim modifié.

### Ensemble de critères v2 — exécuté le 2026-08-01, **non validé**

Protocole : `validation/observed_evm_morphology_criteria_preregistration.md`,
validé le 2026-08-01 avec ses deux amendements.
Résultats : `validation/observed_evm_morphology_criteria_results.md`.
Code : `src/fem_inhouse/workflows/compare_observed_evm_morphology.py`.

**Conditions d'échec 3, 4 et 5 déclenchées. Aucun classement publié.**

Le profil primaire donne une réponse propre — `alpha=2` seul non dominé, le
contrôle translaté dominé, test d'acceptation réussi. **Le profil aveugle la
refuse** : les cinq survivants sont non dominés, dont le contrôle translaté.
L'ensemble de critères n'est donc **pas validé** : il réussit son test
d'acceptation sur le profil dont la morphologie était déjà connue au moment de
choisir les critères, et échoue sur celui qui n'était jamais passé dans le
pipeline. C'est exactement la comparaison que la confirmation aveugle avait été
enregistrée pour faire, et elle tranche contre les critères.

**G1 n'est pas un tampon.** Il a retiré **deux critères sur sept**, à
l'identique sur les deux profils, avant tout scoring :

- `abs(log)` du rapport de petit axe, `tau = −0,111` — **aveugle à la
  translation**. Un décalage de `16 px` le déplace de `0,0079`, une erreur
  d'amplitude de `10 %` de `0,335`, quarante fois plus. C'est un descripteur de
  largeur, et translater une bande ne change pas sa largeur. **C'était un des
  trois critères de morphologie choisis pour cette campagne** ;
- fraction d'énergie de couloir, `tau = −0,286` — **classe une bande parasite
  comme le meilleur cas de toute l'échelle** (`0,218`, sous tous les autres
  défauts) : une bande absente de la référence dépose son énergie *hors*
  couloir, ce qui fait baisser la fraction. Le critère récompense le défaut.

**G2 passe sur les deux profils**, grâce à l'amendement 2 : l'écart de ligne
centrale est apparié à la DIC. La définition absolue de v1 n'est pas nulle pour
la DIC contre elle-même.

**Le défaut trouvé par le profil aveugle.** Sur `declared_medium_v4` le contrôle
translaté obtient un **nombre d'objets parfait** — mais son second objet est un
éclat de `413 px` (petit axe `17,5 px`) contre la seconde bande DIC de
`8 340 px`. **Un comptage brut de composantes connexes n'est pas un descripteur
de morphologie** : un fragment à peine au-dessus du plancher de `256 px` achète
un score parfait. Le critère est jouable, et le contrôle négatif l'a joué.
Figure : `object_count_speck.png`. **Non réparé** — tout correctif serait choisi
en sachant ce qu'il fait à ce contrôle et relève d'un pré-enregistrement v3.
Test qui verrouille le défaut :
`test_the_object_count_is_satisfiable_by_a_speck`.

**Limite trouvée dans le dispositif lui-même** : les deux profils **partagent
leur référence**. `dic_evm.npy` est identique octet pour octet
(`f8cde6b0…`) parce que l'EVM DIC est reconstruit depuis les déplacements
mesurés et ne passe jamais par DISFlow. Le recalcul d'Otsu sur le second profil
était donc un **no-op**. La confirmation aveugle teste un changement
d'observation des candidats seulement, pas une mesure indépendante de la
référence : elle est plus faible que le pré-enregistrement ne le laissait
entendre.

**Ce que la non-blindness a coûté, mesuré** : sans le profil aveugle, la
campagne aurait publié un candidat unique non dominé et un contrôle négatif
rejeté — conclusion propre, publiable, et fausse.

Le bootstrap reste cohérent sur les deux profils : le couplage est robustement
meilleur que le modèle local sur l'erreur de masse (`P = 1,000`), mais
`alpha=2` ne se sépare **jamais robustement** du contrôle translaté
(`0,845` et `0,895`), et `alpha=1` comme `alpha=4` en restent indiscernables.

#### Le pré-enregistrement, pour mémoire

`validation/observed_evm_morphology_criteria_preregistration.md`, écrit le
2026-07-31.

Il remplace les seuls critères Pareto de v1 ; candidats, opérateur
d'observation, seuil Otsu, géométrie des bandes, bootstrap et vocabulaire de
décision sont repris tels quels. Sept critères, dont **trois de morphologie**
(nombre d'objets, `abs(log)` du rapport de petit axe, écart d'excentricité) ;
l'erreur de largeur par section est retirée comme redondante avec le petit axe.

**Ce document n'est pas aveugle et le dit.** La morphologie des six champs est
déjà connue, et on sait qu'elle sépare le contrôle translaté : ces critères
sont choisis *parce qu'ils* donnent la bonne réponse sur un contrôle. Trois
dispositifs bornent ça, chacun portant sur quelque chose de jamais calculé :

- **G1**, le banc de falsification de `falsification_cases.py`, qui **n'a
  jamais tourné** hors de son test unitaire — un critère qui ne classe pas
  correctement les défauts connus est retiré avant tout scoring ;
- **G2**, la DIC contre elle-même, qui doit être parfaite sur chaque critère —
  c'est ce contrôle qui avait révélé le défaut de détection de v1 ;
- **la confirmation aveugle sur `declared_medium_v4`**, jamais passée dans ce
  pipeline. Otsu y est **recalculé sur la DIC de ce profil**, pas repris du
  premier. Les quatre modèles y sont archivés, **les deux contrôles non** : il
  faut les ré-observer depuis leur grille de déplacement, sans remécanique.

**Test d'acceptation enregistré** : les critères ne sont acceptés que si le
contrôle translaté est dominé ou éliminé. C'est une condition sur les
*critères*, pas sur les candidats.

### Défaut de reproductibilité de v1, trouvé et corrigé

En vérifiant l'archive avant d'écrire v2 : les quatre modèles venaient de
`reference_data/`, mais **les deux contrôles étaient lus depuis un scratchpad
de session**, non persistant. Les conclusions qui reposent sur le contrôle
translaté — donc l'échec des critères — n'étaient pas reproductibles.

Le scratchpad était encore intact. Les deux champs sont archivés dans
`validation/reference_data/observed_evm_controls_p0043_v1/` (LFS, manifeste
`SHA256SUMS`), et **leurs SHA-256 reproduisent exactement ceux du `report.json`
de v1** : ce sont bien les champs utilisés, pas des régénérations. Aucun
chiffre de v1 ne change ; v1 devient vérifiable. Les grilles de déplacement
sont archivées avec, pour la ré-observation sur le second profil.

**Leçon générale** : un contrôle négatif lu depuis un répertoire temporaire
n'est pas un contrôle. Toute campagne doit archiver ses contrôles au même
niveau que ses candidats.

## Campagne EN COURS — matrice (ell, alpha) sur P43 et outil de sélection

**Lancée le 2026-08-01 vers 10 h. Mécanique en cours, fin estimée vers 19 h.**
Protocole : `validation/p0043_small_parameter_matrix_preregistration.md`,
validé avec corrections C1–C4 et amendements A1–A4.
Validation §9 des indicateurs : **passée**, résultats dans
`validation/p0043_indicator_validation_results.md`.

### Comment reprendre si la session s'interrompt

Tout est reprenable, dans cet ordre :

```bash
.venv/bin/python scripts/run_p0043_parameter_matrix.py        # saute les points complets
bash scripts/finish_p0043_campaign.sh                          # attend, observe, score
```

Le second attend la fin du premier, lance les observations manquantes par
l'opérateur symétrique, puis score les deux profils DISFlow. Suivi vivant dans
`results/mm-matrix-logs/progress.txt` (point courant, incrément, ETA) et un log
verbeux par point.

### Ce qui est déjà acquis

- **`(alpha=4, ell=20 um)` ne converge pas** : cascade de cutbacks depuis
  l'incrément 9 jusqu'au plancher. Amendement A3 : le point est rapporté et
  **exclu**, ses réglages ne sont pas retouchés — les retoucher le rendrait
  incomparable aux quinze autres. Conséquence : la paire iso-`Achi` à `1600`
  perd un membre, seule celle à `800` reste comparable ;
- **la chaîne d'observation est reproductible octet pour octet** : les trois
  points archivés à `ell=58,88`, ré-observés aujourd'hui, redonnent leurs
  SHA-256 ;
- **le §9 a rattrapé une erreur de facteur douze dans le plancher de mesure**.
  Le résidu de répétition calé sur l'amplitude de déplacement produisait
  `1,64e-3` d'EVM RMS contre `1,363e-4` mesuré. Il est désormais calé sur la
  déformation, ce que les indicateurs consomment. Sans ça, `D_self` valait
  `0,737` en présence, **pire que le modèle local lui-même**.

### Décisions à connaître avant de lire les résultats

- **le front de Pareto est calculé sur les défauts BRUTS** (correction C3), là
  où la normalisation ancrée sur les contrôles ne peut pas agir. Seul le
  minimax utilise `Z` ;
- **`D_null` est pris par indicateur**, comme le meilleur score qu'atteint
  l'un ou l'autre contrôle, et **lequel est enregistré** ;
- **la zone se construit sur des différences appariées** (amendement A4), pas
  sur des bandes qui se recouvrent — les tirages partagent leurs tuiles, donc
  deux bandes peuvent se recouvrir pendant que la différence appariée n'approche
  jamais zéro. Test : `test_the_zone_uses_paired_differences_not_overlapping_bands` ;
- **le cas A exige en plus que le gagnant batte les deux contrôles** sur au
  moins un indicateur chacun ;
- **le bootstrap utilise des tuiles de 49 px** (amendement A2), pas le « bloc 8 »
  du cahier des charges, qui vient du bootstrap 1D de v1 et se situe très en
  dessous de la cohérence mesurée de `38,2 px` ;
- **le plancher de reproductibilité solveur** vient du réplicat `(alpha=2,
  ell=40)` à 40 incréments et s'imprime sous chaque carte thermique. Deux
  points plus proches que lui sont indiscernables quoi que dise le bootstrap.

### Modules

`validation/selection_indicators.py` (les quatre défauts, normalisation,
minimax), `validation/tile_bootstrap.py` (rééchantillonnage spatial, sommes par
tuile exactes et 4x plus rapides), `workflows/validate_selection_indicators.py`
(§9), `workflows/select_p0043_parameters.py` (front, minimax, stabilité,
iso-`Achi`, sept figures, sept sorties §13).

**Reste à faire** : le document de résultats interprétatif avec la conclusion
§12 — cas A, B ou C — une fois les deux profils scorés.

## Diagnostic exploratoire — critères de fluctuation sur les gradients

Exécuté le 2026-08-01. Rapport :
`validation/gradient_fluctuation_criteria_diagnostic.md`.
Code : `src/fem_inhouse/validation/gradient_fluctuation.py` et
`src/fem_inhouse/workflows/compare_gradient_fluctuation_criteria.py`.

**Étude exploratoire, pas une procédure de décision.** Aucun paramètre
sélectionné, aucun candidat déclaré optimal, rien sur la validité de la
formulation non locale. Les jeux v1 et v2 sont intacts et aucun de ces critères
n'entre dans un front de Pareto. Travaille sur les **déplacements**, pas sur
l'EVM.

**Limite d'interprétation, à répéter partout** : toutes les solutions haute
fidélité comparées ici utilisent une seule portée spatiale, `ell = 58,88 um`,
avec `alpha` dans `{0,1,2,4}`. C'est une coupe de l'espace `(ell, alpha)`.

**La séparation exigée fonctionne** : sur le champ DIC réel, une rotation
rigide déplace `J_∇u` de `0,119` et laisse `J_ε` à `1,8e-13`. Un déplacement
uniforme ne change rien ; un offset affine est invisible à `J_fluct`.

**Résultat principal, négatif : aucun des quatre critères spécifiés ne rejette
le contrôle homogène.** Il se classe 3e ou 4e sur six partout, **devant
`alpha=1`** sur `J_ε` (`0,438` contre `0,461`) et sur `J_fluct` (`0,934` contre
`0,947`).

**Le mécanisme est structurel, pas accidentel.** `J_fluct` sature à `≈1` pour
tout champ sans contenu à l'échelle considérée, car le résidu se réduit alors à
la référence. **Prédire rien vaut `~1`, prédire au mauvais endroit vaut `>1`** :
le critère récompense le lissage. Mesuré : à `8 px` les champs couplés portent
`10` à `18 %` de l'énergie de déformation haute fréquence de la DIC, le contrôle
homogène `5,8 %`, et tous scorent entre `0,956` et `1,032`. **Aucun critère n'a
de pouvoir de résolution aux échelles fines.** Confusion connue à ne pas
oublier : le rejeu n'ajoute aucun bruit de décorrélation, donc une part de
l'énergie fine manquante est du bruit de mesure absent, pas du modèle absent.

**Le contrôle translaté, lui, est correctement rejeté** par les quatre critères.
Les critères mesurent donc le placement, mais pas la présence.

**Deux quantités auxiliaires du §4.3 font mieux que les quatre critères
principaux** :

- la **corrélation de Pearson de la carte de norme** rejette **les deux**
  contrôles (`0,468` homogène et `0,417` translaté contre `0,629` pour le pire
  modèle) — la seule de tout le jeu spécifié à le faire ;
- le **rapport de quantile q95** présente un **optimum intérieur** à `alpha=2`
  (`0,989`, soit `1,1 %` d'écart) alors que `alpha=1` surestime de `10,6 %` et
  `alpha=4` sous-estime de `12,6 %`. Les critères `L2` sont monotones vers
  `alpha=4` sur les mêmes données.

**Désaccord amplitude / fluctuations, mesuré deux fois** : `IoU q90` est
**inversé** en `alpha` (meilleur à `alpha=1` avec `0,323`, pire à `alpha=4`
avec `0,255`) tandis que tous les critères d'amplitude s'améliorent avec
`alpha` ; et le q95 se retourne à `alpha=2` quand les distances `L2` continuent
de descendre.

**Ce qui reste à faire** : si l'un de ces critères doit servir, il faut un
pré-enregistrement distinct, et **il devra enregistrer un critère de présence**
— l'échec documenté ici est une incapacité à pénaliser l'absence.

Réserve sur un cas synthétique : `band_removal` annule la fluctuation dans un
couloir à bord franc, et la marche au bord du masque domine son score
(`4,7` à `23`). Seul son signe est lisible.

## Campagne à lancer — identification micromorphique

**Statut : spécifiée, chiffrée, prête à déléguer. Non lancée.**
Protocole : `validation/micromorphic_symmetric_identification_preregistration.md`.
Mode opératoire, y compris cluster :
`docs/how-to/run_micromorphic_identification.md`.
Table des points prête pour job array : `campaigns/mm_id_points.tsv`.

**Ce que la campagne répond.** Pas « quel `alpha` est le meilleur », mais :
`Hchi` et `ell` sont-ils **séparément identifiables** depuis cette observation,
ou l'objectif est-il dégénéré le long de `Achi = Hchi * ell**2` ? Si seul le
produit est identifiable, aucune longueur spatiale distincte ne peut être
revendiquée — et c'est un résultat négatif publiable.

**Pourquoi maintenant.** Le couplage est le seul levier testé qui rapproche
mesurablement l'EF de la DIC : la suractivité q90 passe de `+61 %` (local) à
`+2,6 %` (`alpha=4`). Mais `Hchi` et `ell` n'ont jamais été variés
indépendamment : toutes les campagnes archivées font varier `alpha` à
`ell = 58,88 µm` figé.

**Deux pièges, documentés dans le guide.**

1. **Ne pas réutiliser `configs/joint_nonlocal_identification_p0043.yaml` tel
   quel.** Il contient déjà une grille 21×21 et paraît prêt, mais son objectif
   n'applique **aucun opérateur d'observation côté EF**
   (`spatial_filter: Literal["none"]`). C'est l'objectif brut asymétrique que
   V3 a réfuté, et qui a motivé la suspension.
2. **Ne pas activer `spatial_reduction: 2`.** C'est l'économie évidente et elle
   est incompatible : l'opérateur symétrique n'a de sens qu'à un élément = un
   pixel. Seule la réduction temporelle est admise, bornée à 20 incréments car
   la sensibilité mesurée de `0,20 %` porte sur 20 contre 40, pas sur 10.

**Design.** Grille `5 x 5`, `ell ∈ {20, 30, 40, 50, 58,88} µm` et
`alpha ∈ {1, 2, 3, 4, 6}`, avec `Hchi = alpha * 5168,147582748343 MPa`. Trois
points archivés et déjà observés sont réutilisés, donc **22 calculs neufs**.

**Ressources.** `~31 min`, 8 cœurs (`~3,2` effectifs), `~10 Go` RSS et `~150 Mo`
par point ; soit `11 à 12 h` en série, ou une passe de job array. L'étape
d'observation symétrique est séparable et coûte des secondes par point : elle
peut tourner ailleurs que sur le cluster, ce qui évite d'y copier les images DIC.

**Règle de lecture, verrouillée.** Aucun score scalaire unique ne sélectionne un
point : le classement dépend déjà de l'objectif. Livrable = quatre surfaces
métriques, ensemble Pareto, optima par objectif. Marges de significativité fixées
d'avance (`0,0202` / `0,0185` / `0,0189` / `0,0217`). Un Pareto large est un
résultat probable et légitime, pas un échec.

Artefacts du filtrage : `validation/modal_filter_peeq_excess_preregistration.md`
et `validation/modal_filter_peeq_excess_results.md`.

Jalon documentaire au 2026-07-26 : **réécriture publique science-first
terminée et vérifiée**. La navigation principale suit désormais un récit
linéaire DIC → base locale → défaut morphologique → diagnostic de largeur →
modèle micromorphique → identification F0/F1/F2 → portée des claims. Les
guides sont orientés par tâche et les détails MGIS, Kelvin, condensation 3D,
point fixe, CSR et PARDISO ont été déplacés vers `docs/reference/`. Les anciens
rapports restent conservés sous `docs/archive/`, hors navigation et recherche.
`validation/documentation_evidence_registry.json` est la source unique des
conclusions, claims et preuves synthétiques ; son schéma 2 vérifie désormais
les assertions contre les JSON primaires avant génération. Le script
`scripts/generate_documentation_evidence.py` produit les fragments Sphinx et
les figures de preuve. Validation à ce jalon : Ruff vert, mypy vert sur
les 44 fichiers d'alors, 321 tests verts avec MFront réel, HTML Sphinx strict
vert, linkcheck strict vert, PDF LuaLaTeX strict de 48 pages. **Chiffres au
2026-07-30 : 444 tests verts, mypy vert sur 69 fichiers, HTML Sphinx strict
vert, linkcheck strict vert, PDF LuaLaTeX strict de 82 pages compilé et rendu
inspecté.** Le PDF est passé de 48 à 82 pages avec les chapitres et preuves
ajoutés depuis. Reconstruction : `make -C docs html latexpdf`, venv actif ; le
Makefile applique `-W --keep-going` par défaut. Le README compte 85 lignes. La consolidation est publiée jusqu'au
commit `18f0fac`.

Jalon atteint au 2026-07-25 : **couplage constitutif micromorphique J2 sur la
partition P154 d'un découpage 20×20**. Le protocole et les résultats sont
respectivement dans `validation/nonlocal_p154_preregistration.md` et
`validation/nonlocal_p154_validation_results.md`. Le meilleur point testé
(`alpha=2`) passe sept critères sur huit, mais l'aire active q90 reste à
`21,85 %` pour une borne pré-enregistrée de `20 %`. La conclusion est donc
« interaction spatiale partiellement soutenue » et aucun `Hchi` n'est figé.

## 1. Rôle de ce document

Ce fichier est la feuille de route vivante du projet. Il doit être mis à jour à
chaque jalon avec :

- l'état réel des tâches ;
- les décisions scientifiques et techniques ;
- les commandes de validation exécutées ;
- les résultats mesurés ;
- les écarts restant à traiter ;
- la date et, si disponible, le commit correspondant.

Une tâche ne doit être marquée terminée que si son critère d'acceptation est
vérifié par un test, un rapport ou une mesure reproductible.

## 1.0 Environnement d'exécution installé sur cette machine

Cette section est **autoritative** pour reprendre le projet. TFEL, MFront et
MGIS sont déjà installés : ne pas conclure qu'ils sont absents avant d'avoir
chargé leur environnement. Un shell neuf ne voit pas spontanément les bindings
Python MGIS et pytest ignore alors les tests réels MFront.

### Chemins vérifiés le 2026-07-29

| Composant | Chemin |
|---|---|
| Racine du dépôt | `/home/jeff/CNRS/Theses/Adil/Data_code/fem_inhouse` |
| Python du projet | `/home/jeff/CNRS/Theses/Adil/Data_code/fem_inhouse/.venv/bin/python` |
| CLI du projet | `/home/jeff/CNRS/Theses/Adil/Data_code/fem_inhouse/.venv/bin/fem-inhouse` |
| Préfixe TFEL/MFront/MGIS | `/home/jeff/.local` |
| Script d'environnement TFEL | `/home/jeff/.local/share/tfel/env/env.sh` |
| Bindings Python TFEL/MGIS | `/home/jeff/.local/lib/python3.12/site-packages` |
| Exécutable MFront | `/home/jeff/.local/bin/mfront` |
| Bibliothèque des comportements | `/home/jeff/CNRS/Theses/Adil/Data_code/fem_inhouse/build/mfront/src/libBehaviour.so` |
| Sources MFront du projet | `/home/jeff/CNRS/Theses/Adil/Data_code/fem_inhouse/mfront` |

Versions vérifiées : TFEL/MFront `5.1.0` au commit `deee4cd`, MGIS `3.1`.
Le Python du venv est Python `3.12` et charge `mgis` depuis le préfixe
`/home/jeff/.local`, pas depuis le venv lui-même.

### Initialisation obligatoire d'un shell

Depuis la racine du dépôt :

```bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
TFEL_PREFIX="/home/jeff/.local"
PYTHON_ENV="${PROJECT_ROOT}/.venv"

# env.sh n'est pas compatible avec `set -u` lorsque ces variables sont
# absentes. Les initialiser avant de le sourcer.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
source "${TFEL_PREFIX}/share/tfel/env/env.sh"

export PYTHONPATH="${TFEL_PREFIX}/lib/python3.12/site-packages:${PYTHONPATH:-}"
export MFRONT_BEHAVIOUR_LIBRARY="${PROJECT_ROOT}/build/mfront/src/libBehaviour.so"
export PATH="${PYTHON_ENV}/bin:${PATH}"
```

Il n'est pas nécessaire d'exécuter `source .venv/bin/activate` : utiliser les
exécutables absolus de `.venv/bin` est plus robuste. Si l'activation est
souhaitée, la faire **après** avoir défini `PYTHON_ENV` :

```bash
source "${PYTHON_ENV}/bin/activate"
```

### Compilation et canari avant toute validation

```bash
test -x "${PYTHON_ENV}/bin/python"
test -f "${TFEL_PREFIX}/share/tfel/env/env.sh"

bash scripts/build_mfront_behaviour.sh
test -f "${MFRONT_BEHAVIOUR_LIBRARY}"

mfront --version
tfel-config --version
"${PYTHON_ENV}/bin/python" -c \
  'import tfel, mgis.behaviour; print(tfel.getTFELVersion(), mgis.__file__)'
```

La bibliothèque contient quatre comportements :

- `PixelLudwikJ2Plasticity` sous `PlaneStress` ;
- `PixelLudwikJ2Plasticity3D` sous `Tridimensional` ;
- `PixelMicromorphicLudwikJ2Plasticity` sous `PlaneStress` ;
- `PixelMicromorphicLudwikJ2Plasticity3D` sous `Tridimensional`.

Canari MGIS réel, qui ne doit produire aucun `skip` :

```bash
"${PYTHON_ENV}/bin/pytest" -q \
  tests/unit/core/test_mfront.py \
  tests/integration/test_mfront_newton.py \
  tests/integration/test_tensor_histories.py
```

Résultat vérifié le 2026-07-29 : `35 passed`. Pour la suite complète :

```bash
"${PYTHON_ENV}/bin/pytest" -q
```

Résultat vérifié avec ce même environnement : `345 passed` en `24,89 s`,
aucun test MFront ignoré.

Si pytest affiche `MFRONT_BEHAVIOUR_LIBRARY is not set`, il ne teste pas le
backend réel : ce n'est pas une indisponibilité de MFront, mais un shell mal
initialisé. Si `import mgis` échoue, vérifier `PYTHONPATH`. Si le chargement de
`libBehaviour.so` échoue sur un symbole TFEL, vérifier `LD_LIBRARY_PATH` et
re-sourcer `env.sh`. Le script `scripts/build_mfront_behaviour.sh` protège
déjà son propre `source` par `set +u`.

### Convention d'état

- `[ ]` : à faire
- `[~]` : en cours
- `[x]` : terminé et vérifié
- `[!]` : bloqué

## 1.1 Stratégie prioritaire de validation — décision du 2026-07-27

Cette section est **autoritative pour la prochaine phase**. Elle suspend les
nouveaux balayages de `alpha`, `Hchi` et `ell` décrits plus bas tant que la
chaîne de mesure et l'opérateur d'observation ne sont pas caractérisés. Les
sections historiques restent conservées comme journal de provenance ; elles
ne constituent plus l'ordre d'exécution courant.

### Position scientifique

Le modèle 2D est interprété comme un modèle effectif obtenu en éliminant la
direction d'épaisseur non observée. Cette réduction peut produire une
interaction spatiale effective. La longueur `ell` est donc appelée
**portée d'interaction apparente ou structurelle du modèle réduit**, et non
longueur interne intrinsèque du 316L.

Les résultats F1 actuels soutiennent un effet morphologique propre à `ell` :
les couples à `Achi = Hchi * ell**2` constant ne sont pas équivalents.
Cependant, `Hchi` et `ell` ne sont pas encore identifiés séparément et aucun
transfert sans recalage n'a été réalisé. Une échelle obtenue par EBSD sera
traitée comme une **échelle microstructurale indépendante servant de prior ou
d'hypothèse de fermeture**, jamais comme une mesure directe du paramètre
micromorphique.

### Règles de cette phase

- [ ] Pré-enregistrer toute campagne dans `validation/` avant son exécution.
- [ ] Conserver les seuils après calcul et documenter les résultats négatifs.
- [ ] Mettre à jour le registre de preuves et la matrice des claims dans le
  même commit que tout nouveau résultat.
- [ ] Étiqueter chaque grandeur comme mesurée, calculée ou supposée.
- [ ] Ne jamais comparer PEEQ ou une contrainte locale à la DIC comme s'il
  s'agissait de la même observable.
- [x] Ne lancer aucune nouvelle identification micromorphique couplée avant la
  caractérisation DIC et l'opérateur d'observation honnête. **Condition levée le
  2026-07-30** : V2 et V3 sont terminés, la porte est ouverte sous les trois
  conditions de la section « Prochaine action » en tête de fichier.

### Lot V0 — Inventaire expérimental, bloquant

Livrable autonome :
`docs/reference/experimental_data_inventory.md`.

- [x] Inventorier tous les pas DIC disponibles et distinguer images brutes,
  déplacements préparés et cartes dérivées.
- [x] Vérifier l'existence d'une branche de déchargement et du signal de
  charge synchronisé `F(t)`.
- [x] Documenter épaisseur, largeur utile, résolution, ROI et méthode de
  mesure.
- [x] Rechercher une paire statique ou une translation rigide connue.
- [x] Documenter la couverture, le pas et le recalage EBSD/DIC.
- [x] Reconstituer tous les paramètres DISFlow, notamment l'epsilon
  Charbonnier actuellement absent.
- [x] Répondre à chaque ligne par une valeur traçable ou `not available`.
- [x] Ajouter une note de décision sur l'acquisition éventuelle du
  déchargement.

**Résultat V0 révisé au 2026-07-29 :**

- une séquence externe de 42 TIFF bruts `000294`--`000335`, issue de l'essai
  de Qi Hu, est désormais accessible sous `essais/9_numerical/DIC_images` ;
  le crop `rows[400:4000], columns[1211:4311]` correspond exactement au
  support `3600×3100` des champs préparés ;
- le mapping `000294=référence`, `000295..000334=pas 1..40`,
  `000335=répétition finale` est fortement soutenu par le compte d'images et
  le recalage, mais reste provisoire sans journal d'acquisition ;
- aucune branche de déchargement ni série de force synchronisée n'est encore
  accessible ;
- l'article rapporte `t=2 mm`, un ROI initial `7×10 mm²`, un crop
  `6,624×5,704 mm²` et `1,84 µm/pixel`, mais pas la largeur utile ni la méthode
  de mesure de l'épaisseur ;
- les paramètres publiés sont `alpha=100`, `delta=1`, `gamma=0`,
  `epsilon=0,002`, 30 itérations ; la version OpenCV, le preset, la pyramide
  et les patches restent absents ; OpenCV 4.14 permet d'appliquer l'epsilon
  Charbonnier rapporté, mais cela ne résout pas l'absence de version et de
  configuration historique complète ;
- `essais/CP_dataset.h5` contient orientations et facteur de Schmid déclarés
  co-enregistrés sur `3600×3100`, mais le pas EBSD natif et la méthode de
  recalage sont absents ; 60 valeurs hors domaine et six pixels nuls doivent
  être masqués ou expliqués avant analyse ;
- décision : acquérir un cycle décharge/recharge si le montage peut être
  récupéré ; KD-064 reste bloqué jusque-là.

La caractérisation V2.1/V2.2 est pré-enregistrée dans
`validation/dic_measurement_chain_preregistration.md`. Elle distinguera
explicitement la chaîne historique non reproductible bit à bit de
l'implémentation DISFlow de reproduction dont tous les paramètres seront
enregistrés.

### Lot V1 — Contrôles mécaniques gratuits

#### Équilibre de section

Le contrôle naïf

`N_y(y) = t * integral sigma_yy(x,y) dx = constante`

est **interdit sur une partition intérieure telle que P43**. En intégrant
l'équilibre local :

`d/dy integral sigma_yy dx = -(sigma_xy(x_R,y)-sigma_xy(x_L,y))`.

La constance n'est attendue que sur une section couvrant la largeur physique
complète avec bords latéraux sans traction de cisaillement. Deux voies sont
autorisées :

- [ ] exécuter le contrôle sur un domaine couvrant réellement toute la
  section utile ; ou
- [x] inclure explicitement les flux de cisaillement des bords artificiels
  dans le résidu intégré.

Le diagnostic doit rapporter séparément la résultante, le terme de bord et le
résidu d'équilibre. L'épaisseur s'élimine dans la dispersion relative mais est
obligatoire pour comparer à une force mesurée.

**Résultat V1 au 2026-07-27 :** la commande
`diagnose-section-equilibrium` charge les champs `S` après contrôle
d'empreinte et analyse le domaine paddé ainsi que le cœur. Sur les quatre
campagnes P43 archivées, la dispersion descriptive de la résultante vaut
`2,54–5,45 %`. Le flux de cisaillement latéral ferme `68,5–70,9 %` du
déséquilibre naïf entre sections ; le résidu RMS restant vaut
`1,53e-4–2,27e-4` de la résultante moyenne. Ce résultat constitue une base
numérique sans seuil d'acceptation, car les contraintes de bord sont
approchées aux centres des cellules. Voir
`validation/section_equilibrium_p0043_preregistration.md` et
`validation/section_equilibrium_p0043_results.md`.

La validation contre une force physique reste bloquée : aucune série de
charge synchronisée et aucune largeur utile confirmée ne sont disponibles.

#### Calibration de force, conditionnelle aux données

- [ ] Utiliser l'épaisseur mesurée, jamais ajustée.
- [ ] Calibrer un seul facteur `lambda` commun à `sigma_y` et `K` sur un seul
  point de charge, en relançant la mécanique.
- [ ] Geler `lambda`, puis comparer `F_hat(t)` à toute la courbe `F(t)`.
- [ ] Interpréter séparément erreur de niveau et erreur de courbure.
- [ ] Ne jamais calibrer sur toute la courbe puis la présenter comme
  validation.

### Lot V2 — Caractérisation prioritaire de la chaîne DIC

Ce lot est le plus rentable scientifiquement et précède toute nouvelle
interprétation de `ell`.

#### V2.1 Test nul

- [x] Corréler deux images du même état ou une translation rigide connue avec
  les paramètres de production exacts.
- [x] Rapporter `sigma_u` en pixel, RMS de l'EVM parasite et longueur
  d'autocorrélation radiale.
- [x] Comparer le RMS parasite au RMS du champ DIC étudié sans modifier les
  seuils après observation.

#### V2.2 Fonction de transfert

- [x] Déformer synthétiquement l'image de référence par un balayage sinusoïdal
  et mesurer la modulation en fonction de la longueur d'onde.
- [x] Imposer des bandes de largeur 4, 8, 16 et 32 pixels et mesurer leur
  largeur reconstruite.
- [x] Rapporter résolution effective, biais d'amplitude et fidélité de largeur
  en pixels et micromètres.
- [x] Distinguer la fonction de transfert algorithmique synthétique des
  artefacts expérimentaux d'éclairage, de speckle et de mouvement hors plan.

**Résultat V2.1/V2.2 corrigé au 2026-07-29 :**

- une commande reproductible `characterise-dic-measurement-chain` applique
  les paramètres rapportés dans OpenCV 4.14 et sérialise les paramètres
  demandés et relus ;
- la première exécution utilisait `finest_scale=1`, donc s'arrêtait avant
  l'échelle native. Ce choix invalide toute conclusion métrologique sur les
  bandes de `4–32 px` ; V1 est conservée uniquement comme provenance ;
- V2 impose `finest_scale=0`. La paire finale candidate donne alors un RMS EVM
  parasite de `1,363e-4`, soit `4,52 %` du RMS EVM DIC final ; le seuil
  pré-enregistré classe encore cette
  amplitude comme faible ;
- cette paire ne constitue pas encore un bruit blanc certifié : le flot est
  cohérent sur `38,2 px` (`70,3 µm`) et le journal d'acquisition manque ;
- le MTF-50 sinusoïdal se situe vers `49 px`, tandis qu'une bande intégrée
  de `16 px` est reconstruite à `12–13 px` ; ces deux essais mesurent des
  contenus spectraux différents et doivent rester présentés ensemble ;
- la bande de `4 px` est résolue à `4 px` avec environ `83 %` du pic ; les
  bandes de `8`, `16` et `32 px` sont reconstruites à `7`, `12–13` et `28 px` ;
- la comparaison V1/V2 démontre que l'échelle finale est un paramètre
  métrologique majeur : elle déplace le MTF-50 d'environ `127` à `49 px` ;
- conclusion : l'opérateur de mesure est spatialement non neutre. V3 devient
  prioritaire avant toute reprise des balayages micromorphiques.

Artefacts :
`validation/dic_measurement_chain_results.md`,
`validation/reference_data/dic_measurement_chain_v2/` et
`validation/figures/dic_measurement_chain_v2/`. V1 reste archivée et
explicitement invalidée.

Validation de la correction V2 : Ruff et mypy verts ; `323` tests verts et
`22` tests MFront ignorés faute de bibliothèque déclarée ; Sphinx HTML,
linkcheck et PDF stricts verts. Le changement de valeur par défaut imposant
l'échelle native est publié au commit `339f76a` ; les résultats V2 et leur
consolidation documentaire appartiennent au commit final suivant.

`finest_scale=0` est désormais obligatoire. V2 a invalidé V1 ; V4 reproduit
les mêmes valeurs corrigées et constitue maintenant l'artefact public complet.

Extension visuelle V4 au 2026-07-29 :

- [x] afficher l'EVM exacte imposée et l'EVM récupérée par DISFlow pour les
  bandes de `4`, `8`, `16` et `32 px` ;
- [x] tracer sur chaque carte une coupe normale à la bande ;
- [x] superposer sur cette coupe le profil gaussien réellement imposé, le
  profil EVM récupéré et un créneau de lecture de même FWHM ;
- [x] conserver les amplitudes physiques d'EVM et afficher les coordonnées en
  pixels et micromètres ;
- [x] documenter qu'une largeur FWHM correcte ne garantit ni l'amplitude ni
  la morphologie.

Artefacts V4 :
`validation/reference_data/dic_measurement_chain_v4/`,
`validation/figures/dic_measurement_chain_v4/synthetic_band_evm_sections.png`
et `docs/explanation/dic_synthetic_measurement_tests.md`.

Validation V4 : Ruff vert sur `src`, `tests` et `scripts` ; mypy vert sur les
`53` fichiers source ; `346` tests verts avec MGIS/MFront réel et aucun skip ;
Sphinx HTML, linkcheck et PDF stricts verts. Le Ruff global reste pollué par
les scripts historiques nouvellement ajoutés sous `dic_analysis/`, hors de ce
lot et non modifiés ici.

Sensibilité epsilon sur la bande de `32 px` :

- [x] pré-enregistrement grossier puis raffinement explicite de la transition
  entre `0,002` et `0,02` ;
- [x] balayage `0,0002`, `0,002`, `0,004`, `0,006`, `0,01`, `0,02`, `0,2`,
  `2`, avec tous les autres réglages fixes et `finest_scale=0` ;
- [x] à `epsilon=0,01`, CV longitudinal réduit de `0,070` à `0,030` et pic
  presque exact (`0,958`), mais largeur réduite de `28` à `26 px` ;
- [x] à `epsilon=0,02`, CV réduit à `0,011`, mais largeur portée à `39 px` et
  pic réduit à `0,728` ;
- [x] conclusion : epsilon modifie fortement le défaut visible, mais la
  disparition de l'ondulation aux grandes valeurs est une
  sur-régularisation qui déplace l'erreur vers la largeur et l'amplitude ;
  aucune nouvelle valeur de production n'est sélectionnée.

Artefacts : `validation/dic_epsilon_band32_preregistration.md`,
`validation/dic_epsilon_band32_results.md` et
`validation/reference_data/dic_epsilon_band32_v2/`.
Validation : Ruff et mypy verts sur le périmètre ; `9` tests ciblés verts ;
Sphinx HTML, linkcheck et PDF stricts verts.

#### V2.3 Sensibilité aux paramètres DISFlow

- [ ] Répéter uniquement le diagnostic spatial output-only avec une variation
  pré-enregistrée de `alpha` DISFlow et de l'epsilon Charbonnier.
- [ ] Ne pas relancer à ce stade une identification micromorphique couplée.
- [ ] Rapporter le déplacement de la longueur diagnostique et déterminer si
  elle suit le lissage de mesure.

#### V2.4 Qualité locale et incertitudes

- [x] Construire un résidu photométrique local et le comparer à la carte
  d'erreur FEM–DIC.
- [x] Produire métriques brutes et métriques masquées/pondérées sans supprimer
  la première.
- [x] Séparer une propagation légère jusqu'aux métriques EVM d'une propagation
  complète relançant corrélation, identification des cartes et FEM.
- [x] Présenter l'incertitude de PEEQ comme incertitude d'une sortie du modèle,
  jamais comme incertitude expérimentale directe.

**Résultat KD-023 au 2026-07-29 :**

- campagne pré-enregistrée puis exécutée sur les quatre rejeux V3 P43 du
  profil `legacy_script_2021`, sans mécanique ni identification ;
- résidu direct
  `cell_average(abs(I40(x + u_DIC(x)) - I0(x)))`, bilinéaire, sans correction
  d'intensité et sans le `mask.png` historique absent ;
- support géométrique valide sur 100 % du coeur ; seuil q90 de `20,75`
  niveaux de gris et sensibilité conservant `90,14 %` des éléments ;
- corrélations résidu/erreur EVM négligeables : Pearson de `-0,025` à `0,023`
  et Spearman de `-0,019` à `0,009` ;
- retirer le pire décile change L2 d'au plus environ `1,1 %`, la corrélation
  FEM/DIC de moins de `0,0015` et ne change pas le classement ;
- conclusion négative : ce proxy photométrique local n'explique pas l'erreur
  structurée restante et ne justifie aucun masque pour l'identification.
  L'incertitude propagée reste une tâche distincte.

Artefacts :
`validation/reference_data/dic_photometric_quality_p0043_v1/`,
`validation/figures/dic_photometric_quality_p0043_v1/` et
`validation/dic_photometric_quality_p0043_results.md`.
Validation : Ruff et mypy verts ; `383` tests avec MGIS/MFront réel ;
`7` tests `measurement` sans skip ; Sphinx HTML, linkcheck et PDF stricts
verts.

**Résultat KD-024 au 2026-07-29 :**

- propagation légère pré-enregistrée sur les quatre rejeux V3 P43, 256
  tirages, graine `20260729`, sans relancer la mécanique ;
- le résidu DISFlow mesuré entre `000334` et `000335` est recentré puis
  extrait par fenêtres contiguës de la taille du support résolu P43, avec
  signe aléatoire ;
- un premier pilote périodique a été rejeté et documenté : la jonction de
  bords non concordants créait une ligne EVM artificielle. L'amendement
  méthodologique a été figé avant le rejeu accepté ;
- `alpha=4` reste premier sur RMSE, L2, corrélation et erreur d'aire active
  q90 pour 100 % des tirages ; `alpha=1` reste premier sur l'IoU q90 absolue
  et le cas local sur l'IoU top-10 % relative ;
- conclusion : l'incertitude structurée mesurée ne change pas le conflit
  entre familles d'objectifs et ne permet toujours pas de choisir un unique
  couplage ;
- les intervalles sont des sensibilités de substitution, pas des intervalles
  de confiance. `PEEQ` porte explicitement le statut
  `not_propagated_requires_mechanical_rerun`.

Artefacts :
`validation/dic_uncertainty_propagation_p0043_preregistration.md`,
`validation/dic_uncertainty_propagation_p0043_amendment.md`,
`validation/dic_uncertainty_propagation_p0043_results.md`,
`validation/reference_data/dic_uncertainty_propagation_p0043_v1/` et
`validation/figures/dic_uncertainty_propagation_p0043_v1/`.
Validation finale : Ruff global et mypy sur `64` fichiers sources verts ;
`386` tests avec MGIS/MFront réel ; `7` tests `measurement`, aucun skip ;
Sphinx HTML, linkcheck et PDF stricts verts ; registre de preuves
`E-DIC-006` vérifié contre le JSON primaire.

### Lot V3 — Opérateur d'observation symétrique

- [x] Ajouter un mode `synthetic_disflow` à l'opérateur d'observation :
  déformer l'image de référence par `U_FEM`, relancer DISFlow avec les
  paramètres de production, puis reconstruire l'EVM.
- [x] Conserver le mode actuel pour la non-régression et enregistrer le mode,
  les paramètres et les empreintes d'images dans le cache.
- [x] Recalculer la baseline locale et les campagnes couplées archivées, sans
  nouveau balayage de paramètres.
- [x] Rapporter toutes les métriques avant/après, notamment l'aire active au
  seuil DIC q90.
- [x] Documenter que cet opérateur reproduit le transfert algorithmique mais
  pas nécessairement les défauts expérimentaux complets.

**Résultat V3 au 2026-07-29 :**

- les sources DIC historiques ont été archivées sans modification sous
  `references/legacy_dic/` ; leur source fixe `finest_scale=0`,
  `patch_size=4`, `patch_stride=1`, les paramètres variationnels
  `100/1/0/0,002` et 30 itérations, mais laisse le preset, les itérations DIS,
  la normalisation de moyenne et la propagation spatiale aux valeurs d'usine ;
- deux profils immuables sont disponibles :
  `legacy_script_2021`, primaire par provenance, et `declared_medium_v4`,
  sensibilité entièrement explicite. Sous OpenCV 4.14, les valeurs d'usine
  relues pour le premier sont 16 itérations DIS, normalisation et propagation
  activées ;
- le masque historique reste absent. Conformément à la décision utilisateur,
  cela ne bloque pas V3 : un masque booléen tout-valide, déterministe et
  empreinté est déclaré, sans prétendre reproduire le masque historique ;
- la convention a été déterminée sur les champs réels :
  `U_40=flow[...,0]` est le déplacement de colonne, `V_40=flow[...,1]` celui
  de ligne ; le repère canonique associe ligne à `x/ux` et colonne à `y/uy`,
  sans transposition spatiale cachée ;
- le warp nominal inverse maintenant exactement la carte directe par point
  fixe en `float64`. La FWHM est sous-pixel et distingue maximum et barycentre.
  L'ancienne approximation et l'ancienne FWHM entière restent uniquement
  pour la non-régression ;
- sur la bande horizontale de 32 px, V4 corrigée donne `25,31 px` et le
  profil legacy-source `18,33 px`. Le warp corrigé modifie fortement cette
  métrologie alors qu'il déplace le MTF-50 de moins d'un pixel ;
- huit rejeux P43 ont été effectués sans relancer la mécanique :
  `alpha=0,1,2,4` avec les deux profils. Pour le profil principal, la baseline
  locale passe de `L2=0,952` à `0,486` et de `r=0,379` à `0,604` ;
- après observation, `alpha=4` minimise L2 (`0,292`) et maximise la
  corrélation (`0,664`), tandis que `alpha=1` maximise l'IoU absolue q90
  (`0,323`). Le classement dépend donc de l'observable et aucune valeur
  unique de couplage n'est sélectionnée ;
- la PEEQ n'a pas changé : sa redistribution reste une preuve mécanique
  séparée, jamais une PEEQ expérimentale ;
- décision : **ne pas reprendre l'identification `Hchi,ell` sur l'ancienne
  surface d'objectif**. Une nouvelle pré-inscription utilisant V3 et séparant
  amplitude/localisation est obligatoire.

Artefacts principaux :
`validation/dic_legacy_profile_comparison_results.md`,
`validation/dic_symmetric_observation_p0043_results.md`,
`validation/reference_data/dic_legacy_profile_comparison_v1/`,
`validation/reference_data/dic_symmetric_observation_p0043_v1/` et
`docs/explanation/current_evidence.md`.

Validation finale A0--A8 : Ruff vert globalement ; mypy vert sur 61 fichiers
source ; `378` tests verts avec MGIS/MFront réel ; job ciblé measurement
`2 passed, 376 deselected`, aucun test OpenCV ignoré ; Sphinx HTML et
linkcheck stricts verts ; PDF LuaLaTeX strict compilé. La CI contient
désormais un job OpenCV dédié qui échoue si un test measurement est sauté.

### Lot V4 — Valeur informative réelle des cartes

- [x] Baseline homogène : mêmes conditions de bord, `sigma_y` et `K`
  uniformes aux valeurs macroscopiques.
- [x] Contrôle permuté : transformer conjointement `sigma_y` et `K` de façon à
  préserver leurs distributions et leur dépendance mutuelle tout en détruisant
  leur correspondance spatiale.
- [x] Pré-enregistrer transformation, traitement des bords et métriques.
- [x] Quantifier le gain dû aux seules conditions de bord, puis l'information
  spatiale ajoutée par les cartes.
- [ ] Auditer avec le laboratoire partenaire la résolution, les orientations,
  la morphologie d'épaisseur et l'identification du calcul CPFEM comparé.

**Résultat V4 au 2026-07-27 :** les deux contrôles P43 convergent sans
cutback. Le contrôle homogène nominal (`sigma_y=124 MPa`, `K=380 MPa`) obtient
la meilleure erreur globale (`L2=0,351`, corrélation `0,420`) mais ne prédit
aucun pixel au-dessus du seuil DIC q90 : il efface les bandes. La translation
conjointe des cartes de `(600,500)` pixels fait chuter la corrélation de
`0,379` à `0,140` et l'IoU top-10 % de `0,207` à `0,113`. La position
originale des cartes contient donc une information réelle de localisation,
mais la baseline homogène démontre que L2 et corrélation favorisent fortement
un champ de fond lisse. Voir
`validation/material_map_controls_p0043_preregistration.md` et
`validation/material_map_controls_p0043_results.md`.

### Lot V5 — Échelle microstructurale indépendante

Ce lot commence seulement après V2 et V3.

- [x] Pré-enregistrer la statistique EBSD avant de lire sa valeur.
- [x] Utiliser comme analyse principale la longueur de décroissance
  exponentielle d'un champ orientationnel mécaniquement motivé ; rapporter le
  rayon RMS comme contrôle.
- [x] Exploiter le facteur de Schmid maximal pixelisé, archivé sous forme de
  moyenne par grain, comme proxy
  clairement étiqueté, sans prétendre qu'il représente la contrainte locale
  multiaxiale.
- [x] Traiter la reconstruction des grains et le choix des macles comme une
  analyse secondaire ; ils ne doivent pas modifier arbitrairement le champ
  orientationnel pixelisé.
- [x] Examiner l'anisotropie avant toute moyenne radiale et estimer
  l'incertitude par bootstrap spatial.
- [ ] Définir `xi_EBSD` comme échelle mesurée et pré-enregistrer l'hypothèse
  `ell = c * xi_EBSD`, avec `c=1` comme hypothèse principale et une bande de
  sensibilité annoncée à l'avance.
- [ ] Ne jamais appeler `xi_EBSD` une mesure directe de `ell`.
- [ ] Une fois cette hypothèse figée, balayer uniquement `Hchi` et publier
  aussi `ell/2`, `ell` et `2*ell` comme sensibilité ; ne pas réajuster `ell`
  si les critères échouent.

**Résultat V5 indépendant au 2026-07-27 :** le champ Schmid moyen par grain
donne une décroissance radiale de `179,38 µm`, avec une médiane spatiale
bootstrap de `108,57 µm` (`IC 95 % [90,92 ; 122,38] µm`). Les directions
diffèrent nettement (`132,93 µm` selon x, `212,31 µm` selon y ; rapport
`1,60`). Le contrôle par second moment est beaucoup plus long (`311,73 µm`) :
la valeur dépend donc fortement de la définition, exactement comme anticipé.
Ces nombres constituent une **échelle structurale EBSD/Schmid indépendante**,
pas une mesure directe de `ell`. Aucun calcul micromorphique n'a été lancé.
Voir `validation/ell_ebsd_definition_preregistration.md` et
`validation/ell_ebsd_structural_length_results.md`.

### Lot V6 — Histoire temporelle et validation conditionnelle

- [x] Imposer les déplacements de bord mesurés à chaque pas disponible au lieu
  d'une rampe proportionnelle vers l'état final. **Débloqué le 2026-07-30** :
  l'histoire mesurée de 40 états passe intégralement après correction du
  prédicteur élastique (`68,1 min`, 65 incréments convergés, 3 cutbacks).
- [x] Comparer accumulation incrémentale DISFlow et corrélation directe
  référence-vers-état courant pour quantifier la dérive.
- [ ] Identifier les cartes sur les pas 1–20 et évaluer les pas 21–40 avec
  cartes gelées et état interne propagé.
- [ ] Nommer ce test `conditional temporal prediction` tant que les conditions
  de bord futures restent mesurées.
- [ ] Cartographier les zones candidates de déchargement/non-proportionnalité,
  mais ne pas interpréter une baisse d'EVM comme preuve de Bauschinger.
- [ ] N'introduire Armstrong–Frederick ou Chaboche que si un véritable
  renversement de charge est disponible.
- [ ] Limiter l'indistinguabilité isotrope/cinématique au cas uniaxial
  proportionnel monotone ; ne pas la généraliser aux chemins locaux
  multiaxiaux.

#### V6 — état au 2026-07-30 : **résolu**

Lecture rapide, avant le journal des tentatives qui suit.

1. **L'histoire mesurée s'exécute intégralement.** 40 états, 65 incréments
   convergés sur 68, 3 cutbacks, `68,1 min`.
2. **La cause du blocage était logicielle**, pas mécanique : le prédicteur
   élastique de la branche histoire était résolu sur un buffer CSR déjà écrasé
   par la tangente élastoplastique. Le chemin proportionnel n'y était pas
   exposé, d'où l'asymétrie qui a fait perdre des semaines.
3. **Dépendance au trajet** : `15,8 %` sur PEEQ à point final identique,
   concentrée dans les bandes (rapport `13,11`), contre `0,20 %` de
   discrétisation. Ce n'est pas du bruit accumulé.
4. **Mais indiscernable face à la DIC** sous observation symétrique, sur les
   quatre métriques et les deux profils. Résultat d'identifiabilité.
5. **Filtrage modal à 3 modes** : retire `5,3×` sous le bruit, supprime tous
   les cutbacks et divise par deux le travail Newton, pour `1,63 %` sur PEEQ.
   Gain numérique, pas gain de fidélité.

Ce qui reste ouvert dans V6 : le test de prédiction temporelle conditionnelle
(identification sur les pas 1–20, évaluation 21–40 à cartes gelées) est
désormais **exécutable** et n'a pas été lancé. Les cases correspondantes de la
liste ci-dessus restent non cochées.

Documents : `dic_multistep_p0043_newton_instrumentation_results.md`,
`dic_multistep_p0043_predictor_fix_results.md`,
`dic_multistep_p0043_path_dependence_results.md`,
`dic_multistep_p0043_observed_path_comparison_results.md`,
`dic_multistep_p0043_modal_boundary_filter_results.md`.

#### Résultats du 2026-07-30, en détail

**BLOCAGE MULTI-PAS RÉSOLU le 2026-07-30.** L'histoire mesurée de 40 états
s'exécute intégralement (`completed_local_measured_boundary_history`). Les
quatre critères pré-enregistrés passent : aucun résultat archivé ne bouge, la
déformation d'essai à l'incrément 4 itération 1 tombe de `1,855e-02` à
`5,440e-04`, la signature d'opérateur gelé disparaît, et la transition état 3
vers état 4 est franchie. Bilan : `68,1 min`, 65 incréments convergés sur 68,
3 cutbacks, 469 itérations Newton, `max|D_ep|/max|C_el| = 1,000000`, zéro
diagonale non positive. Détails dans
`validation/dic_multistep_p0043_predictor_fix_results.md`.

Ce que cela **n'établit pas** : que les champs intérieurs reconstruits ont un
sens physique. Cette question reste régie par l'asymétrie de l'opérateur
d'observation et les résultats de bruit DIC déjà consignés.

**Dépendance au trajet mesurée le 2026-07-30, sur PEEQ au dernier état.** Les
deux calculs finissent sur une condition aux limites bit-à-bit identique, donc
la différence intérieure est de la dépendance au trajet et non une différence
de ce qui est imposé.

- L2 relative sur le cœur `360×310` entre histoire mesurée et rampe
  proportionnelle à 40 incréments : **`15,82 %`**, bande pré-enregistrée
  « présente mais non dominante » ;
- le trajet mesuré accumule **plus** de plasticité, et l'excès croît avec le
  niveau : `+4,9 %` sur la moyenne, `+9,7 %` au p99, `+14,8 %` au maximum ;
- **contrôle de discrétisation** : 40 contre 20 incréments en proportionnel ne
  change PEEQ que de `0,20 %`, soit `78×` moins que l'effet de trajet. Le veto
  pré-enregistré exigeait un facteur `3` : il ne se déclenche pas, et la marge
  n'est pas serrée ;
- **ce n'est pas le rochet de bruit** : `15,82 %` vaut `4,4×` l'estimation de
  `3,6 %`, et surtout le rapport de structure de bande vaut **`13,11`**,
  c'est-à-dire que l'excès est treize fois plus fort dans les bandes qu'en
  dehors. Un rochet de bruit n'a aucune raison de préférer les bandes. La
  contribution du bruit n'est toutefois **pas soustraite** ;
- corrélation `0,987`, IoU top-10 % `0,863` : la morphologie est largement
  conservée mais `13,7 %` du décile supérieur change d'appartenance ;
- coût numérique : 469 itérations Newton et 3 cutbacks en mesuré, contre 225 et
  zéro en proportionnel pour le même point final.

**Conséquence pour les campagnes micromorphiques archivées**, qui utilisent
toutes le trajet proportionnel : un systématique d'environ `16 %` sur PEEQ, non
comptabilisé jusqu'ici, exactement là où les métriques de recouvrement de bande
sont évaluées. Cela **ne renverse pas** les classements archivés, séparés par
des marges bien plus larges (IoU top-10 % EF/DIC autour de `0,25–0,30` contre
`0,863` entre trajets). À porter comme systématique connu, pas comme correction
à appliquer.

Artefacts : `validation/dic_multistep_p0043_path_dependence_preregistration.md`
et `validation/dic_multistep_p0043_path_dependence_results.md`.

**Comparaison à la DIC : les deux trajets sont indiscernables.** Sous
observation symétrique au niveau image, piloter le modèle par l'histoire
incrémentale réellement mesurée ne rapproche **pas** la déformation totale
finale de la DIC par rapport à une rampe proportionnelle au même point final.

- profil primaire `legacy_script_2021`, écarts A−B : L2 relative `+0,01545`
  (marge `0,0202`), Pearson `+0,00023` (marge `0,0185`), IoU top-10 %
  `−0,00444` (marge `0,0189`), IoU q90 absolu `+0,01488` (marge `0,0217`).
  **Aucune métrique ne franchit sa marge** ;
- le profil de sensibilité `declared_medium_v4` donne les mêmes verdicts ; les
  deux profils ne se contredisent pas ;
- les marges viennent des intervalles de sensibilité au bruit DIC déjà mesurés
  dans `dic_uncertainty_propagation_p0043_results.md`, pas d'un choix ad hoc ;
- tendance faible mais cohérente sur les deux profils, **sous la marge** : le
  trajet mesuré est légèrement moins bon en amplitude et légèrement meilleur en
  recouvrement q90, avec une fraction active `0,152` contre `0,161` pour une
  référence DIC à `0,10`. C'est la trace observable du résultat PEEQ, le trajet
  mesuré concentre davantage la plasticité ;
- la vue **brute**, biaisée et conservée comme contrôle, donnerait le trajet
  mesuré *moins bon* en amplitude (`+0,02809`, au-delà de la marge). Ce n'est
  pas la conclusion enregistrée ;
- l'attente pré-enregistrée était que le trajet mesuré soit au moins aussi
  proche. Elle n'est pas vérifiée.

**Pourquoi `15,82 %` d'écart PEEQ ne donnent aucun gain mesurable** : l'EVM
n'est pas le PEEQ et reste dominée par la cinématique imposée, identique aux
deux trajets par construction ; et le MTF-50 de la chaîne à `49 px` lisse
précisément les filaments étroits où les trajets diffèrent. Les EVM observées
diffèrent de `1 %` au maximum et `0,08 %` en moyenne.

**Conséquence d'identifiabilité** : l'histoire mesurée modifie de `15,8 %` une
variable interne non observable tout en modifiant l'observable moins que la
sensibilité au bruit DIC. Elle ne peut donc être ni validée ni réfutée par cet
observable. Discriminer les trajets exigerait un observable sensible à la
plasticité cumulée, que la chaîne de mesure actuelle ne fournit pas. La rampe
proportionnelle reste un choix défendable, à `2,2×` moins de travail Newton.

Artefacts :
`validation/dic_multistep_p0043_observed_path_comparison_preregistration.md`
et `validation/dic_multistep_p0043_observed_path_comparison_results.md`.

**Filtrage modal du bord, 2026-07-30.** Le bord mesuré est tronqué à 3 modes de
son écart à la rampe droite vers l'endpoint. Origine, endpoint et intérieur
restent bit-à-bit identiques ; le Dirichlet reste dur et exact, seule la donnée
imposée change.

- contenu retiré `0,00972 px` contre un bruit mesuré de `0,0511 px`, soit
  **`5,3×` sous le plancher de bruit** ; `99,989 %` de l'énergie de l'écart
  conservée ;
- la règle de rugosité de l'étape 0 sélectionne **indépendamment exactement
  3 modes** (`0,031 / 0,169 / 0,221` puis `0,561 / 0,512 / 0,660`). Le rang fixé
  d'avance et le critère mesuré coïncident sans ajustement ;
- **effet numérique majeur** : 40 incréments sur 40, **zéro cutback**, 245
  itérations Newton en `34,8 min`, contre 65/68, 3 cutbacks et 469 itérations en
  `68,1 min` sans filtre. C'est le comportement d'une rampe proportionnelle
  (225 itérations). La difficulté numérique de l'histoire mesurée était donc
  portée **entièrement par du contenu sous le plancher de bruit** ;
- PEEQ perturbé de `1,63 %`, soit dix fois moins que l'effet de trajet ;
- **la dépendance au trajet survit au filtre** : `15,36 %` contre la rampe, à
  comparer aux `15,82 %` sans filtre. Les `15,8 %` ne sont donc pas un artefact
  de bruit de bord mais une propriété du chemin de chargement ;
- accord DIC **indiscernable** sur les quatre métriques et les deux profils, le
  plus grand mouvement valant `+0,00346` pour une marge de `0,0202` ;
- le run filtré accumule `+0,64 %` de PEEQ moyen de plus que le non filtré alors
  que son pic **baisse** de `1,65 %`. **Expliqué le 2026-07-31** : c'est une
  **redistribution**, pas une hausse d'amplitude. Le filtre retire de la
  plastification marginale éparse aux bas niveaux (déciles 3 et 4 négatifs,
  `217` éléments plastifiés en moins), ajoute dans la gamme des bandes où les
  quatre déciles supérieurs portent `94,6 %` de l'excès, et rabote la queue
  extrême. Le confondant de sous-incréments, que j'avais affirmé « du bon
  ordre » sans le mesurer, n'en explique que `5,8 %` : cette affirmation était
  **fausse** et est corrigée.

Ce que le filtre apporte est **numérique, pas probant** : moitié moins de
travail Newton pour une perturbation de `1,63 %` d'une variable interne non
observable et aucun changement mesurable face à la DIC. Bon compromis de
production, à décrire comme tel et non comme un gain de fidélité.

Artefacts :
`validation/dic_multistep_p0043_modal_boundary_filter_preregistration.md` et
`validation/dic_multistep_p0043_modal_boundary_filter_results.md`.

**Pénalisation du bord disponible en option, 2026-07-30.**
`boundary_enforcement="penalty"` conserve les DDL prescrits dans le système avec
un ressort fini et expose `BOUNDARY_MISFIT`, l'écart nodal entre la mesure et
la valeur réellement imposée, avec la réaction devenue force du ressort
conjuguée. Erreur de cohérence en `1/k` vérifiée (`4,0e-9` mm à `k=1e8`,
`4,0e-11` à `1e10`), puis remontée à `2,9e-6` à `k=1e12` : limite de
conditionnement, d'où la nécessité d'un poids fini issu de la mesure.
L'élimination reste le défaut, aucune campagne archivée n'est affectée. Le choix
de `k` comme `1/sigma**2` normalisé reste à pré-enregistrer.

**Cause racine identifiée le 2026-07-30 par l'instrumentation Newton — le
blocage multi-pas est un défaut logiciel, pas numérique ni physique :**

- `FixedCSRAssembler.assemble` renvoie **le même objet CSR** en réécrivant
  `matrix.data` sur place, ce qui est documenté dans sa docstring. Dans
  `run_fem`, `KII_el` et `K_tang` viennent du même assembleur, donc
  `K_tang is KII_el`. Après le premier assemblage élastoplastique, `KII_el` ne
  contient plus l'opérateur élastique ;
- `solve_el` n'est utilisé qu'à deux endroits : **avant** la boucle pour le
  chemin proportionnel, où le buffer est encore élastique et le prédicteur
  n'est ensuite que multiplié par `dt` ; et **dans** la boucle pour le chemin
  à histoire mesurée. Seule la branche histoire rencontre le buffer corrompu ;
- cela explique la contradiction restée ouverte : la même partition converge en
  rampe proportionnelle et échoue en histoire mesurée. L'explication est
  logicielle, pas physique. L'argument sur les deux chemins traversant
  différemment l'activation plastique reste vrai sur le chargement, mais il
  n'est pas la cause de l'échec ;
- signature décisive dans la trace : les incréments 5 à 11 échouent dès
  l'itération 1 avec une déformation d'essai exactement proportionnelle à `dt`
  (`4423, 2211, 1106, 553, 276, 138, 69,1`). Un halving exact sur sept
  incréments est la signature d'un opérateur **gelé** : ces incréments
  n'atteignent jamais un assemblage de tangente ;
- la tangente constitutive est innocentée : le ratio `max|D_ep|/max|C_el|` vaut
  exactement `1,000000` à chaque itération ;
- **aucun résultat scientifique archivé n'est affecté** : toutes les campagnes
  archivées utilisent le chemin proportionnel, dont le prédicteur est calculé
  avant la corruption. Seules les exécutions à histoire mesurée le sont, et
  elles avaient toutes déjà échoué ;
- conséquence : la campagne de line search de `2 h 47`, les suppressions de
  frames et l'hypothèse de bruit de bord traitaient tous un symptôme ;
- le discriminateur pré-enregistré est consigné comme **inadéquat** : une
  diagonale strictement positive n'établit pas un opérateur bien posé, et
  l'opérateur inspecté n'était pas celui utilisé par le prédicteur.

Artefacts : `validation/dic_multistep_p0043_newton_instrumentation_preregistration.md`
et `validation/dic_multistep_p0043_newton_instrumentation_results.md`.

#### Journal des tentatives, conservé pour provenance

Les points ci-dessous datent d'avant la résolution. Ils restent exacts comme
compte rendu de ce qui a été essayé et écarté, mais leur conclusion
(« globalisation de Newton ») a été **remplacée** par le point 2 ci-dessus.

**État V6 au 2026-07-29 :**

- le pilote EF accepte désormais une histoire nodale transactionnelle de
  `N+1` états, interpole uniquement lors des cutbacks et conserve strictement
  le chemin proportionnel historique lorsqu'aucune histoire n'est fournie ;
- les 40 champs ont été reconstruits par corrélation directe de la même image
  de référence vers chaque état : aucune accumulation DISFlow n'est présente
  dans cette série, donc le test de dérive incrémentale ne s'applique pas à
  cette provenance ;
- l'état final OpenCV 4.14 diffère du champ préparé de `1,583 %` en norme
  vectorielle ; l'histoire est ancrée linéairement sur l'endpoint immuable,
  sans supprimer sa déviation au chemin proportionnel ;
- les états 31 et 32 contiennent un artefact EVM massif, cohérent avec la
  déclaration `CORRUPTED_FRAMES` du script historique. Une correction
  pré-enregistrée interpole les déplacements entre les états 30 et 33 :
  l'EVM incrémentale maximale passe de `5,459e-2` à `5,623e-3`, sans changer
  les autres états ni l'endpoint ;
- malgré cette réparation, le calcul local MFront échoue dès la transition
  état 3 → état 4 (`pseudo-time=0,10`) puis sous cutback jusqu'à
  `0,0750244`. L'échec précède donc les frames réparées et ne peut pas leur
  être attribué ;
- l'instrumentation du rejet montre des essais Newton non physiques :
  déformation ingénieur maximale `82,257` au premier échec et `58,011` au
  dernier, dans deux éléments voisins du bord supérieur. MFront rejette donc
  correctement un sursaut du Newton non amorti ; ce n'est pas une limite
  constitutive sur la petite déformation DIC imposée ;
- une tentative de line search résiduelle a empêché les premiers sursauts mais
  a été arrêtée après `2 h 47` : elle accumulait les itérations et cutbacks
  sans fournir un chemin de production acceptable ; aucun résultat partiel
  n'a été conservé ;
- un diagnostic indépendant de courbure temporelle place son maximum à l'état
  3 (`4,214e-4 mm` RMS). Son remplacement pré-enregistré par
  `u3=(u2+u4)/2` préserve bit-à-bit les autres états et l'endpoint, mais le
  calcul échoue encore sur la transition vers l'état 4 : 3 incréments
  convergés, 11 cutbacks, maximum de déformation rejetée `69,529` puis
  `90,230` ;
- le prédicteur `secant-corrected-elastic` n'extrapole que la correction de
  déplacement intérieur. Il ne change pas le protocole MGIS : toutes les
  variables internes repartent du dernier état engagé, sont intégrées pendant
  l'incrément, puis engagées une seule fois après convergence. Leur
  interpolation directe reste interdite ;
- un second contrôle pré-enregistré remplace uniquement l'état cible 4 par
  `(u3+u5)/2` et rétablit le prédicteur élastique. Il échoue exactement à la
  même limite (`t=0,0750244`, 3 incréments, 11 cutbacks), dans les mêmes
  éléments de bord. La suppression de frames supplémentaires est donc
  arrêtée : le verrou est la globalisation de Newton au voisinage de cet état,
  pas une frame DIC isolée ;
- un audit spatial et photométrique explicite des états 1–6 confirme que
  l'état 3 est réellement précoce (`4,79 %` de l'EVM RMS finale), mais ne
  trouve aucun outlier aux éléments 402245/402246. À l'état 4, le résidu
  non affine du bord ne représente que `0,273 %` de l'amplitude RMS, le
  résidu photométrique n'est pas maximal, et la déformation exacte aux points
  de Gauss fautifs vaut au plus `8,89e-5`. Les essais Newton rejetés de
  `58–82` sont au moins `3,64e5` fois plus grands que toute déformation
  mesurée sur ces éléments aux états 1–6 ;
- la convergence du champ final n'est pas contradictoire : le baseline suit
  une rampe proportionnelle droite vers l'endpoint, alors que l'histoire DIC
  suit un chemin différent. À l'état 4, la contraction affine transverse a
  atteint `7,29 %` de sa valeur finale, mais l'extension affine axiale
  seulement `1,09 %`. Les deux calculs ne traversent donc pas l'activation
  plastique hétérogène par le même chemin ;
- l'état constitutif est restauré après chaque tentative et aucun résultat
  mécanique partiel n'est présenté comme convergé. Le test multi-pas reste
  bloqué. La prochaine instrumentation doit porter sur la correction Newton
  des ddl libres avant l'essai constitutif rejeté (norme, incrément de
  déformation élémentaire, conditionnement de la tangente), pas sur une
  nouvelle suppression de frame. Aucun rejet automatique piloté par la
  convergence et aucun filtre de Kalman ne sont autorisés implicitement ;
- une hypothèse de bruit de bord a été pré-enregistrée puis **réfutée** le
  2026-07-30 par le diagnostic étape 0 de sous-espace de chargement. Le critère
  enregistré demandait `|z| >= 3` à l'état 4 : les scores mesurés valent `0,13`
  sur le coefficient de chargement et `1,66` sur la déformation affine. Le
  maximum sur les 40 états vaut `1,99`, sous le `~2,7` attendu de 39 tirages
  gaussiens : le chemin de bord est plus lisse que du bruit pur et ne contient
  aucun outlier. Le SNR de l'incrément à l'état 4 vaut `3,73`, au-dessus de la
  médiane `3,52`. La lecture antérieure à `3,5 sigma` provenait d'une comparaison
  d'incréments bruts à cinq voisins d'une série en tendance ; le
  second-différenciage la fait disparaître. L'instrumentation Newton différée
  est donc réinstaurée comme piste principale.

**Acquis conservés du diagnostic étape 0, 2026-07-30 :**

- bruit de mesure par état `0,047–0,051 px`, estimé par différences temporelles
  secondes sur 37 réalisations, en accord avec la borne `0,06283 px` obtenue par
  paire d'images répétées. Deux routes indépendantes convergent près de
  `0,05 px` ; l'autocorrélation lag-1 des différences secondes vaut `-0,561`
  contre `-2/3` théorique, ce qui valide l'estimateur ;
- **le bruit est affine à ~90 %** (fraction non affine médiane `9,63 %`). Il se
  comporte comme un tremblement global cohérent du bord, pas comme une
  décorrélation nodale. Cela explique le facteur `26` entre le sigma archivé et
  le résidu non affine archivé : la métrique de résidu retire par construction
  la bande qui porte le bruit ;
- conséquence Saint-Venant : la bande contre laquelle un padding protège est
  celle qui ne porte presque pas de bruit, et la bande qui porte le bruit est
  quasi uniforme et ne décroît pas vers l'intérieur. Un filtre spatial n'a rien
  à retirer et le padding n'est pas une défense ici ;
- le chargement de bord est **un seul mode lisse** à `99,91 %` de l'énergie,
  rugosité temporelle `0,0023` ; les modes 1 à 3 portent `99,999 %`. Une
  régularisation temporelle de faible dimension est donc l'architecture
  correcte si l'étape 1 est poursuivie ;
- SNR par incrément de déformation affine : médiane `3,52`, minimum `0,62`,
  5 incréments sur 40 sous l'unité (états 5, 8, 19, 24, 25). La régularisation
  temporelle garde une justification mesurée, mais **faible** : elle ne corrige
  pas l'échec, elle améliore 5 incréments et réduit d'environ `3,6 %` le biais
  d'accumulation plastique, contre `18 %` estimé avant mesure. L'étape 1 doit
  être re-pré-enregistrée sur cette base ou différée.

Artefacts : `validation/dic_multistep_p0043_audit.md`,
`validation/dic_multistep_p0043_preregistration.md`,
`validation/dic_multistep_p0043_endpoint_amendment.md`,
`validation/dic_multistep_p0043_corrupted_frames_amendment.md` et
`validation/dic_multistep_p0043_results.md`,
`validation/dic_multistep_p0043_state4_bridge_preregistration.md`,
`validation/dic_multistep_p0043_state_bridge_indexing_amendment.md` et
`validation/dic_multistep_p0043_blocked_state4_preregistration.md` et
`validation/dic_multistep_p0043_state_bridge_results.md`, puis
`validation/dic_multistep_p0043_boundary_outlier_analysis_plan.md` et
`validation/dic_multistep_p0043_boundary_outlier_results.md`, puis
`validation/dic_boundary_temporal_regularisation_preregistration.md` et
`validation/dic_boundary_loading_subspace_p0043_results.md`.

### Lot V7 — Test jumeau de la revendication data-driven

- [ ] Générer une vérité avec une loi différente de Ludwik, idéalement CPFEM
  ou Voce.
- [ ] Passer ses déplacements de surface par le bruit et la fonction de
  transfert mesurés en V2.
- [ ] Rejouer sans modification toute la chaîne d'identification des cartes et
  de reconstruction J2/Ludwik.
- [ ] Comparer le nuage de phase reconstruit à la vérité indépendante.
- [ ] Si Ludwik réapparaît quelle que soit la vérité, déclarer la revendication
  data-driven réfutée.
- [ ] Positionner explicitement le travail face à Data-Driven Identification
  et à la Virtual Fields Method avant toute revendication de supériorité.

### Ordre d'exécution et portes de décision

1. V0 inventorie et débloque les données.
2. V1 exécute uniquement les contrôles mécaniques mathématiquement admissibles.
3. V2 mesure la chaîne DIC.
4. V3 rétablit la symétrie de l'opérateur d'observation.
5. V4 mesure la contribution réelle des cartes et rend la comparaison CPFEM
   interprétable.
6. V5 teste une échelle microstructurale indépendante.
7. V6 teste l'histoire temporelle.
8. V7 décide si la revendication espace des phases survit à un jumeau
   numérique.

Une nouvelle campagne d'identification micromorphique n'est autorisée qu'après
V2 et V3. **Les deux sont terminés, la porte est donc ouverte depuis le
2026-07-30** ; voir « Prochaine action » en tête de fichier pour les trois
conditions attachées. Une revendication de longueur structurelle imposée exige V5. Une
revendication de longueur matérielle reste interdite sans transfert sur une
autre ROI, une autre résolution d'observation et idéalement un autre essai.

## 2. Sources de vérité

1. `ArticleSource/ArticleAdil.pdf`
2. Les données DIC et cartes de paramètres versionnées dans
   `data/raw/case_study`
3. Le manifeste de provenance et le profil de préparation produits par le
   présent projet
4. Les tests automatisés du présent projet
5. À titre de validation différée : les fichiers d'entrée Abaqus, ODB et
   scripts d'extraction ayant produit les résultats historiques

Toute contradiction entre l'article, les entrées Abaqus et le code doit être
documentée et résolue explicitement. Le comportement courant du code n'est pas
considéré comme une spécification par défaut.

### Priorité de développement

Le chemin critique est la reproduction du calcul **à partir des données DIC**.
Le dépôt doit permettre, depuis un clone neuf :

1. de récupérer et vérifier les données scientifiques brutes versionnées ;
2. de les convertir sans opération implicite vers le contrat canonique ;
3. de lancer ou reprendre les partitions indépendamment ;
4. de raccorder les champs globaux ;
5. de reconstruire les grandeurs de l'article depuis les déplacements ;
6. d'obtenir des manifestes et rapports contenant les paramètres, empreintes et
   versions du code.

La comparaison Abaqus reste souhaitable, mais elle n'est plus un prérequis au
développement ni à l'exécution de ce pipeline principal. Elle constitue une
campagne de validation externe ultérieure.

## 3. Objectif scientifique

Le projet ne cherche pas à reproduire Abaqus de manière générale.

Il doit fournir, pour le cas d'étude de l'article, un moteur de reconstruction
cinématique :

- mécaniquement admissible ;
- limité aux petites déformations et aux contraintes planes ;
- fondé sur un maillage CPS4 rectangulaire structuré ;
- piloté par les déplacements DIC prescrits aux frontières ;
- utilisant des descripteurs élastoplastiques effectifs identifiés à l'échelle
  du pixel ;
- capable de reconstruire l'organisation spatiale des bandes de localisation ;
- capable de traiter le domaine complet par sous-domaines avec recouvrement et
  raccordement ;
- reproductible depuis les quatre tableaux bruts versionnés, sans chemin
  personnel ni donnée cachée.

Les paramètres locaux sont des **descripteurs effectifs dépendant du chargement,
de la résolution DIC et des hypothèses constitutives**. Ils ne doivent pas être
présentés comme des propriétés intrinsèques des grains.

## 4. Périmètre supporté

### Inclus

- matériau 316L du cas d'étude ;
- élasticité homogène : `E = 205 GPa`, `nu = 0.30` ;
- plasticité J2/von Mises ;
- loi de Ludwik-Hollomon ;
- exposant `n = 0.245` pour le cas nominal ;
- cartes spatiales de limite d'élasticité et de coefficient d'écrouissage ;
- contrainte plane ;
- éléments quadrilatéraux bilinéaires CPS4, intégration 2×2 ;
- maillage régulier à un élément par pixel ;
- déplacements mesurés imposés sur les frontières des sous-domaines ;
- résolution sparse avec PyPardiso/MKL ;
- partitionnement sans recouvrement et avec padding ;
- raccordement des cœurs des partitions ;
- post-traitement DIC/EF commun à partir des déplacements ;
- comparaison avec les champs expérimentaux ;
- préparation traçable des données DIC historiques.

### Hors périmètre

- maillages non structurés ;
- éléments autres que CPS4 ;
- grandes transformations ;
- contact ;
- endommagement et rupture ;
- dynamique ;
- 3D ;
- plasticité cristalline ;
- chargements généraux sans rapport avec le cas d'étude ;
- solveur EF généraliste ou remplacement global d'Abaqus.

La comparaison avec Abaqus appartient au périmètre de validation, mais peut
être réalisée après la mise en service du calcul autonome depuis la DIC.

## 5. Échelle du problème de production

- ROI : `3600 × 3100` pixels
- Nombre d'éléments : environ `11,16 millions`
- Domaine physique : `6,624 × 5,704 mm²`
- Taille de pixel : `1,84 µm`
- Schémas étudiés dans l'article :
  - 25 partitions sans recouvrement ;
  - 25 partitions avec padding ;
  - 100 partitions avec padding ;
- Padding de production mentionné dans l'article : environ 150 éléments

Le solveur ne doit pas tenter de charger ou résoudre le domaine complet de
manière monolithique. Les entrées, sorties et raccordements doivent être conçus
pour fonctionner hors mémoire.

## 6. État initial vérifié

### Points positifs

- [x] Environnement `.venv` créé
- [x] NumPy, SciPy et Matplotlib installés
- [x] PyPardiso 0.4.7 et MKL 2026.1.0 installés
- [x] Backend réellement sélectionné :
  `pypardiso (MKL, multithreaded)`
- [x] Test biaxial homogène 20×20 exécuté avec succès
- [x] Erreur affichée sur la contrainte de von Mises : 0 %
- [x] Tangente cohérente vérifiée par différences finies
- [x] Erreur relative de tangente observée : `1e-10` à `7e-9`
- [x] Matrice élémentaire symétrique avec trois modes rigides
- [x] Cas hétérogène convergeant en quatre itérations de Newton par incrément
- [x] Équilibre global observé de l'ordre de `1e-14`

### Blocages et défauts initiaux

Une case cochée dans cette liste signifie que le défaut initial a été corrigé
et vérifié ; une case vide indique qu'il reste à traiter.

- [x] `test_config.py` est absent du projet livré
- [~] Les scripts de validation ne sont pas exécutables de manière autonome
- [x] Des chemins Windows absolus sont présents
- [x] La courbe étiquetée « FEM stress » remplace la contrainte EF directe par
      une reconstruction de Ludwik après plastification
- [x] Les quatre courbes scientifiques de l'article ne sont pas séparées
- [x] La table plastique par défaut utilise 50 points, contre 1000 points dans
      l'article
- [x] Les conventions d'axes DIC ne sont pas cohérentes dans tous les scripts
- [x] Les conventions cisaillement tensoriel/ingénieur ne sont pas garanties
- [x] Le seul test intégré n'asserte pas la valeur de PEEQ
- [x] Aucun moteur de partitionnement/raccordement n'existe
- [x] Aucun traitement hors mémoire du ROI complet n'existe
- [x] Aucun manifeste de dépendances ou verrouillage des versions n'existe
- [x] Aucun historique Git exploitable n'est présent dans le dossier
- [ ] Aucun seuil automatique de parité Abaqus n'est défini
- [x] Les quatre tableaux scientifiques du ROI sont absents du dépôt
- [~] Aucun pipeline autonome ne transforme les noms, unités et conventions
      historiques vers les quatre entrées canoniques
- [ ] La règle de complétion nodale `3600×3100 → 3601×3101` n'est pas encore
      ratifiée scientifiquement
- [ ] L'écart entre le facteur d'écrouissage `380 MPa` de l'article et
      `396 MPa` du générateur historique doit rester explicite et paramétrable

## 7. Grandeurs scientifiques à maintenir séparées

Le logiciel doit produire et nommer sans ambiguïté :

1. la courbe macroscopique mesurée ;
2. la contrainte reconstruite depuis la déformation DIC ;
3. la contrainte reconstruite depuis la déformation EF ;
4. la contrainte EF directe calculée depuis `S11`, `S22`, `S12`.

Les courbes 2 et 3 sont des contrôles de cohérence obtenus en réappliquant la loi
constitutive à une mesure de déformation. Elles ne constituent pas des
prédictions indépendantes de contrainte.

L'écart entre la contrainte EF directe et la courbe macroscopique mesurée doit
être conservé et analysé. Il ne doit pas être corrigé par le post-traitement.

## 8. Planification révisée

Durée prévisionnelle : **12 semaines pour une personne à temps plein**, avec
revue scientifique régulière.

### Phase prioritaire A — Dépôt autonome depuis la DIC

- [x] Versionner sous Git LFS les quatre tableaux bruts sans les modifier
- [x] Enregistrer forme, type, taille, rôle et SHA-256 de chaque tableau
- [x] Conserver les générateurs Abaqus reçus uniquement comme provenance
- [x] Ajouter `fem-inhouse prepare-case`
- [x] Vérifier les empreintes avant toute transformation
- [x] Convertir `V → u_x`, `U → u_y` et pixels → millimètres
- [x] Rendre le facteur macroscopique `K` explicite, avec `380 MPa` nominal et
      `396 MPa` historique
- [x] Détecter les neuf valeurs non finies et appliquer seulement une politique
      explicitement sélectionnée et enregistrée
- [x] Compléter la grille nodale selon une politique explicite et enregistrée
- [x] Écrire les quatre `.npy` canoniques et un manifeste reproductible
- [x] Ajouter un test d'intégration depuis des données brutes synthétiques
- [x] Ajouter un contrôle d'intégrité des données réelles, sans les charger
      entièrement en mémoire
- [x] Documenter une séquence unique `clone → install → prepare → partition →
      stitch → postprocess`
- [x] Exécuter un sous-domaine réel versionné depuis cette séquence

**Critère de sortie :** aucune donnée ou transformation scientifique nécessaire
au calcul principal ne se trouve hors du dépôt ou dans un chemin personnel.

### Phase prioritaire A.1 — Remplacement constitutif par MFront

- [x] Installer depuis les sources TFEL/MFront 5.1.0 et MGIS 3.1
- [x] Épingler les tags, commits, options CMake et procédure d'activation
- [x] Implémenter la loi J2/Ludwik sous l'hypothèse `PlaneStress`
- [x] Exposer les cartes locales `sy0`, `K` et `n` comme propriétés matériau
- [x] Compiler l'interface générique MFront de façon reproductible
- [x] Ajouter l'adaptateur Python MGIS avec conversions Kelvin/ingénieur
- [x] Gérer explicitement les états d'essai, `commit` et `revert`
- [x] Comparer et sauvegarder trois trajets au point matériel sur 200 incréments
- [x] Déclarer les seuils avant la comparaison et conserver tous les champs
- [x] Retenir la loi MFront analytique régularisée sans plafond de PEEQ ; garder
      les 1000 segments uniquement comme régression historique explicite
- [x] Brancher MFront derrière une sélection de backend dans la boucle Newton
- [x] Vérifier la tangente MFront dans les conventions d'assemblage CPS4
- [x] Comparer les deux backends sur le crop DIC réel `10×10`
- [x] Mesurer coût et mémoire sur une partition à la taille de l'article ;
      `510×460` éléments mesurés avec MFront en 650,08 s et 4 163 308 KiB RSS,
      sans construire la table Python
- [x] Basculer le backend par défaut vers MFront après parité du sous-domaine

**Critère de sortie :** le même sous-domaine DIC converge avec les deux
backends, les six champs sauvegardés respectent des seuils ratifiés, et aucun
état MFront d'une itération Newton rejetée n'est commis.

### Phase prioritaire A.2 — Tenseurs 3D complets en contraintes planes

- [x] Maintenir strictement le solveur, les inconnues, les éléments, Newton et
      le tangent condensé en 2D
- [x] Centraliser les conversions engineering, tensorielle et Kelvin
- [x] Reconstruire `ep33` par incompressibilité plastique J2 pour Python
- [x] Reconstruire `ee33` par élasticité isotrope en contraintes planes
- [x] Assembler `S_3D`, `E_3D`, `EE_3D`, `PE_3D` après convergence seulement
- [x] Préserver `S`, `E`, `PE`, `PEEQ` et toutes leurs conventions historiques
- [x] Identifier par métadonnées MGIS `AxialStrain`, `ElasticStrain` et
      `Stress`, sans supposer leurs offsets
- [x] Vérifier par essai matériel que le gradient Kelvin ne porte pas le
      `e33` natif de ce comportement
- [x] Conserver le `S33` MFront natif comme résidu de contraintes planes
- [x] Interdire le fallback analytique implicite MFront ; n'autoriser la
      complétion J2 qu'avec la capacité explicite
      `j2_isotropic_analytical`
- [x] Étendre `FEMResult`, les exports de partitions, le raccordement et le
      chargeur des résultats anciens
- [x] Séparer `EVM_HISTORICAL` de `EVM_RECONSTRUCTED_3D`
- [x] Tester traction, traction équibiaxiale, cisaillement, déchargement et
      chargement non proportionnel
- [x] Comparer Python/MFront sur le crop DIC réel `10×10`
- [x] Comparer les six champs historiques avec la campagne antérieure
- [x] Finaliser la documentation Sphinx et reconstruire HTML/PDF

**Preuve DIC 10×10 :**
`validation/reference_data/plane_stress_tensor_reconstruction_dic_10x10_v1`.
Les trois groupes de contrôles passent. Le maximum `|S33|` vaut `0` pour
Python et `1,046e-14 MPa` pour MFront ; le maximum
`|trace(epsilon_p)|` vaut respectivement `0` et `1,406e-19` ; le maximum de
la décomposition additive vaut `8,132e-20` et `1,355e-19`. La différence
maximale avec les sorties historiques est nulle pour Python et
`4,263e-14 MPa` pour MFront.

**Critère de sortie :** tout résultat FEM convergé expose les quatre tenseurs
symétriques `3×3` et le résidu `S33`, sans nouvelle résolution mécanique et
sans régression des sorties 2D.

### Phase prioritaire A.3 — Loi 3D condensée en contraintes planes

- [x] Ajouter les conversions Kelvin 3D à six composantes et vérifier l'ordre
      MGIS `[11,22,33,12,13,23]` par métadonnées et essais élémentaires
- [x] Généraliser le résidu à `[S33,S13,S23]` tout en conservant
      `S33_RESIDUAL_MPA` comme vue compatible
- [x] Introduire le protocole transactionnel commun
      `PlaneStressMaterialBatch`
- [x] Adapter les backends Python J2 et MFront `PlaneStress` au protocole
- [x] Compiler la même loi J2/Ludwik sous l'hypothèse `Tridimensional`
- [x] Résoudre localement `[epsilon33,gamma13,gamma23]` depuis le même état
      constitutif validé à chaque itération
- [x] Condenser la tangente 6×6 par complément de Schur sans inversion
      explicite
- [x] Ajouter les diagnostics de résidu au point de Gauss, itérations locales,
      échecs locaux et conditionnement de `Cbb`
- [x] Vérifier la tangente condensée par différences finies dans un état
      plastique éloigné du seuil
- [x] Tester l'échec local et l'absence de pollution de l'état validé
- [x] Comparer les deux chemins sur les trajets matériels et un maillage 4×4
- [x] Comparer et sauvegarder les deux chemins sur le crop DIC réel 10×10
- [x] Comparer temps complet, temps constitutif et pic RSS des trois backends
      sur le même crop DIC 100×100, trois processus frais par backend
- [x] Documenter l'architecture, ses limites et le contrat pour une future loi
      cristalline 3D

**Preuve DIC 10×10 :**
`validation/reference_data/mfront_3d_condensed_dic_10x10_v1`. Les deux chemins
convergent en 66 itérations Newton globales, sans cutback. L'écart maximal sur
la contrainte dans le plan vaut `4,804e-08 MPa`. Le backend condensé atteint
un résidu transverse maximal au point de Gauss de `2,705e-08 MPa` en quatre
itérations locales au plus, avec zéro échec et
`max(cond(Cbb)) = 1,896`.

**Preuve de performance DIC 100×100 :**
`validation/reference_data/plane_stress_backend_performance_100x100_v1`.
Les neuf calculs convergent sans cutback. Les médianes
temps mur / pic RSS sont `134,36 s / 248,96 MiB` pour Python,
`27,03 s / 269,65 MiB` pour MFront natif et
`83,43 s / 320,30 MiB` pour MFront 3D condensé. Les deux chemins MFront
diffèrent au maximum de `2,307e-07 MPa` sur la contrainte ; Python diffère au
maximum de `6,763e-02 MPa` et respecte tous les seuils déclarés du cas
d'étude.

**Critère de sortie :** le solveur global ne connaît plus la loi J2 ou les
détails MGIS ; les chemins J2 MFront natif et J2 3D condensé sont équivalents
aux tolérances numériques sur le cas DIC, et la substitution future d'une loi
3D petites déformations reste confinée à l'adaptateur constitutif.

### Phase prioritaire A.4 — Diagnostic de non-localité par Helmholtz

- [x] Ajouter un filtre scalaire de Helmholtz aux centres des éléments
      structurés, avec flux nul et résolution DCT orthonormale
- [x] Garantir que `ell=0` restitue une copie exacte, sans DCT ni projection
      élément-nœud-élément
- [x] Vérifier conservation de la moyenne, principe du maximum, décroissance
      de variance, anisotropie `hx != hy`, résidu et référence sparse directe
- [x] Reconstruire séparément EVM DIC et EVM FEM avec la chaîne commune
      `strain_from_displacement → plane_stress_equivalent_strain →
      cell_average`
- [x] Filtrer le domaine résolu complet avec padding et calculer les métriques
      uniquement sur le cœur issu des métadonnées
- [x] Signaler les longueurs dont le rapport padding/longueur est inférieur au
      seuil numérique configurable
- [x] Ajouter les erreurs de champ, recouvrements par quantile, seuils absolus
      DIC et métriques de diffusivité
- [x] Conserver PEEQ comme indicateur interne séparé, sans RMSE ou MAE
      d'amplitude contre EVM DIC
- [x] Ajouter les modes exploratoire et confirmatoire, avec seuils
      confirmatoires fournis en YAML ou JSON avant calcul
- [x] Ajouter `fem-inhouse diagnose-nonlocality`, les rapports atomiques,
      manifestes, champs et figures reproductibles
- [x] Exécuter le balayage `0–58,88 µm` sur la partition article 0 sauvegardée
      avec padding 150 pixels
- [x] Documenter la méthode et la campagne selon Diátaxis

**Preuve partition article :**
`validation/reference_data/nonlocality_helmholtz_article_p0000_v1`.
Le filtre porte sur les `510×460` éléments résolus et les métriques sur le
cœur `360×310`. À `58,88 µm`, RMSE et erreur L2 relative diminuent de
`49,45 %`, la corrélation passe de `-0,0292` à `0,0926` et l'IoU des 10 %
les plus élevés de `0,0503` à `0,1312`. La moyenne dérive au plus de
`8,674e-19`, le résidu relatif au plus de `5,575e-13`, et toutes les longueurs
respectent `padding/ell >= 4`.

**Interprétation :** hypothèse de largeur spatiale **partiellement soutenue**
sur cette partition exploratoire. Le meilleur point des critères principaux
est la borne supérieure du balayage et atténue fortement les pics ; la
corrélation reste faible. Aucune longueur interne matérielle n'est identifiée.
Une confirmation devra fixer la longueur sur une partition puis l'appliquer
sans ajustement à des partitions tenues à l'écart.

**Révision pré-enregistrée avant nouveau calcul :** la partition 0 est jugée
peu représentative à partir des figures 6 et 8. La partition 48, cœur
`x=[1440,1800)`, `y=[2480,2790)` et domaine paddé `660×610`, devient l'unique
partition de sélection. P0 est exclue de la sélection. La décision donnera la
priorité à la corrélation, à l'IoU top-10 % et au seuil absolu DIC 90 %, avec
RMSE/L2 comme métriques d'amplitude secondaires. Le protocole complet est figé
dans `validation/nonlocality_p48_preregistration.md` avant le calcul.

**Résultat de sélection P48 :** le calcul MFront converge sur 402 600 éléments
en `1335,97 s` de temps processus, avec 20/20 incréments, zéro cutback et
`7 869 356 KiB` de RSS maximal. Les trois métriques spatiales
pré-enregistrées sélectionnent `ell=58,88 µm` : corrélation
`0,2983 → 0,6160`, IoU top-10 % `0,1598 → 0,2822` et IoU au seuil absolu DIC
90 % `0,1676 → 0,3085`. RMSE et L2 relative diminuent de `64,61 %`. L'aire
active q90 reste `14,09 %` contre `10 %` DIC, sans collapsus. Le candidat étant
à la borne supérieure, l'optimum n'est pas encadré. Il est figé pour une
application sans ajustement aux partitions de confirmation.

**Confirmation tenue à l'écart :** P42, proposée avant l'exécution P48, est
pré-enregistrée comme premier cas de transfert. Seules les longueurs `0` et
`58,88 µm` seront comparées. Les seuils automatiques sont fixés avant calcul :
gain de corrélation `>=0,05`, réduction L2 relative `>=5 %`, gain d'IoU top-10
`>=0,02`, dérive moyenne relative `<=1e-10`. Au seuil absolu DIC 90 %, le gain
d'IoU doit être `>=0,02` et l'aire active filtrée rester entre 5 % et 20 %.
Voir `validation/nonlocality_p42_confirmation_preregistration.md`.

**Résultat confirmatoire P42 :** le calcul MFront converge sur 402 600
éléments en `1484,55 s` de temps processus, 20/20 incréments et zéro cutback.
Sans aucun ajustement de longueur, `58,88 µm` passe tous les seuils :
corrélation `0,4007 → 0,7036`, réduction L2 `65,43 %`, IoU top-10 %
`0,1334 → 0,2759`, IoU au seuil DIC 90 % `0,1774 → 0,2573`, et aire active
q90 `7,74 %` dans la plage pré-déclarée `[5 %,20 %]`.

**Conclusion de l'étape 1 : hypothèse de largeur spatiale soutenue.** Le même
candidat améliore les métriques d'amplitude et de localisation sur la
partition de sélection P48 et sur la partition P42 tenue à l'écart. Cette
conclusion ne transforme pas `58,88 µm` en longueur interne matérielle : le
point reste la borne supérieure du balayage et une seule partition de
confirmation est disponible.

**Critère de sortie :** un résultat FEM sauvegardé peut faire l'objet d'une
campagne de largeur spatiale traçable sans modifier le calcul mécanique, et le
rapport sépare faits numériques, sélection diagnostique et interprétation
physique.

### Phase prioritaire A.5 — Couplage constitutif micromorphique J2

- [x] Pré-enregistrer P154, le profil `20×20`, le padding 128, la longueur
      `58,88 µm`, le balayage de `Hchi` et les critères scientifiques
- [x] Ajouter la configuration typée et les options CLI non locales
- [x] Conserver la compatibilité `--count 25/100` et ajouter
      `--parts-x/--parts-y`
- [x] Ajouter les comportements MFront natif et tridimensionnel sans modifier
      les deux comportements de référence
- [x] Exposer `MicromorphicCouplingModulus` et
      `NonlocalEquivalentPlasticStrain`
- [x] Ajouter `Hchi*(p-chi)` au rayon de charge et `Hchi` à sa dérivée locale
- [x] Réutiliser le solveur DCT Helmholtz existant aux centres des éléments
- [x] Imbriquer le point fixe relaxé dans chaque essai Newton, sans `commit`
      intermédiaire
- [x] Restaurer conjointement déplacement, état MFront et `chi` lors d'un
      cutback
- [x] Sauvegarder `PEEQ_NONLOCAL`, `PEEQ_MISMATCH`,
      `NONLOCAL_HARDENING_MPA`, `YIELD_SURFACE_RADIUS_MPA` et
      `NONLOCAL_RESIDUAL`
- [x] Ajouter les temps MFront/Helmholtz, itérations, résidus, dérive de
      moyenne et échecs aux diagnostics
- [x] Vérifier que `Hchi=0` reproduit le calcul MFront local dans Newton
- [x] Vérifier le cas homogène, la tangente à `chi` fixé, les transactions et
      l'équivalence natif/3D condensé sur cas réduit
- [x] Ajouter une commande empreintée calculant
      `Href = median(K*n*p**(n-1))` sur le cœur plastifié local
- [x] Exécuter P154 local à 20 incréments et produire `HREF.json`
- [x] Ajouter un validateur empreinté local/couplé qui reconstruit les EVM
      depuis les déplacements bruts sur le cœur, sans post-filtrage
- [x] Exécuter les smoke tests à 5 incréments pour `alpha=0,0.5,1`
- [x] Exécuter les candidats retenus à 20 incréments avec padding 128
- [x] Comparer les champs bruts couplés à la DIC sur le cœur P154
- [!] Figer `Hchi` avant tout transfert vers P42 ou P48 : aucun candidat ne
      passe les huit critères pré-enregistrés
- [x] Rejouer un cas réduit avec le backend 3D condensé

**Preuve logicielle :** commits `3fe01d9`, `2102520`, `d3dfd33` et les commits
de validation ultérieurs. Le cas homogène couplé converge sans cutback. La
norme du point fixe est la norme mixte relative \(L_\infty\), indépendante du
nombre d'éléments, et utilise une branche absolue unitaire lors de
l'apparition de plasticité.

**Référence locale P154 :**
`validation/nonlocal_p154_local_reference.md`. Les 179 196 éléments convergent
en `793,98 s`, 20/20 incréments, 119 Newton et zéro cutback. Le cœur contient
24 507 éléments plastifiés sur 27 900. La médiane pré-enregistrée donne
`Href = 6547,530617 MPa`, donc `Hchi = 3273,765308 MPa` pour `alpha=0,5` et
`6547,530617 MPa` pour `alpha=1`.

**Smoke P154 :** `validation/nonlocal_p154_smoke_results.md`. Après
pré-enregistrement d'une norme mixte \(L_\infty\) indépendante du maillage,
`alpha=0,5` converge en `406,28 s`, `alpha=1` en `503,04 s` et `alpha=2` en
`226,30 s`, sans aucun échec du point fixe. Les trois passent tous les
critères smoke ; le prolongement à `alpha=2` était autorisé parce que le
meilleur point se trouvait à la borne supérieure du balayage initial.

**Validation P154 :** `validation/nonlocal_p154_validation_results.md`. Les
trois candidats positifs convergent à 20 incréments, padding 128, sans cutback.
`alpha=2` est le meilleur point testé : `+0,1643` de corrélation, `42,17 %`
de réduction L2, `+0,0331` d'IoU top-10 et `+0,0722` d'IoU q90. Il échoue
seulement sur l'aire active q90 (`21,85 %` au lieu de `<=20 %`). Le seuil
n'est pas déplacé a posteriori et aucun transfert confirmatoire n'est lancé.

**Critère de sortie :** partiellement atteint. P154 padding 128 converge à 20
incréments pour trois `Hchi>0` et la voie 3D condensée reproduit la voie native
sur le cas réduit. Aucun candidat ne passe toutefois tous les critères
scientifiques ; `Hchi` ne peut donc pas être figé ni transféré sans nouveau
protocole prospectif.

### Phase prioritaire A.6 — ROI P43 et chemin constitutif léger

- [x] Conserver le classement morphologique automatisé comme outil de
      présélection, sans lui déléguer le choix scientifique
- [x] Retenir P43 `(4,3)` après inspection visuelle de ses deux bandes
      diagonales ; cœur `360×310`, `x=[1440,1800)`, `y=[930,1240)`
- [x] Séparer les essais MFront sans tangente, avec tangente, puis la
      complétion tensorielle 3D finale
- [x] Préallouer les buffers Kelvin, PEEQ et `chi` du point fixe
- [x] Précalculer la direction du prédicteur pour le chargement DIC
      proportionnel
- [x] Chronométrer MFront avec/sans tangente, Kelvin, tenseurs 3D, forces
      internes, matrices élémentaires, assemblage, extraction et PARDISO
- [x] Comparer les états constitutifs bit à bit sur un crop réel et sur P43
- [x] Comparer un solveur EF complet avant/après sur la même zone et avec les
      mêmes paramètres
- [x] Figer la structure CSR libre-libre et mettre à jour uniquement `data`
- [x] Piloter explicitement PARDISO : phase 11 unique, puis phases 22/33
- [x] Conserver et tester le chemin générique `mtype=11`
- [x] Activer le CSR triangulaire et `mtype=2` uniquement pour le J2 vérifié
- [x] Rejeter tout tangent J2 dont l'asymétrie relative dépasse `1e-12`
- [x] Lancer la référence locale P43 avec le profil scientifique retenu
- [x] Estimer `Href` sur le cœur P43, puis pré-enregistrer le balayage
      `alpha=0,1,2,4`
- [x] Lancer et visualiser les campagnes P43 sans ajustement rétroactif

**Preuve :** commit `d5b0e7e` et
`validation/performance/nonlocal_hot_path_optimization.json`. Sur P43, le
benchmark constitutif passe de `14,357 s` à `7,605 s` et de `796 856` à
`564 508 KiB`, avec quatre empreintes de champs identiques. Sur le gate EF
P187, le temps processus passe de `396,78 s` à `273,56 s` et le pic RSS baisse
de `12,7 %`. Les deux versions conservent exactement 20 tentatives,
13 incréments acceptés, 7 cutbacks, 156 Newton et 623 itérations non locales.
Les écarts des champs physiques restent inférieurs à `1,1e-12` relativement à
leur amplitude globale.

**Campagne P43 :** `validation/nonlocal_p0043_validation_results.md`. La
référence locale et les candidats `alpha=1,2,4` convergent tous à 20
incréments sans cutback. La corrélation EVM passe de `0,3791` à
`0,4624/0,4814/0,5036` et l'erreur L2 relative de `0,9516` à
`0,6174/0,5256/0,4341`. `alpha=2` maximise légèrement l'IoU top-10 tandis que
`alpha=4` maximise corrélation et IoU q90 : les deux restent non dominés et
aucun `Hchi` n'est figé.

**Interprétation détaillée :**
`docs/explanation/p43_coupled_results.md` commente séparément les cartes EVM,
les erreurs signées, les champs et distributions PEEQ, le coût numérique et
les conclusions temporaires. Le couplage réduit le pic PEEQ de `81,9 %`, son
RMS de gradient de `65,3 %` et sa variation totale de `56,0 %`, pour seulement
`9,3 %` de baisse de moyenne : il s'agit bien d'une redistribution. La baisse
du rappel q90 et le léger recul de l'IoU top-10 à `alpha=4` empêchent toutefois
de conclure que toute augmentation supplémentaire serait bénéfique.

**Critère de sortie :** atteint pour l'optimisation technique et l'exécution
du balayage P43. Cette phase ne modifie ni la loi, ni
`ell`, ni `Hchi`, ni les tolérances, ni Newton, ni le point fixe, ni la
tangente. Le second lot modifie seulement l'assemblage sparse et le cycle
PARDISO. Sur P187, il ajoute `-10,6 %` de temps processus et `-16,7 %` de pic
RSS par rapport au chemin constitutif déjà optimisé ; une phase 11 et 139
paires 22/33 sont enregistrées. Le troisième lot J2 symétrique réduit encore
le temps `244,67→227,34 s`, PARDISO de `38,0 %` et le pic RSS de `8,7 %`.
La plasticité cristalline reste par défaut sur le chemin complet `mtype=11`.

### Phase prioritaire A.7 — Identification conjointe rapide de `ell` et `Hchi`

**Mission :** déterminer si la longueur `ell` et le module de couplage `Hchi`
sont séparément identifiables et transférables, sans grille F2 exhaustive et
sans déclarer prématurément une longueur matérielle. Le domaine initial est
`alpha∈[1,6]`, `ell∈[20,60] µm`, avec le témoin local unique `alpha=0`.

**Contrats scientifiques :**

- conserver le modèle micromorphique, MFront, MGIS transactionnel, Newton et
  les deux voies de contraintes planes inchangés ;
- paramétrer et enregistrer `alpha`, `Hchi`, `ell` et
  `Achi=Hchi*ell**2`, avec interpolation possible dans
  `(log(Hchi), log(Achi))` ;
- comparer uniquement l'EVM totale reconstruite par le même opérateur de
  mesure DIC ; PEEQ reste un diagnostic interne ;
- distinguer explicitement F0 heuristique, F1 de classement et F2
  scientifique ;
- ne jamais réutiliser un cache dont les empreintes physiques, numériques ou
  d'observation diffèrent ;
- ne lancer aucun nouveau calcul F2 sans validation humaine explicite.

**Ordre imposé et suivi :**

- [x] Auditer les solveurs Helmholtz, métriques, validateurs, partitions,
      formats de campagne et commandes réutilisables
- [x] Formaliser les unités et les conversions
      `(alpha,ell) <-> (Hchi,Achi)`, y compris le cas local canonique
- [x] Formaliser et empreinter l'opérateur `M_DIC`
- [x] Ajouter les métriques d'amplitude, localisation, spectre spatial et
      diagnostics PEEQ
- [x] Implémenter le crible F0 sur PEEQ local figé et ses diagnostics
      énergétiques/spectraux
- [x] Valider les tendances F0 contre les F2 P43 existants
      `alpha=0,1,2,4`, `ell=58,88 µm`
- [x] Implémenter F1 avec réduction spatiale configurable, historique complet,
      reprise, cache strict et statuts individuels
- [x] Valider le classement F1 contre les quatre points F2 P43 existants
- [x] Implémenter le profil `Hchi*(ell)`, PCHIP/sécante contrôlée et la courbe
      de recherche principalement unidimensionnelle
- [x] Construire le front de Pareto amplitude-localisation, le genou et les
      cartes `(ell,alpha)` / `(Hchi,Achi)`
- [x] Générer un manifeste de cinq nouveaux calculs F2 au maximum, incluant
      obligatoirement `(ell=58,88 µm, alpha=6)`, sans les lancer
- [x] Préparer une validation de transfert de trois couples au maximum sur une
      autre ROI, sans recalage
- [x] Produire configuration, CSV consolidé, figures, rapport, documentation,
      tests, HTML et PDF

**Point d'arrêt obligatoire :** après génération du manifeste F2, présenter
pour chaque candidat `ell`, `alpha`, `Hchi`, `Achi`, justification, coût
estimé et métrique discriminée. Attendre une validation humaine avant toute
exécution haute fidélité. Un éventuel second lot de deux points au maximum
nécessitera également une commande explicite.

**État initial :** P43 fournit quatre F2 réutilisables à
`ell=58,88 µm`, `alpha=0,1,2,4`. `H_ref` doit toujours être lu dans
`HREF.json` ou les métadonnées de campagne ; la valeur numérique n'est jamais
codée en dur. La page `docs/explanation/p43_coupled_results.md` constitue le
diagnostic scientifique de départ.

**Audit d'architecture (terminé) :**

- F0 réutilise directement
  `postprocessing.helmholtz.helmholtz_filter_element_field`; une seule
  résolution DCT est effectuée par longueur, puis tous les `Hchi` réemploient
  le même écart `p-chi`.
- L'observable principale réutilise
  `workflows.nonlocality_diagnostic.reconstruct_historical_evm` et les
  métriques existantes de `postprocessing.metrics`.
- Les lectures de campagnes doivent reprendre les contrôles de manifeste,
  statut et empreinte de `coupled_nonlocal_validation`; les fonctions
  génériques seront extraites dans un module partagé au lieu d'être copiées.
- F1 ne constitue pas un nouveau solveur : les champs globaux sont réduits
  de façon déterministe, puis transmis au `PartitionWorkflow` existant, qui
  conserve MFront/MGIS, Newton, PARDISO, les cutbacks, les sorties atomiques
  et la reprise.
- Le cache d'identification complète, sans remplacer, les manifestes du
  `PartitionWorkflow`. Sa clé inclut les empreintes de maillage, DIC,
  paramètres locaux, historique, opérateur DIC, fidélité, paramètres
  micromorphiques et commit.
- La CLI est une sous-commande unique avec actions explicites. Seule l'action
  F1 peut lancer des calculs réduits ; la génération F2 écrit uniquement un
  manifeste et des commandes reproductibles.

**Socle d'identification implémenté :**

- `identification.parameters.NonlocalIdentificationPoint` canonise le témoin
  local et fournit les deux systèmes de coordonnées avec unités explicites ;
- `identification.observation.DICObservationOperator` versionne et empreinte
  la mesure EVM, le support, le masque, le cœur et l'éventuelle réduction par
  nœuds coïncidents ;
- `identification.metrics` sépare erreurs globales, objectif d'amplitude par
  quantiles, recouvrement top-10 relatif, seuil DIC absolu, gradient,
  variation totale, spectre radial et diagnostics PEEQ ;
- les tests unitaires couvrent les conversions inverses, l'unicité de
  `alpha=0`, la réduction de l'opérateur DIC, les deux IoU et le spectre d'un
  sinus.

**Crible F0 implémenté :**

- configuration versionnée :
  `configs/joint_nonlocal_identification_p0043.yaml` ;
- commande :
  `fem-inhouse identify-nonlocal {inspect,screen-frozen} --config ...` ;
- accès partagé aux campagnes avec vérification du manifeste, du statut et de
  chaque empreinte de champ ;
- une DCT par longueur, puis réemploi exact de `p0-chi_ell` pour les 21
  valeurs de `alpha` et le témoin local unique ;
- sorties cache-strictes `manifest.json`, `frozen_screen.csv`,
  `length_diagnostics.json` et `proxy_validation.json` ;
- énergies locale et de gradient, multiplicateur spectral, normes, quantiles,
  gradient, variation totale, résidu Helmholtz et dérive de moyenne ;
- aucune résolution mécanique et aucune exécution F2 dans cette action.

**Résultat F0 P43 :** 22 longueurs et 21 niveaux positifs de `alpha`, soit
463 points avec le témoin local, sont évalués en `7,84 s` mur et `141404 KiB`
de pic RSS. À `ell=58,88 µm`, l'intensité du proxy présente une corrélation de
rang `-1` avec l'erreur L2 F2, le maximum, l'écart-type et la variation totale
PEEQ sur `alpha=0,1,2,4`. Cette monotonie justifie un criblage, mais ne valide
pas les amplitudes mécaniques prédites.

**F1 implémenté :**

- réduction surfacique des cartes élémentaires et sous-échantillonnage des
  déplacements aux nœuds physiquement coïncidents ;
- étendue physique, découpage et séparation cœur/padding conservés ;
- contrôle obligatoire `ell/h_F1 >= 3` ;
- reprise du `PartitionWorkflow` de production, sans second solveur ;
- historique rejoué depuis l'état initial avec 10 incréments, cutbacks
  conservés et tolérance F1 explicitement enregistrée ;
- manifestes et rapports individuels cache-stricts, métriques EVM/PEEQ sur le
  cœur et validation automatique du classement contre F2 ;
- plan P43 initial : quatre F1 à `ell=58,88 µm`,
  `alpha=0,1,2,4`, grille `1800x1550`, padding `75`, résolution
  `ell/h_F1=16`. Le lancement reste explicite via
  `identify-nonlocal run-low-fidelity`.

**Validation F1 P43 :** les quatre points réduits convergent en `14 min
43,21 s` au total, avec `1428156 KiB` de pic RSS. Les durées mécaniques sont
`127,06 / 217,32 / 243,72 / 289,60 s` pour `alpha=0/1/2/4`.
Les six critères pré-enregistrés passent : mêmes classements L2 et
corrélation, erreur absolue de corrélation `<=0,05`, erreur L2 relative
`<=15 %`, et erreurs IoU top-10/q90 `<=0,05`. F1 est donc autorisé à classer
les candidats, jamais à remplacer F2 comme résultat scientifique.

**Profil et Pareto implémentés :**

- collecte immuable F1/F2 avec tableau consolidé et empreintes complètes ;
- plan F1 sparse par défaut `ell={20,40,60} µm`,
  `alpha={1,3.5,6}`, soit neuf points non exhaustifs ;
- profil par PCHIP et minimisation bornée, avec détection explicite de
  monotonie, optimum de bord et besoin d'un point de confirmation ;
- front non dominé sur `(J_amp, 1-IoU_q90_absolue)` et genou calculé après
  normalisation des deux objectifs ;
- les actions de profil/sélection refusent de proposer F2 tant que les points
  F1 requis manquent ou que la validation F1 n'est pas réussie.

**Plan F1 séquentiel exécuté :**

- support initial sparse :
  `ell={20,40,60} µm`, `alpha={1,3.5,6}` ;
- sept points convergés sur neuf ; les points `(20 µm,3.5)` et
  `(20 µm,6)` échouent proprement lorsque le cutback passe sous le minimum ;
- l'échec n'est ni masqué ni contourné par une modification de tolérance :
  les deux points adaptatifs `(20 µm,2)` et `(20 µm,2.5)` sont ajoutés pour
  encadrer le plus fort couplage court convergé ;
- plan initial : `2967,86 s`, pic RSS `1443392 KiB` ;
- complément adaptatif : `630,0 s`, pic RSS `1403376 KiB` ;
- treize résultats F1 et quatre F2 sont consolidés dans une collection
  immuable avec CSV, JSON, empreintes et fidélité explicite.

**Diagnostic de convergence aux couplages courts (en cours) :**

- [x] conserver le Picard amorti historique comme stratégie par défaut et
      préserver exactement son chemin arithmétique ;
- [x] ajouter une évaluation MFront légère sans tangente qui expose à la fois
      `PEEQ` et `YieldSurfaceRadius`, sans reconstruction des tenseurs 3D ;
- [x] enregistrer par itération le résidu micromorphique absolu/relatif, la
      relaxation, les variations maximales de `p` et `chi`, les bornes de
      `Hchi*(p-chi)`, le résidu Helmholtz et les bornes du rayon de charge ;
- [x] rattacher ces données à l'incrément, au pseudo-temps, à l'itération de
      Newton et au résidu mécanique lorsque celui-ci est disponible ;
- [x] arrêter explicitement un essai dès que le rayon de charge devient non
      positif, au lieu de chercher à forcer numériquement la convergence ;
- [x] classifier un échec en données insuffisantes, oscillation, divergence,
      stagnation lente ou domaine constitutif non admissible ;
- [x] implémenter Aitken optionnel, borné, avec rejet si le résidu croît de
      plus de 25 % et repli automatique vers une relaxation plus faible ;
- [x] exposer tous les réglages par configuration et CLI, sans activer Aitken
      dans les campagnes historiques ;
- [x] valider le code par `312 passed`, Ruff et mypy avant le rejeu réel ;
- [x] rejouer `(ell=20 µm, alpha=3.5)` avec les paramètres F1 historiques et
      l'instrumentation seule ;
- [x] établir que le point fixe converge sans échec, avec au plus 12
      itérations et `R_min=31,597 MPa` ; le premier cutback vient de Newton,
      dont le résidu relatif vaut `3,209e-6` à l'itération limite 15 pour une
      tolérance de `3e-6` ;
- [x] tester causalement 25 Newton avec tous les autres contrôles historiques
      inchangés : 10/10 incréments, aucun cutback, 134 Newton, maximum 17,
      `441,85 s`, `R_min=28,386 MPa` ;
- [ ] n'activer le profil Aitken/40 incréments/50 itérations que si une trace
      montre effectivement une oscillation ou un échec micromorphique ;
- [x] essayer `(20 µm,6)` avec le même profil Newton-25 : 10/10 incréments,
      aucun cutback, 174 Newton, maximum 22, `503,21 s`,
      `R_min=33,248 MPa` ;
- [x] documenter le diagnostic, les limites et la différence entre
      stabilisation numérique et modification constitutive.

La campagne instrumentée est volontairement séparée dans
`configs/joint_nonlocal_fixed_point_diagnostic_p0043.yaml` et
`results/joint-nonlocal-fixed-point-diagnostic-p0043`. Elle ne peut donc ni
écraser ni valider par accident les anciens points F1. Le commit technique de
référence est `a5c1de4`. Le test causal Newton utilise à son tour
`configs/joint_nonlocal_newton_diagnostic_p0043.yaml` et un répertoire de
résultats distinct.

**Conclusion temporaire du diagnostic court :** l'ancien plafond convergé
`alpha=2.5` à `ell=20 µm` était numérique. Avec Newton-25, `alpha=3.5` puis
`alpha=6` convergent, sans Aitken et sans rayon de charge non positif.
L'objectif d'amplitude continue de décroître (`0,8191` à `alpha=2.5`,
`0,6491` à 3.5, `0,4007` à 6). La proposition F2 existante qui contenait
`(20 µm,2.5)` est donc **superseded** et ne doit pas être lancée. Il faudra
reconstruire une collection F1 et une proposition F2 immuables avec la
politique Newton-25 déclarée. Le registre compact est
`validation/joint_nonlocal_fixed_point_diagnostic_p0043.json`.

**Résultat historique des profils (superseded à `ell=20 µm`) :**

- `ell=20 µm` : `alpha={1,2,2.5}`, minimum d'amplitude sur la borne
  convergée `alpha=2.5` dans la collection initiale ; le diagnostic
  Newton-25 étend maintenant la tendance monotone jusqu'à `alpha=6` ;
- `ell=40 µm` : `alpha={1,3.5,6}`, minimum sur `alpha=6` ;
- `ell=60 µm` : `alpha={1,3.5,6}`, minimum sur `alpha=6` ;
- tous les profils restent monotones : aucun optimum intérieur et aucune
  identifiabilité séparée de `Hchi` et `ell` ne sont démontrés ;
- front F1 non dominé :
  `(40 µm,6)` genou, `(58,88 µm,4)` meilleure localisation q90 absolue,
  `(60 µm,6)` meilleure amplitude.

**Point d'arrêt F2 historique, sans lancement et désormais superseded :**

- manifeste immuable :
  `results/joint-nonlocal-identification-p0043/f2-proposals/`
  (clé exacte donnée par le rapport courant) ;
- quatre propositions, et non cinq :
  `(58,88 µm,6)`, `(60 µm,3.5)`, `(40 µm,6)` et `(20 µm,2.5)` ;
- `(60 µm,6)`, meilleur point F1 d'amplitude, est volontairement représenté
  par le calcul obligatoire `(58,88 µm,6)` : l'écart de longueur n'est que
  `1,9 %` et deux F2 seraient presque redondants ;
- temps séquentiel total estimé : `9584,7 s`, soit `2,66 h` ;
- chaque ligne contient `ell`, `alpha`, `Hchi`, `Achi`, coût, justification,
  métrique discriminée, destination et commande complète ;
- `automatic_execution=false`, `human_approval_required=true` et tous les
  statuts restent `proposed_not_run`.
- ce manifeste est conservé pour la provenance mais ne doit plus être exécuté
  depuis que `(20 µm,3.5)` et `(20 µm,6)` ont convergé avec Newton-25.

**Transfert et rapport préparés :**

- manifeste de transfert limité à trois couples, paramètres figés et
  `recalibration_allowed=false` ; statut `awaiting_validation_roi` ;
- figures SVG/PNG/PDF : plan sparse, coordonnées `(log Hchi,log Achi)`,
  profils, front de Pareto et hiérarchie de coût ;
- page anglaise détaillée :
  `docs/explanation/joint_nonlocal_identification.md` ;
- guide reproductible :
  `docs/how-to/run_joint_nonlocal_identification.md` ;
- registre de performance :
  `validation/joint_nonlocal_identification_p0043_benchmarks.json`.

**Validation finale du jalon :**

- Ruff : réussi sur tout le dépôt ;
- mypy : réussi sur les 44 fichiers source ;
- suite complète avec MGIS/MFront réel : `313 passed` ;
- Sphinx HTML strict : réussi, page
  `docs/_build/html/explanation/joint_nonlocal_identification.html` ;
- Sphinx LaTeX/PDF strict : réussi, manuel de 184 pages sous
  `docs/_build/latex/kinematics-driven-316l-strain-reconstruction.pdf`.

### 2026-07-26 — Séparation de `alpha` et `ell` par expériences discriminantes

**Décision scientifique :** aucun optimum intérieur n'est établi. Les
profils à `ell=20/40/60 µm` atteignent encore leur meilleure amplitude sur
la borne `alpha=6`. Le nouveau plan ne cherche donc pas un « meilleur couple »
sur une grille arbitraire. Il doit tester :

1. une éventuelle saturation selon `alpha=6 -> 9 -> 12` ;
2. la dégénérescence à `Achi = Hchi*ell²` constant ;
3. l'effet propre de `ell` à `alpha=6` constant ;
4. l'évolution des champs à 25, 50, 75 et 100 % du chargement.

**Protocole F1 homogène versionné :**

- configuration :
  `configs/joint_nonlocal_identifiability_p0043_newton25.yaml` ;
- répertoire neuf et cache incompatible avec l'ancienne collection :
  `results/joint-nonlocal-identifiability-p0043-newton25` ;
- 25 itérations Newton maximum partout ;
- 10 incréments, tolérance relative `3e-6`, Picard fixe `0.5`, 15
  itérations micromorphiques et même cutback pour tous les points ;
- Aitken reste désactivé ;
- 23 calculs F1 uniques incluant le témoin local, la reproduction homogène
  des anciens points, les profils de saturation et les deux nouveaux points
  à `Achi` constant ;
- aucun calcul F2 automatique.

**Infrastructure implémentée :**

- option CLI `--identifiability-design`, distincte du produit cartésien
  historique `--design` ;
- fusion déterministe des rôles expérimentaux et déduplication des points ;
- ligne `Achi` constant ancrée sur `(ell=20 µm, alpha=6)` :
  `(20,6)`, `(30,2.666666...)`, `(40,1.5)` ;
- sauvegarde atomique et vérifiée des snapshots `U`, `E` et `PEEQ` par le
  workflow partitionné ;
- reconstruction de l'EVM DIC à chaque niveau par mise à l'échelle du
  déplacement imposé proportionnel, et EVM FEM depuis le snapshot convergé ;
- métriques de structure spatiale ajoutées : largeur et longueur de bande,
  orientation, position d'axe, longueurs de corrélation x/y, centroïde
  spectral et distance spectrale radiale ;
- seuils DIC absolus q80/q90/q95 conservés séparément des quantiles propres à
  chaque champ.
- analyse automatique des trois expériences (saturation, `Achi` constant et
  `alpha` constant), avec comparaison aux quatre snapshots de chargement ;
- génération F2 désormais bloquée si la collection discriminante est
  incomplète ou si au moins un profil atteint encore la borne `alpha=12`
  sans satisfaire les critères de plateau ; l'ancien manifeste reste donc
  inutilisable par construction.

**État :**

- [x] architecture, configuration, garde F2 et tests synthétiques ;
- [x] exécuter les 23 points F1 homogènes : 21 convergences en environ
  `2 h 58 min`, échecs propres uniquement pour `(20 µm,9)` et `(20 µm,12)` ;
- [x] consolider saturation, ligne `Achi` constante et évolution temporelle ;
- [x] régénérer le front de Pareto, les cartes EVM/PEEQ et les conclusions ;
- [x] bloquer toute proposition F2 : statut `numerically_censored`.

**Résultats temporaires :**

- optimum d'amplitude intérieur à `(40 µm, alpha=9)`,
  `Jamp=0.0437629` ;
- optimum d'amplitude intérieur à `(60 µm, alpha=6)`,
  `Jamp=0.0484263` ;
- à `alpha>alpha*`, L2 et corrélation continuent de s'améliorer mais l'IoU
  absolue, la largeur apparente et la structure PEEQ se dégradent fortement ;
- la ligne exacte `Achi` constant
  `(20,6)/(30,2.666666...)/(40,1.5)` n'est pas dégénérée : dispersion
  relative `17.0 %` sur L2, `65.1 %` sur l'objectif d'amplitude et `34.6 %`
  sur l'erreur spectrale radiale ;
- la sensibilité propre à `ell` est donc observable en F1 sur P43, mais la
  séparation statistique des deux paramètres n'est pas démontrée tant que la
  sensibilité maillage/DIC/inter-ROI et la censure à `20 µm` ne sont pas
  résolues ;
- aucun manifeste F2 nouveau et aucune conclusion de longueur matériau.

**Traçabilité :**

- collection :
  `a084774ae9940c6fdfc0da16473464dc84549d83bc774873bd117e433176905f` ;
- attestation d'exécution :
  `validation/joint_nonlocal_identifiability_p0043_newton25_execution.json` ;
- le processus a chargé le code `f5596d1` au lancement ; des commits de
  documentation pendant la boucle ont fait dériver le `HEAD` observé par
  certains manifestes sans changer le code Python en mémoire ;
- correction ajoutée : les prochaines campagnes épinglent l'état Git une
  seule fois avant la boucle des points.

### Phase différée B — Validation externe Abaqus

- [ ] Récupérer ou régénérer un petit `.inp` de référence
- [ ] Extraire les mêmes champs aux mêmes emplacements physiques
- [ ] Comparer automatiquement `U/S/E/PE/PEEQ/RF`
- [ ] Étendre la comparaison à plusieurs pseudo-temps si les ODB deviennent
      disponibles

Cette phase ne bloque pas la phase A ni les campagnes DIC/EF du cas d'étude.

### Semaine 1 — Contrat scientifique

- [x] Écrire les conventions d'axes `U/V`, `x/y`, axes NumPy 0/1
- [x] Définir les unités de toutes les entrées et sorties
- [x] Définir `epsilon_xy` tensoriel et `gamma_xy` ingénieur
- [x] Définir la formule de `epsilon_vM` sous contrainte plane
- [x] Définir les quatre courbes de contrainte-déformation
- [ ] Vérifier la section et l'épaisseur réellement utilisées dans Abaqus
- [x] Identifier et versionner les quatre tableaux disponibles du ROI
- [~] Établir un jeu de données réduit pour les tests rapides ; les données
  complètes sont versionnées par Git LFS
- [ ] Décider et documenter les tolérances avant comparaison finale

**Critère de sortie :** document scientifique relu et approuvé, sans convention
implicite.

### Semaines 2–3 — Préparation DIC et calcul autonome

- [x] Reproduire la table plastique Abaqus :
  - domaine `0 <= ep <= 0.2` ;
  - 1000 points ;
  - traitement documenté du premier incrément `1e-6`.
- [x] Tester séparément loi analytique et loi tabulée
- [x] Corriger le calcul et l'étiquetage de la contrainte EF directe
- [x] Corriger les conventions d'axes et de cisaillement
- [ ] Comparer contraintes et déformations au même emplacement physique
- [x] Définir la méthode commune de calcul des déformations depuis `U`
- [~] Vérifier `U1`, `U2`, `S11`, `S22`, `S12`, `PEEQ` : contrats internes
  couverts, comparaison Abaqus encore absente
- [x] Vérifier le signe et la définition des réactions
- [x] Ajouter des assertions sur PEEQ au test biaxial

**Critère de sortie :** sous-domaine réel préparé depuis les données brutes,
calculé par `fem_inhouse` et accompagné d'un manifeste complet. La parité Abaqus
est reportée à la phase B.

### Semaines 4–5 — Ingénierie logicielle

- [x] Créer un `pyproject.toml`
- [x] Verrouiller les dépendances et versions
- [x] Créer un paquet sous `src/fem_inhouse`
- [~] Séparer :
  - [x] maillage ;
  - [x] élément ;
  - [x] matériau constitutif ;
  - [x] assemblage ;
  - [x] solveur non linéaire ;
  - [x] résultats ;
  - [x] post-traitement.
- [x] Remplacer les 19 paramètres de `run_fem` par des configurations typées
- [x] Ajouter les validations d'entrée
- [x] Supprimer les effets de bord lors des imports
- [x] Supprimer les chemins absolus
- [x] Ajouter une CLI limitée au cas d'étude
- [x] Ajouter Ruff, Pyright ou mypy, pytest et couverture
- [x] Ajouter une journalisation structurée
- [x] Échouer explicitement si PyPardiso n'est pas disponible en production

**Critère de sortie :** installation fraîche et cas réduit exécutables par une
commande documentée.

### Semaines 6–8 — Partitionnement, padding et raccordement

- [x] Définir une grille déterministe de 25 partitions
- [x] Définir une grille déterministe de 100 partitions
- [x] Gérer correctement les partitions de bord et de coin
- [x] Extraire les cartes matériau et les déplacements locaux
- [x] Ajouter le padding configurable
- [x] Résoudre indépendamment chaque partition
- [x] Enregistrer uniquement les résultats nécessaires par partition
- [x] Extraire et raccorder les cœurs non recouverts
- [x] Garantir l'absence de trous, doublons et décalages d'indices
- [x] Permettre une reprise après interruption
- [x] Ajouter un manifeste et une empreinte des entrées
- [x] Produire des fichiers `.npy` mappés en mémoire pour le champ global
- [x] Rendre l'ordre d'exécution des partitions sans effet sur le résultat
- [x] Ajouter un modèle de job array pour le calcul parallèle

**Critère de sortie :** domaine réduit identique entre calcul monolithique et
calcul partitionné avec padding suffisant.

### Semaine 9 — Performance et ressources

- [~] Mesurer temps et mémoire pour 10k, 50k, 100k et 350k éléments
- [~] Mesurer séparément assemblage, factorisation, Newton et écriture
- [~] Vérifier le nombre de threads MKL
- [ ] Définir la taille maximale d'une partition pour la machine cible
- [x] Réserver le repli SciPy au diagnostic ; PyPardiso reste obligatoire
- [ ] Définir un budget mémoire et un budget de temps par partition
- [~] Vérifier l'absence de copies mémoire évitables : tenseurs constitutifs
  globaux supprimés, structures sparse encore à profiler à grande taille
- [x] Documenter la stratégie de parallélisation

**Critère de sortie :** dimensionnement documenté avant tout calcul sur
11,16 millions d'éléments.

### Semaines 10–11 — Validation hiérarchique

#### Niveau 1 : vérification mathématique

- [x] Partition de l'unité et dérivées des fonctions de forme
- [x] Jacobien positif
- [x] Patch test élastique
- [x] Trois modes rigides
- [x] Retour plastique uniaxial, biaxial et en cisaillement
- [x] Tangente par différences finies
- [x] Cas tabulé dans chaque segment et au-delà de `ep = 0.2`
- [x] Équilibre des réactions
- [x] Convergence en nombre d'incréments

#### Niveau 2 : parité Abaqus

- [ ] Campagne différée jusqu'à stabilisation du pipeline DIC autonome
- [ ] Comparaison sur 10×10 ou 20×20
- [ ] Comparaison sur un sous-domaine hétérogène représentatif
- [ ] Comparaison à plusieurs pseudo-temps
- [~] Rapport automatique avec seuils de succès : commande prête, références
  Abaqus/DIC encore absentes

#### Niveau 3 : partitionnement

- [x] Référence monolithique sur domaine réduit
- [x] Comparaison sans recouvrement
- [ ] Comparaison avec padding 50, 100, 150 et 200
- [ ] Étude du nombre de partitions
- [!] Calcul et convergence de la métrique BGE
- [x] Mesure spécifique des erreurs aux interfaces

#### Niveau 4 : reproduction scientifique

- [~] RMSE du déplacement `U2` : outil et champ final DIC disponibles,
  résultat EF global à produire
- [~] RMSE et MAE de `epsilon_vM` : outil et champ final DIC disponibles,
  résultat EF global à produire
- [~] Carte de différence signée : génération prête, préparation DIC à finaliser
- [ ] BGE
- [~] Corrélation spatiale des champs : outil et champ DIC disponibles
- [~] Recouvrement des zones de plus forte localisation : métrique testée,
  résultat EF global encore absent
- [ ] Quatre courbes de contrainte-déformation séparées
- [ ] Intervalles de confiance calculés selon la méthode documentée
- [ ] Comparaison 25 partitions / 100 partitions avec padding 150

**Critère de sortie :** rapport reproductible expliquant les accords, écarts et
artefacts de raccordement.

### Semaine 12 — Documentation et version de référence

- [x] README de démarrage rapide
- [x] Documentation Sphinx intégralement en anglais et structurée avec Diátaxis
- [x] Landing page Read the Docs orientant vers tutoriels, guides, référence et explications
- [x] Compilations HTML stricte et PDF disponibles
- [x] Figures scientifiques vectorielles SVG/PDF reproductibles
- [x] Vérification automatique de la documentation HTML et des figures dans la CI
- [x] Tutoriel complet du cas réduit
- [x] Documentation du modèle numérique
- [x] Documentation des conventions
- [x] Documentation du partitionnement
- [x] Documentation de la validation
- [x] Documentation des limites scientifiques
- [x] Commandes uniques `test`, `validate`, `example`
- [x] CI verte sur une installation fraîche
- [ ] Revue indépendante scientifique
- [ ] Revue indépendante logicielle
- [ ] Version figée `1.0.0-case-study`

## 9. Architecture cible

```text
fem_inhouse/
├── pyproject.toml
├── README.md
├── LICENSE                 # décision juridique encore ouverte
├── Claude.md
├── data/
│   └── raw/case_study/     # tableaux immuables suivis par Git LFS + manifeste
├── references/
│   └── legacy_abaqus/      # provenance, jamais importée par le paquet
├── src/fem_inhouse/
│   ├── data_preparation.py
│   ├── config.py
│   ├── core/
│   │   ├── mesh.py
│   │   ├── element.py
│   │   ├── constitutive.py
│   │   ├── assembly.py
│   │   ├── nonlinear.py
│   │   └── solver_legacy.py
│   ├── partitioning/
│   │   ├── layout.py
│   │   ├── overlap.py
│   │   ├── extract.py
│   │   └── stitch.py
│   ├── postprocessing/
│   │   ├── strain.py
│   │   ├── invariants.py
│   │   ├── stress_curves.py
│   │   └── metrics.py
│   ├── workflows/
│   │   ├── solve_partition.py
│   │   └── reconstruct_roi.py
│   ├── results.py
│   └── cli.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── examples/case_study/
├── validation/
│   ├── abaqus_input/
│   ├── abaqus_extraction/
│   ├── reference_data/
│   └── reports/
└── docs/
```

Cette architecture doit rester limitée au cas d'étude. Aucun système générique
de plugins pour des éléments ou matériaux non prévus n'est demandé.

## 10. Critères de maturité 4/5

### Noyau numérique

- [x] Tous les tests mathématiques critiques du modèle supporté passent
- [x] Tangente cohérente vérifiée automatiquement
- [x] Convergence robuste sur cas homogène et hétérogène
- [x] Échec de convergence diagnostiqué sans résultat silencieusement invalide

### Validation scientifique

- [ ] Parité Abaqus démontrée sur petits cas, validation externe différée
- [ ] Métriques de l'article reproduites ou écarts expliqués
- [x] Contrainte directe séparée des reconstructions
- [ ] Artefacts de partition quantifiés
- [ ] Seuils définis avant lecture des résultats finaux

### Ingénierie logicielle

- [x] API publique typée
- [x] Au moins 85 % de couverture des lignes
- [x] Au moins 80 % de couverture des branches
- [x] Couverture dédiée de toutes les fonctions constitutives critiques
- [x] Aucun avertissement qualité non justifié
- [~] Revue de code obligatoire pour les formules numériques : procédure et
  modèle de PR ajoutés, protection de branche non activée

### Reproductibilité

- [x] Installation fraîche reproductible
- [x] Versions verrouillées
- [x] Données DIC et cartes locales brutes versionnées par Git LFS et identifiées
  par empreinte
- [x] Préparation brute → canonique automatisée, atomique et manifestée
- [x] Aucun chemin dépendant d'un poste personnel
- [x] Résultats accompagnés de leur configuration et version du code
- [x] Workflow reprenable partition par partition

### Performance

- [x] PyPardiso/MKL utilisé et vérifié
- [~] Temps et mémoire mesurés : 10k, 50k et 100k terminés ; 350k reporté
- [x] Cas de production compatible avec la machine cible : les partitions
  intérieures P48 et P42 de 402 600 éléments ont convergé en 22 min 16 s et
  24 min 45 s, avec respectivement 7,51 GiB et 7,71 GiB de pic RSS
- [x] Traitement hors mémoire du ROI complet
- [~] Absence de régression de performance supérieure au seuil défini :
  comparaison A/B disponible, seuil global encore à ratifier

### Documentation

- [x] Une personne externe peut installer et exécuter le cas réduit
- [ ] Une personne externe peut reproduire les figures principales
- [x] Les hypothèses et limites sont visibles
- [x] Les descripteurs locaux ne sont pas présentés comme propriétés de grains

### Évaluation provisoire au 2026-07-25

| Axe | Note | Justification principale |
|---|---:|---|
| Noyau numérique | 4,5/5 | Cas fermés, tangente, cutback, réactions et cisaillement testés |
| Validation scientifique | 4,0/5 | Sélection P48 et confirmation P42 pré-enregistrées et archivées ; la même longueur améliore amplitude et localisation, mais le ROI raccordé et la comparaison Abaqus globale manquent |
| Ingénierie logicielle | 4,5/5 | API typée, modules séparés, CI, 230 tests, revue documentée |
| Reproductibilité | 4,5/5 | Données LFS, préparation atomique, manifestes et smoke test DIC réel |
| Performance | 4,0/5 | 10k–100k mesurés ; deux partitions intérieures de 402,6k exécutées avec temps et mémoire archivés |
| Documentation | 4,5/5 | Site Sphinx anglais structuré avec Diátaxis, PDF, contrats, API et figures vectorielles reproductibles ; revue externe encore requise |

Les notes ne doivent pas être relevées artificiellement par des cas
synthétiques. Pour atteindre 4/5 partout, les chemins critiques sont désormais :

1. terminer et vérifier le pipeline autonome depuis les données DIC versionnées ;
2. sélectionner une longueur diagnostique sur une partition, la tester sans
   ajustement sur des partitions tenues à l'écart, puis raccorder les champs
   du calcul réel et reproduire les métriques expérimentales accessibles ;
3. réserver une fenêtre machine permettant les mesures 350k et le
   dimensionnement d'une partition de production ;
4. réaliser ensuite la validation externe Abaqus lorsqu'une référence
   exploitable sera disponible.

## 11. Seuils de validation à ratifier

Ces valeurs sont des propositions initiales. Elles doivent être approuvées avant
la validation finale et adaptées aux conventions exactes d'extraction Abaqus.

- Déplacements, erreur L2 relative : cible `< 0,5 %`
- Contrainte de von Mises, petit cas Abaqus : cible `< 2 %`
- PEEQ, petit cas Abaqus : cible `< 5 %`
- Réaction globale : cible `< 1 %`
- Variation entre deux raffinements d'incréments : cible `< 0,5 %`
- Tangente par différences finies : cible `< 1e-6` en erreur relative

Valeurs scientifiques rapportées dans l'article à utiliser comme références :

- RMSE `U2` : `1,32 × 10^-2 %`
- RMSE `epsilon_vM`, 25 partitions directes : `0,361 %`
- MAE `epsilon_vM`, 25 partitions directes : `0,039 %`
- RMSE `epsilon_vM`, 100 partitions avec padding 150 : `0,220 %`
- MAE `epsilon_vM`, 100 partitions avec padding 150 : `0,156 %`

Ces métriques ne doivent pas être interprétées isolément. Une amélioration de
RMSE peut accompagner une carte visuellement plus bruitée aux interfaces.

## 12. Décisions ouvertes

| Décision | Statut | Responsable | Échéance |
|---|---|---|---|
| Fichiers Abaqus exacts de référence | Différé, non bloquant | À définir | Phase B |
| Épaisseur de section EF utilisée dans Abaqus | Différé, non bloquant | À définir | Phase B |
| Convention définitive U/V et x/y | Résolu dans `docs/scientific_contract.md` | Projet | S1 |
| Complétion nodale du bord supérieur | À ratifier ; profil initial explicite `edge-pad` | Revue scientifique | Phase A |
| Facteur de carte d'écrouissage | `380 MPa` nominal, `396 MPa` historique | Revue scientifique | Phase A |
| Traitement des neuf NaN d'écrouissage | Politique explicite à enregistrer | Revue scientifique | Phase A |
| Format des données globales hors mémoire | Résolu : `.npy` memmap | Projet | S4 |
| Machine cible et budget mémoire | Ouvert | À définir | S9 |
| Seuils finaux de parité Abaqus | Différé | Revue scientifique | Phase B |
| Schéma de production 25 ou 100 partitions | Ouvert | Revue scientifique | S11 |
| Métrique de localisation complémentaire | Ouvert | Revue scientifique | S10 |
| Licence du logiciel avant publication | Ouvert | Propriétaire du projet | S12 |

## 13. Registre des validations

| Date | Validation | Commande ou rapport | Résultat | Statut |
|---|---|---|---|---|
| 2026-07-24 | Backend PyPardiso | Import et résolution sparse 2×2 | Backend MKL actif | Réussi |
| 2026-07-24 | Test biaxial 20×20 | `.venv/bin/python fem_pixel.py` | 0 % erreur SVM | Réussi |
| 2026-07-24 | Tangente constitutive | Différences finies | `1e-10` à `7e-9` | Réussi |
| 2026-07-24 | Cas hétérogène | 6×6, quatre incréments | 4 NR/incrément | Réussi |
| 2026-07-24 | Scripts complets | Imports des scripts | `test_config.py` absent | Bloqué |
| 2026-07-24 | Socle de paquet | `pytest --cov=fem_inhouse` | 44 tests, 100 % | Réussi |
| 2026-07-24 | Qualité du nouveau code | `ruff check src tests` | Aucun défaut | Réussi |
| 2026-07-24 | Partitionnement et raccordement | `pytest --cov=fem_inhouse --cov-branch` | 62 tests, 98 % | Réussi |
| 2026-07-24 | Grilles de l'article | Tests `(5,5)` et `(10,10)`, padding 150 | 25/100 cœurs sans trou | Réussi |
| 2026-07-24 | API solveur et noyau EF | `pytest --cov=fem_inhouse --cov-branch` | 82 tests, 94 %, sans avertissement | Réussi |
| 2026-07-24 | Tangente cohérente automatisée | Différences finies du retour plastique | Erreur relative `< 1e-5` | Réussi |
| 2026-07-24 | Compatibilité historique | `.venv/bin/python fem_pixel.py` via pytest | Biaxial SVM/PEEQ réussi | Réussi |
| 2026-07-24 | Workflow reprenable | Tests manifestes, corruption, reprise, raccordement | 87 tests, 95 % | Réussi |
| 2026-07-24 | Exemple réduit tabulé 4×4 | `python -m fem_inhouse validate --nx 4 --ny 4` | erreur SVM `5,84e-6`, PEEQ `2,17e-6` | Réussi |
| 2026-07-24 | Suite complète et seuil CI | `pytest --cov=fem_inhouse --cov-branch` | 92 tests, 95,04 % | Réussi |
| 2026-07-24 | Construction du paquet | `pip wheel . --no-deps` et inspection | cœur, workflow et CLI présents | Réussi |
| 2026-07-24 | Portabilité des scripts historiques | Contrat `.npy`, chemins par environnement | 97 tests, 95,20 %, aucun chemin personnel | Réussi |
| 2026-07-24 | CI sur installation fraîche | GitHub Actions `30086978438` | installation, Ruff et tests verts | Réussi |
| 2026-07-24 | Parité monolithique/partitionnée | Cas homogène 6×6, padding 0 et 1 | `U/S/E/PEEQ` égaux aux tolérances | Réussi |
| 2026-07-24 | Métriques de champs/interfaces | `pytest --cov=fem_inhouse --cov-branch` | 104 tests, 95,36 % | Réussi |
| 2026-07-24 | Robustesse constitutive/globale | 3 trajets plastiques, hétérogène, cutback, réactions | 111 tests, 96,14 % | Réussi |
| 2026-07-24 | Performance 10k/50k/100k | `/usr/bin/time -v fem-inhouse validate` | 5,01/10,60/21,87 s ; 163/557/1061 MiB | Réussi |
| 2026-07-24 | Performance 350k | Vérification mémoire avant lancement | 3,7 GiB disponibles, swap saturé | Reporté |
| 2026-07-24 | Modules maillage/élément/assemblage | Suite complète après extraction | 117 tests, 96,26 % | Réussi |
| 2026-07-24 | Module constitutif public | Suite complète après extraction | 123 tests, 96,32 % | Réussi |
| 2026-07-24 | Module solveur non linéaire | Suite complète et compatibilité historique | 123 tests, 96,33 % | Réussi |
| 2026-07-24 | Diagnostics structurés | Événements `logging` et rapport JSON | Convergence et cutbacks traçables | Réussi |
| 2026-07-24 | Suite après diagnostics | `pytest --cov=fem_inhouse --cov-branch` | 123 tests, 96,66 % | Réussi |
| 2026-07-24 | Typage statique | `mypy src/fem_inhouse` | 25 fichiers, aucun défaut | Réussi |
| 2026-07-24 | Profil par phase 10k hétérogène | `SolverDiagnostics` | 31,948 s, 78 Newton, 0 cutback | Réussi |
| 2026-07-24 | Suite après instrumentation | `pytest --cov=fem_inhouse --cov-branch` | 123 tests, 96,59 % | Réussi |
| 2026-07-24 | Réactions du patch affine | Sommes sur les quatre bords | Signes et résultantes analytiques | Réussi |
| 2026-07-24 | Patch affine en cisaillement | Solution fermée | `U1/U2/E12/S12/PEEQ` conformes | Réussi |
| 2026-07-24 | Suite après patch cisaillement | `pytest --cov=fem_inhouse --cov-branch` | 124 tests, 96,59 % | Réussi |
| 2026-07-24 | CLI partitionnée | Reprise, partition isolée, raccordement | Workflow job array exécutable | Réussi |
| 2026-07-24 | Suite après CLI partitionnée | Ruff, mypy, pytest et `bash -n` | 125 tests, 96,46 % | Réussi |
| 2026-07-24 | Qualité dépôt complet | `ruff check .` | Aucun défaut, scripts historiques inclus | Réussi |
| 2026-07-24 | Wheel typé | `pip wheel . --no-deps` et inspection | `py.typed`, cœur et métadonnées présents | Réussi |
| 2026-07-24 | Recouvrement des localisations | Jaccard, Dice, rappel, précision | Cas identique, partiel et masqué testés | Réussi |
| 2026-07-24 | Suite après métrique de localisation | Ruff, mypy et couverture | 127 tests, 96,55 % | Réussi |
| 2026-07-24 | Provenance de l'article | Manifeste SHA-256 vérifié par test | PDF 2 698 182 octets identifié | Réussi |
| 2026-07-24 | Rapport de comparaison | CLI à seuils pré-déclarés | JSON, carte signée et code retour testés | Réussi |
| 2026-07-24 | Suite après rapport automatique | Ruff, mypy et couverture | 135 tests, 96,70 % | Réussi |
| 2026-07-24 | Assemblage tangent par blocs | A/B hétérogène 10k | -22,4 % tangent, -3,2 % RSS | Réussi |
| 2026-07-24 | Suite après optimisation mémoire | Ruff, mypy et couverture | 143 tests, 96,93 % | Réussi |
| 2026-07-24 | Gouvernance technique | ADR, guide et modèle de PR | Règles numériques explicites | Réussi |
| 2026-07-24 | CI complète distante | GitHub Actions `30089878592` | lint, mypy, wheel et tests verts | Réussi |
| 2026-07-24 | Inventaire scientifique reçu | Formes, statistiques, SHA-256 et scripts de provenance | 4 tableaux `3600×3100` identifiés | Réussi |
| 2026-07-24 | Sous-domaine DIC réel | Centre 10×10, PyPardiso, 10 incréments | Tous champs finis, 0 cutback | Réussi |
| 2026-07-24 | Données scientifiques versionnées | Git LFS + `data/raw/case_study/manifest.json` | 4 tableaux bruts immuables | Réussi |
| 2026-07-24 | Préparation ROI complet | `fem-inhouse prepare-case --nonfinite-policy nearest` | 4 champs canoniques, manifestés, 9 réparations | Réussi |
| 2026-07-24 | Idempotence de préparation | Deuxième exécution sur les mêmes sorties | Empreintes vérifiées, aucune réécriture | Réussi |
| 2026-07-24 | Chaîne DIC réelle 10×10 | Préparation centrale, 25 partitions, raccordement | `U/S/E/PEEQ` finis et complets | Réussi |
| 2026-07-24 | Suite après pipeline DIC | Ruff, mypy, pytest avec branches | 156 tests, 95,26 % | Réussi |
| 2026-07-24 | Clone distant avec Git LFS | Clone isolé, `git lfs pull`, SHA-256, crop 4×4 | Données récupérées et préparées depuis GitHub | Réussi |
| 2026-07-24 | CI distante du pipeline DIC | GitHub Actions `30091651001` | Ruff, mypy, wheel et tests verts | Réussi |
| 2026-07-24 | Sauvegarde exhaustive des partitions | Tests CLI et reprise | `U/S/E/PE/PEEQ/RF` atomiques et empreintés | Réussi |
| 2026-07-24 | Partition article DIC réelle | 100 partitions, padding 150, partition 0 (`510×460`) | 20/20 incréments, 0 cutback, 18 min 08 s, 3,59 GiB RSS | Réussi |
| 2026-07-24 | Intégrité partition article | `validation-report.json`, SHA-256 et contrôles mécaniques | 6 champs finis, bords DIC à `4,16e-17 mm`, équilibre `4,39e-14` | Réussi |
| 2026-07-24 | Comparaison exploratoire DIC/EF | `epsilon_vM` sur la zone résolue | RMSE `0,253 %`, MAE `0,185 %`, corrélation `0,016` | À approfondir |
| 2026-07-24 | MFront nominal sur partition article | Même partition `510×460`, loi analytique non capée, 8 threads MGIS | 20/20 incréments, 10 min 50,08 s, 4 163 308 KiB RSS | Réussi |
| 2026-07-24 | Comparaison longue MFront/tabulé | Champs, temps et mémoire sauvegardés | -40,35 % mur, constitutif 6,905× plus rapide, RSS +10,49 % | Réussi |
| 2026-07-24 | Installation TFEL/MFront et MGIS | Versions et imports depuis `.venv` | TFEL 5.1.0, MGIS 3.1, interface générique active | Réussi |
| 2026-07-24 | Parité constitutive Python/MFront | `validation/reference_data/mfront_material_point_v1/report.json` | L2 contrainte `0,227–0,368 %`, erreur PEEQ max `<3,88e-5` | Réussi |
| 2026-07-24 | Suite après backend MFront | Ruff, mypy, compilation MFront et couverture | 165 tests, 94,25 %, dont 2 tests MGIS réels | Réussi |
| 2026-07-24 | Performance constitutive Python/MFront | 200k points, 20 incréments, 2 répétitions | Python 12,347 s ; MFront série 13,333 s ; MFront 8 threads 3,527 s | Réussi |
| 2026-07-24 | Reproductibilité MFront parallèle | États série/parallèle sur 4 millions de mises à jour | Écarts max contrainte et PEEQ strictement nuls | Réussi |
| 2026-07-24 | Suite après pool MGIS | Ruff, mypy et couverture avec bibliothèque réelle | 167 tests, 94,21 % | Réussi |
| 2026-07-24 | Couplage MFront/Newton | Cas biaxial homogène complet | Parité champs `4,4e-11–1,2e-10`, 0 cutback | Réussi |
| 2026-07-24 | Parité MFront/Python sur DIC réelle | Crop central 10×10, 6 champs sauvegardés | L∞ relatif `4,7e-9–3,3e-4`, 20/20 incréments, 0 cutback | Réussi |
| 2026-07-24 | Performance EF complète MFront | Crop central 10×10, PyPardiso | 0,669 s et 66 Newton contre 1,583 s et 84 Newton | Réussi |
| 2026-07-24 | Suite après couplage Newton | Ruff, mypy, MGIS réel | 172 tests | Réussi |
| 2026-07-25 | Parité MFront natif/J2 3D condensé | Crop DIC 10×10, 20 incréments | 66 Newton chacun, contrainte max `4,804e-08 MPa`, 0 échec local | Réussi |
| 2026-07-25 | Résolution locale de contraintes planes | Résidu GP, itérations et `Cbb` | `2,705e-08 MPa`, 4 itérations max, `cond(Cbb)=1,896` | Réussi |
| 2026-07-25 | Suite architecture constitutive commune | Ruff, mypy et MGIS/MFront réel | 206 tests | Réussi |
| 2026-07-25 | Performance EF des trois backends | Crop DIC 100×100, 20 incréments, 3 répétitions | Python `134,36 s / 248,96 MiB` ; natif `27,03 s / 269,65 MiB` ; condensé `83,43 s / 320,30 MiB` | Réussi |
| 2026-07-25 | Équivalence des trois backends à échelle 100×100 | Comparaison des champs complets sauvegardés | MFront/MFront `2,307e-07 MPa` ; Python/MFront `6,763e-02 MPa`, tous seuils réussis | Réussi |
| 2026-07-25 | Filtre Helmholtz élémentaire | DCT, référence sparse et invariants | 12 tests dédiés, résidu `< 1e-11` | Réussi |
| 2026-07-25 | Workflow de diagnostic non local | Cas synthétique, padding, seuils et non-régression `ell=0` | Sélection cohérente et sorties atomiques | Réussi |
| 2026-07-25 | Campagne Helmholtz partition article 0 | `0–58,88 µm`, cœur `360×310`, padding 150 | RMSE/L2 `-49,45 %`, hypothèse partiellement soutenue | Réussi |
| 2026-07-25 | Suite après diagnostic Helmholtz | Ruff, mypy et MGIS/MFront réel | 230 tests | Réussi |
| 2026-07-25 | Calcul MFront partition de sélection P48 | 402 600 éléments, padding 150, 20 incréments | `1335,97 s`, `7 869 356 KiB`, zéro cutback | Réussi |
| 2026-07-25 | Sélection Helmholtz P48 | Balayage pré-enregistré `0–58,88 µm` | Corrélation `0,2983→0,6160`, IoU top-10 `0,1598→0,2822` | Réussi |
| 2026-07-25 | Calcul MFront confirmation P42 | 402 600 éléments, padding 150, 20 incréments | `1484,55 s`, `8 079 896 KiB`, zéro cutback | Réussi |
| 2026-07-25 | Confirmation Helmholtz tenue à l'écart P42 | `ell=58,88 µm` sans ajustement, seuils pré-déclarés | Corrélation `0,4007→0,7036`, tous critères réussis | Réussi |
| 2026-07-25 | Comportements MFront micromorphiques | Natif PlaneStress et Tridimensional, `Hchi*(p-chi)` | Compilation, métadonnées, signe, tangente et transactions | Réussi |
| 2026-07-25 | Couplage `p ↔ chi` dans Newton | DCT existante, relaxation, commit unique, cutback conjoint | 247 tests avec MGIS réel | Réussi |
| 2026-07-25 | Outil de sélection `Href` | Médiane du tangent de Ludwik sur le cœur plastifié | Tests synthétiques, empreintes et refus d'écrasement | Réussi |
| 2026-07-25 | Référence locale P154 padding 128 | 179 196 éléments, 20 incréments | `793,98 s`, 119 Newton, zéro cutback | Réussi |
| 2026-07-25 | Estimation `Href` sur le cœur P154 | 24 507 éléments plastifiés sur 27 900 | `Href=6547,530617 MPa` | Réussi |
| 2026-07-25 | Smoke micromorphique P154 `alpha=0,5` | 87 164 éléments, norme mixte L∞ | `406,28 s`, 3 cutbacks, tous critères smoke réussis | Réussi |
| 2026-07-25 | Smoke micromorphique P154 `alpha=1` | 87 164 éléments, norme mixte L∞ | `503,04 s`, 2 cutbacks, tous critères smoke réussis | Réussi |
| 2026-07-25 | Smoke micromorphique P154 `alpha=2` | 87 164 éléments, norme mixte L∞ | `226,30 s`, zéro cutback, tous critères smoke réussis | Réussi |
| 2026-07-25 | Validation P154 `alpha=0,5` | 179 196 éléments, padding 128, 20 incréments | `1453,77 s`, zéro cutback, 6/8 critères | Partiel |
| 2026-07-25 | Validation P154 `alpha=1` | 179 196 éléments, padding 128, 20 incréments | `1680,46 s`, zéro cutback, 6/8 critères | Partiel |
| 2026-07-25 | Validation P154 `alpha=2` | 179 196 éléments, padding 128, 20 incréments | `1867,20 s`, zéro cutback, 7/8 critères | Partiel |
| 2026-07-25 | Rejeu micromorphique natif/3D condensé | MFront réel, tangente FD et régression `Hchi=0` | 3 tests ciblés en `1,27 s` | Réussi |
| 2026-07-25 | Validation finale après campagne P154 | Ruff, mypy, MGIS/MFront réel, Sphinx strict | 257 tests en `14,38 s`, HTML et PDF 131 pages | Réussi |
| 2026-07-26 | Sélection scientifique P43 après classement morphologique | Cœur DIC `360×310`, deux bandes diagonales | P43 retenue, aucun calcul lourd lancé | Réussi |
| 2026-07-26 | Benchmark constitutif léger P43 | 446 400 points de Gauss, 14 itérations | `14,357→7,605 s`, RSS `-29,2 %`, champs identiques | Réussi |
| 2026-07-26 | Gate EF complet avant/après | P187 paddée, 39 644 éléments, paramètres identiques | `396,78→273,56 s`, RSS `-12,7 %`, convergence identique | Réussi |
| 2026-07-26 | Validation après optimisation | Ruff, mypy, 271 tests MGIS/MFront, Sphinx strict | HTML et PDF 144 pages | Réussi |
| 2026-07-26 | CSR fixe et phases PARDISO explicites | Même P187, chemin constitutif optimisé inchangé | `273,56→244,67 s`, RSS `-16,7 %`, une phase 11 et 139 phases 22/33 | Réussi |
| 2026-07-26 | J2 symétrique défini positif | Même P187, CSR supérieur et `mtype=2` | `244,67→227,34 s`, PARDISO `-38,0 %`, RSS `-8,7 %` | Réussi |
| 2026-07-26 | Balayage couplé P43 `alpha=1,2,4` | 402 600 éléments, 20 incréments, `ell=58,88 µm` | `26:40 / 29:56 / 36:14`, zéro cutback | Réussi |
| 2026-07-26 | Validation et figures P43 | EVM brute/DIC sur cœur, PEEQ interne | 8/8 critères pour les trois candidats ; `alpha=2` et `4` non dominés | Réussi |
| 2026-07-30 | Bruit temporel et sous-espace de chargement P43 | `diagnose-dic-boundary-loading-subspace` sur 41 états | Bruit `0,047–0,051 px`, affine à `90 %`, un mode à `99,91 %` | Réussi |
| 2026-07-30 | Outlier de bord pré-enregistré à l'état 4 | Même diagnostic, critère `|z| >= 3` | `z = 0,13` et `1,66` ; hypothèse réfutée | Échec enregistré |
| 2026-07-30 | Instrumentation Newton P43 | `run-dic-multistep-mechanics --record-newton-trace`, 28 records | Prédicteur élastique résolu sur un buffer CSR écrasé | Cause racine trouvée |
| 2026-07-30 | Aliasing du buffer d'assemblage | `FixedCSRAssembler.assemble` sur deux tangentes | `A is B` vrai, premier résultat muté | Défaut confirmé |
| 2026-07-30 | Non-régression du correctif | Cas analytique réduit avant/après, 423 tests MFront | Sortie identique bit à bit | Réussi |
| 2026-07-30 | Rejeu P43 histoire mesurée 40 états | `run-dic-multistep-mechanics --mode measured` | Complet en `68,1 min`, 65 incréments, 3 cutbacks | Réussi |
| 2026-07-30 | Contrôle proportionnel 40 incréments | Même workflow `--mode proportional` | 40/40 incréments, 0 cutback, `31,0 min` | Réussi |
| 2026-07-30 | Dépendance au trajet sur PEEQ | `compare-path-dependence`, cœur `360×310` | L2 `15,82 %`, contrôle `0,20 %`, structure de bande `13,11` | Réussi |
| 2026-07-30 | Trajets mesuré/proportionnel face à la DIC | `replay-dic-observation`, deux profils, observation symétrique | Aucune métrique au-delà de sa marge | Indiscernable |
| 2026-07-30 | Filtrage modal du bord à 3 modes | `filter-dic-boundary-history --rank 3` | Retire `0,00972 px`, soit `5,3×` sous le bruit | Réussi |
| 2026-07-30 | Mécanique sur histoire filtrée | `run-dic-multistep-mechanics --mode measured` | 40/40, zéro cutback, 245 itérations en `34,8 min` | Réussi |
| 2026-07-30 | Pénalisation contre élimination | Cas réduit, `k` de `1e8` à `1e12` | Erreur en `1/k` puis limite de conditionnement | Réussi |
| 2026-07-31 | Reproduction du champ archivé par profil | `compare-profile-reproduction`, 4/1 contre 8/3 | `1,673 %` contre `1,738 %`, rapport `1,04` | Ne discrimine pas |
| 2026-07-31 | Calibration du ressort de pénalisation | Principe de l'écart, chaîne analytique | `k/K_ref = 2,7`, écart = misfit à `5 %` | Réussi |
| 2026-07-31 | Excès de PEEQ moyen après filtrage | Déciles et aire active, données archivées | Redistribution ; confondant à `5,8 %` seulement | Expliqué |

## 14. Journal des mises à jour

### 2026-08-04 — Correction Broyden de la jacobienne hourglass : rejetée, et on sait pourquoi

Cahier des charges §15 à §35, câblage solveur inclus. **Verdict : rejetée.**
Trois falsificateurs se déclenchent. `jacobian_correction` reste à `none`, et
CPS4R-AS conserve les 47 itérations que ce travail visait à réduire.

Les entrées CPS4R-AS qui précèdent ce travail — dérivation QUAS4, qualification
élémentaire, campagne SRIX, variante décalée non convergente — ne sont pas dans
ce journal ; elles sont dans `validation/cps4r_assumed_strain_report.md` et
`validation/cps4r_assumed_strain_campaign2_preregistration.md`.

- **§15-18 câblés** : protocole `NonlinearJacobianCorrection`, registre,
  `NoJacobianCorrection` comme objet plutôt que branche `None`, mémoire par
  élément. Les trois règles de transaction du §17 sont tenues : purge en début
  d'incrément, purge au cutback, et paires construites **uniquement** en tête
  d'itération — là où `u` est l'état que l'itération précédente a accepté. Un
  essai de recherche linéaire est une sonde, pas un itéré, et n'atteint jamais
  `observe`.
- **La correction ne touche que la matrice.** `R_I` est formé avant, et n'est
  jamais reconstruit à partir d'elle. C'est ce qui rend le nombre d'itérations
  le seul observable prévu.
- **Le cas préinscrit se reproduit exactement** : CPS4 à 37 itérations,
  CPS4R-AS à 47, sur SRIX Bunge (35, 20, 15), 12×12, huit incréments.
- **F3 se déclenche sur les trois mémoires** : 50, 57, 64 pour `m = 1, 3, 5`.
  Plus de paires sécantes, plus d'itérations, et de façon monotone. Même ordre
  à une tolérance de `1e-8` (56 sans correction, puis 59, 60, 71) et sur un
  maillage 6×6 à quatre incréments.
- **F1 se déclenche sur `m = 3` et `m = 5`**, et il a fallu le mesurer pour le
  savoir. `m = 1` donne `6,6e-10` à une tolérance de `1e-6` et `7,8e-12` à
  `1e-8` : l'écart suit la tolérance, c'est une différence de trajet dans la
  boule de convergence. `m = 3` et `m = 5` donnent `2,26e-5` **aux deux
  tolérances**, inchangé après un resserrement d'un facteur cent. Les deux
  calculs accélérés convergent donc vers un équilibre réellement différent du
  même problème discret. L'explication que suggèrent les indices — la loi
  cristalline indépendante du temps admet des équilibres voisins d'ensemble de
  systèmes actifs différent — n'est **pas démontrée**, et reste consignée comme
  point ouvert.
- **Le §23 explique l'échec.** Sur un élément, loi réintégrée à chaque
  évaluation : la correction annule son défaut sécant à `1,6e-15` le long des
  directions sur lesquelles elle a été ajustée, et **dégrade** la prédiction le
  long d'une direction fraîche — `0,99` devient `1,52`, et elle n'améliore que
  41 % des directions d'essai. Robuste sur contractions `0,3 / 0,6 / 0,8` et
  mémoires `1 / 3 / 5` : `m = 1` est neutre hors échantillon et quasi neutre
  dans le solveur, `m = 5` dégrade jusqu'à trente fois et est le pire des trois.
  Le diagnostic élémentaire et la campagne solveur concordent en ordre et en
  amplitude.
- **La raison est dans la nature du terme.** `(df_stab/dC)(dC/du)` n'est pas un
  opérateur linéaire fixe que cinq paires sécantes identifient : `C` est la
  tangente algorithmique d'une loi **indépendante du temps** et saute avec
  l'ensemble des systèmes actifs. Ajuster un `2×5` sur une suite contractante
  où `C` a changé de façon discontinue produit une matrice qui satisfait toutes
  les conditions stockées et ne décrit rien. La résolution de norme de
  Frobenius minimale fait correctement son travail ; c'est la prémisse qu'un
  ajustement sécant puisse capter ce terme qui est fausse.
- **Ce qui n'est pas réfuté** : la tangente assumed-strain *est* inconsistante,
  de 370 % sur la stabilisation, et c'est bien ce qui coûte les dix itérations.
  Le diagnostic directionnel le confirme indépendamment — la jacobienne réduite
  de base se trompe de `0,85` à `1,2` en erreur relative de prédiction sur la
  vraie loi cristalline.
- **Vitesses mesurées, mais non décisives.** CPS4R-AS sans correction est à
  `2,44` sur le constitutif et `2,10` sur le total ; la borne constitutive de
  `3,5` n'était donc jamais à portée, et c'est précisément le manque que ce
  travail devait combler. `m = 1` affiche un gain constitutif *supérieur* à la
  référence tout en faisant trois itérations de plus, ce qui est impossible si
  les temps étaient exacts : la répétabilité de cette machine est donc de
  l'ordre de 15 à 20 % même sur des médianes de cinq. Le verdict repose sur les
  critères déterministes.
- **Le code est conservé, éteint par défaut** : c'est ce qui rend le résultat
  négatif vérifiable, et `scripts/diagnose_broyden_directional_prediction.py`
  est réutilisable contre tout candidat futur.

#### Revue du CdC : trois objections vérifiées plutôt qu'appliquées

- **« Ce n'est pas good Broyden » — retenue.** La théorie de Broyden porte sur
  une jacobienne **carrée** du résidu global, mise à jour depuis des paires
  globales. Ce module construit une régression multisécante de moindre
  changement **locale** et rectangulaire `2×5`, assemblée ensuite. Renommé, et
  les deux modules sont marqués `experimental_falsified`.
- **« La correction locale dégrade la matrice globale » — retenue, et
  décisive.** Mesure sur les paires que le solveur produit réellement,
  `s = u_{k+1}-u_k` et `y = R_{k+1}-R_k` : défaut sécant global moyen `0,0776`
  sans correction, puis `0,1335`, `0,1808`, `0,3615` pour `m = 1, 3, 5`. Les
  conditions sécantes **locales** sont satisfaites à `1e-15` pendant que le
  défaut **global** est multiplié par 5,3, et la fraction de pas améliorés
  (55 %, 33 %, 26 %) suit exactement les itérations (50, 57, 64). C'est
  l'énoncé le plus propre de l'échec.
- **« Mélange déformations/longueurs » — retenue à moitié.** Le
  conditionnement, oui : `cond(S) = 1,1e4` contre `23,6` une fois les
  amplitudes divisées par `sqrt(aire)`, et `cond(T) = 1537` contre `1,54`.
  Corrigé, `modal_coordinates` prend un `length_scale` valant `sqrt(aire)` par
  défaut. **L'invariance aux unités, non** : à rang plein `(DT)^+ = T^+D^{-1}`
  exactement, et les facteurs se compensent dans `K_B = H^T ΔG T`. Mesuré,
  millimètres contre micromètres en coordonnées **non pondérées** : `3e-15`.
  Une première version de ce contrôle annonçait 41 % et était fausse — tangente
  élastique constante, donc force exactement linéaire, `Z` nul à l'arrondi près
  et correction de bruit pur. Et la nondimensionnalisation ne change pas le
  verdict : hors échantillon, `0,99` devient `0,84`, `1,05`, `1,52`.

La suggestion de la revue — un Broyden **inverse global** sur `R(u)=0`, à
mémoire limitée, avec redémarrages, sauvegardes et repli sur la direction de
Newton — a été implémentée puis qualifiée négativement sur le cas SRIX
enregistré. Avec la même line-search exacte, `m=1` reste à 47 itérations et
`m=3` passe à 49 ; les 31 directions `m=1` proposées passent toutes les
garde-fous et sont utilisées à plein pas. Le temps `m=1` (3,80 s contre 4,21 s)
ne constitue donc pas un gain algorithmique exploitable, et la référence
historique sans line-search reste à 2,53 s. Statut :
`qualified_negative_result`, code conservé pour reproductibilité, désactivé par
défaut. La priorité revient à la campagne 2 de `assumed_strain_energy`.

Route restante pour les dix itérations : fournir le terme manquant plutôt que
l'ajuster — un vrai `dC/du`, ou une formulation décalée dont la matrice est la
dérivée exacte de la force qu'elle emploie. La variante décalée est implémentée
et **ne converge pas** (`f95d351`) ; elle reste ouverte pour une autre raison.

### 2026-08-03 (9) — Qualification SRIX : le modèle est prêt à calibrer, les paramètres ne le sont pas

Cahier des charges §1 à §19. **Conclusion : cas B.** La formulation et son
intégration sont qualifiées ; aucun paramètre 316L n'est identifié.

- **§3** L'en-tête MFront décrivait encore la *première* implémentation — `Deq`
  lu sur `deto`, tangente amputée d'un terme de rang un. Un lecteur en aurait
  conclu que la tangente était sciemment fausse. Réécrit avec la mesure qui a
  tranché : `2e-2` à `4e-1` depuis `deto` contre `7e-7` depuis les inconnues.
  `srix_reference_stress` → `srix_overstress_modulus_from_meric`, alias déprécié
  conservé ; la doc dit maintenant que l'équation (16) est **une** route vers `R`
  et pas sa définition
- **§4 §5 §6** Tous les paramètres sont des `@Parameter` : **rien à recompiler**,
  un jeu s'applique par `setParameter`. Sept jeux enregistrés, provenance **par
  groupe de paramètres** avec les cinq statuts, et un test qui assène qu'**aucun
  jeu ne revendique une identification**. Les constantes élastiques actualisées
  n'ont pas de citation : le registre écrit « primary publication not supplied
  and deliberately not invented » plutôt que d'accrocher un papier vraisemblable
- **§7** Matrice d'interaction **dérivée de la géométrie**, pas de l'ordre des
  nombres, et vérifiée entrée par entrée contre `mfront-query`. Trouvaille :
  MFront **scinde la jonction glissile en deux rangs** selon quel système peut la
  faire glisser, donc la matrice de rangs **n'est pas symétrique** et le résultat
  numérique ne l'est que parce que le coefficient publié unique est écrit dans
  les deux cases. J'avais d'abord affirmé que le littéral de la galerie TFEL
  différait du nôtre : **c'était faux**, il est identique case par case
- **§8** Solution analytique dérivée puis vérifiée : `σ = √6·τ0 + (6/8)·R`,
  reproduite à `1e-16`, 8 systèmes actifs, les quatre autres exactement nuls. La
  surcontrainte relative sur ce plateau **est** exactement `O_R`
- **§9** Dissipation positive partout, indépendance temporelle **bit à bit** sur
  pseudo-temps uniforme, rampé et aléatoire. **Trouvaille conservée** : sous
  ~20 incréments, un renversement ne produit **aucun glissement inverse** — pas
  une grosse erreur, une autre solution
- **§10 §13** Les 24 symétries laissent la réponse axiale invariante à `1e-9` et
  le spectre des glissements à `1e-12`. **Point ouvert assumé** : la carte
  d'indices entre ma permutation et l'ordre MFront ne se réduit à aucune règle
  testée ; propriété donc affirmée sous forme invariante par permutation plutôt
  qu'ajustée à la règle qui passait. Contraintes hors plan sous `1e-6 MPa` sur
  4 orientations × 4 chargements × 3 valeurs de `R`
- **§11** R compte **plus hors axe** : 7,7 % d'écart à `[001]`, 18,6 % à `[123]`,
  et il change le **nombre de systèmes actifs**. Donc `R` ne peut pas être
  calibré sur `[001]` puis transféré — contrainte inscrite dans la préinscription
- **§14** Les douze glissements remontent jusqu'à `FEMResult` et aux archives ;
  **PEEQ reste à zéro** et un test l'assène
- **Trois défauts trouvés.** `mgis.load` **ne donne pas un comportement privé** :
  deux batches avec des jeux différents se seraient silencieusement écrasés —
  paramètres réappliqués avant chaque intégration. Le module d'Young compilé était
  faux de `1,3e-7`, donc « appliquer le jeu historique » n'était pas neutre. Et
  trois erreurs dans mes propres tests, chacune ressemblant d'abord à une loi
  cassée : composition `Q Sᵀ` au lieu de `S Q`, lecture des contraintes sans les
  ramener en repère global, tangente évaluée à incrément nul
- **Non fait, et non simulé** : les cas cuivre et PWA1489 de l'article demandent
  leurs jeux de paramètres, absents. §16 interdit d'inventer les coefficients
  manquants
- 1070 tests, ruff, mypy et Sphinx strict propres


### 2026-08-03 (8) — Remise à plat du dépôt, et travail direct sur `main`

- **Consigne : plus de branches ni de PR.** Personne d'autre ne travaille ici, et
  la fusion est de toute façon refusée par le classificateur de permissions, donc
  chaque PR se terminait par une commande rendue à l'utilisateur. Le commit
  explicatif et l'entrée `Claude.md` remplacent la description de PR comme trace
- **`main` était déjà à jour**, mais une branche distante ne l'était pas :
  `agent/docs-evidence-hardening`, **15 commits absents de `main`**, que j'ai
  failli supprimer avec les autres. Vérification faite, sa PR #1 avait été
  **fermée comme supplantée** par `dadec1b7` — présent dans `main` — et ses
  fichiers ont bougé de 17 à 20 commits depuis le fork. La fusionner
  ressusciterait une documentation antérieure à la conclusion P43. Elle est donc
  à supprimer, pas à fusionner
- Cinq autres branches entièrement contenues dans `main`, supprimées localement.
  **La suppression côté distant est refusée par les permissions** : six branches
  restent sur `origin`, c'est cosmétique et ça vous revient
- `BCC_CrystalPlasticity_PhaseFieldFracture.mfront` versé dans `mfront/`,
  **non compilé** : le script de build utilise une liste explicite, pas un glob,
  donc l'ajouter ne l'active pas. Loi non relue, non testée, sans point d'entrée
  Python — c'est une mise à l'abri sous contrôle de version, pas une intégration
- `.codebase-memory/` **sorti du suivi git** (`git rm --cached`) et ignoré :
  cache d'index régénérable, spécifique à la machine, dont le `graph.db.zst` de
  `5,2 Mo` mettait un diff binaire illisible dans chaque commit et invitait les
  « conflicted copy » du dossier synchronisé. Son `.gitattributes` auto-généré
  (`merge=ours`) existait précisément pour contenir ce problème ; il devient
  sans objet
- Fichiers de reprise de session et `kinematics_extension_v1.diff` ignorés sans
  être supprimés



### 2026-08-03 (7) — Couverture : trois modules qui comptent, et pourquoi 85 % est hors d'atteinte

- **Choix des cibles par la valeur d'appropriation, pas par le coût en lignes.**
  `qualify_roi` : **0 % → 100 %**. Aucun test, alors qu'il décide si un ROI vaut
  dix heures de matrice et que son verdict est cité dans
  `validation/roi_qualification_results.md`. `campaign_access` : **71 % → 100 %**,
  porte unique par laquelle six workflows lisent les campagnes archivées, et
  **toutes** ses lignes non couvertes étaient des refus (partition incomplète,
  empreinte de manifeste discordante, champ réécrit après coup, NaN archivé
  d'un calcul divergé). `dic_partition_selection` : **60 % → 96 %**, le score
  qui remplace la sélection manuelle de ROI, dont toute la branche de notation
  n'était jamais exécutée
- **Quatre de mes attentes étaient fausses ; les tests enregistrent le vrai
  comportement.** Le score de bande est **exactement invariant par recalage
  affine** du champ — il classe la forme et ne peut pas comparer l'intensité de
  localisation entre ROI. Le seuil q85 **fixe l'aire détectée à ~15 %** quelle
  que soit la largeur réelle de la bande, donc aspect et aire sont identiques
  d'une largeur à l'autre et seul le contraste sépare, **non monotonement** : un
  score plus bas ne veut pas dire une bande plus étroite. Un champ uniforme
  score zéro par la borne de contraste, pas par le filtre d'aire
- **Une observation sur le filtre ROI** : `dic_band_is_resolved` exige `73,5 px`
  alors que l'estimateur de largeur intégrale sature vers `35 px` — plafond déjà
  mesuré et documenté dans le rapport de validation. Cette condition ne peut donc
  passer pour aucun champ. Ça n'enlève rien à la conclusion archivée, qui repose
  sur les conditions directionnelles (`1,06` contre `1,33` requis), mais elle ne
  porte aucune information et ne devrait pas être comptée parmi les motifs de
  rejet
- **Le verdict sur la barrière à 85 % : elle n'est pas atteignable par des tests
  qui portent leur poids.** Mesuré : `67,16 % → 70,88 %`. Il resterait à couvrir
  **~2317 des 4780 lignes et branches manquantes**, dont **70 % (3359) vivent
  dans 14 modules pilotes de campagne** — `joint_nonlocal_identification` à lui
  seul en concentre 871. Ces modules lisent des archives de plusieurs Go, lancent
  des calculs d'une heure et écrivent rapports et figures. Les tester
  sérieusement demande des jeux de données de campagne ; les tester
  superficiellement veut dire tout simuler, ce qui ne documente rien et fige des
  détails d'implémentation
- 906 tests avec MFront, ruff et mypy propres



### 2026-08-03 (6) — Qualification CPS4R : elle échoue, et le diagnostic est réfuté

- **Préinscription d'abord** (`validation/cps4r_qualification_preregistration.md`),
  seuils dérivés et non choisis : le profil de référence archivé reproduit le
  champ à `1,673 %`, exiger que la formulation élémentaire n'inflate ça que de
  `10 %` en quadrature donne `0,766 %`, arrondi à `0,5 %` sur PEEQ et dix fois
  plus serré sur le déplacement
- **Verdict : CPS4R n'est pas autorisé, aucun β n'est recommandé.** A1 échoue
  partout — erreur PEEQ de `1,9 %` à `10,1 %` contre CPS4 sur le cas J2
  hétérogène `32×32`, `1,35 %` en contrainte sur le cas SRIX incliné
- **F3 a tiré dans toutes les configurations** : chacune passe le seuil de `1 %`
  d'un ordre de grandeur en ratant la borne de précision d'un facteur 4 à 20.
  Et la lecture spatiale ne tient pas non plus : sur 1024 éléments, corrélation
  énergie hourglass ↔ erreur PEEQ `r = 0,033`, énergie ↔ PEEQ `r = 0,066`. Les
  seuils `1 % / 5 %` sont **retirés** de la doc et du contrat, conséquence
  enregistrée à l'avance
- **H2 réfutée dans sa direction, et c'est le résultat physique intéressant** :
  baisser β **rapproche** de CPS4 au lieu d'en éloigner. `β=1` garde la référence
  élastique complète alors que la tangente constitutive s'effondre, donc les
  modes hourglass restent élastiquement raides pendant que tout le reste plastifie
  — `β=1` sur-raidit exactement là où CPS4 s'assouplirait. `β=0,1` tombe six fois
  plus près sur le déplacement
- **Deux faits qui vont dans l'autre sens, conservés** : le coût tient
  (`3,7×` à `4,8×` sur le constitutif, `1,9×` à `2,9×` sur le total, le meilleur
  étant le cristal) ; et l'écart de déplacement est **30 à 200 fois sous le bruit
  DIC**. L'échec est de cohérence numérique, pas de physique mesurable
- **Deux défauts de mon propre script, corrigés avant tout enregistrement** :
  les lois cristallines laissent PEEQ à zéro, donc la première version comparait
  un champ vide à un champ vide et **annonçait un score parfait sur le critère le
  plus important** — A1 bascule maintenant sur la contrainte et refuse une
  référence sans aucun des deux champs (test de non-régression écrit). Et la
  perturbation du cas cristallin valait `2,5×` le chargement, ce qui envoyait les
  deux formulations en cutback et fabriquait `62 %` d'écart dû à des trajets
  divergents, pas à l'élément
- **Un chronométrage unique n'est pas une mesure** : le premier balayage donnait
  `0,91×` à `β=1`, soit CPS4R plus lent que CPS4 à nombre d'itérations de Newton
  identique. C'était du bruit machine, et j'ai failli le publier. Médiane de cinq
  résolutions désormais
- `pythonpath = ["."]` ajouté à pytest : le test importait `scripts`, ce qui
  passait sous `python -m pytest` et **cassait sous le `pytest` nu de la CI**
- Reste ouvert, listé dans le rapport : étude de convergence en maillage,
  stabilisation bâtie sur la tangente courante plutôt que sur la référence
  élastique figée (c'est là que pointe le résultat sur β), et un estimateur qui
  prédise réellement l'écart
- 843 tests, ruff, mypy et Sphinx strict propres

### 2026-08-03 (5) — Documentation §18 de l'intégration réduite, sur la PR #3

- **Travail porté sur `agent/fix-cps4r-qualification`, pas sur `main`.** La PR
  #3 (brouillon) traitait déjà §16 et la propagation des diagnostics ; écrire la
  doc sur `main` aurait garanti un conflit. Ce qu'elle apporte et qu'il faut
  connaître : le dénominateur du ratio hourglass était `|F_all @ u|`, un produit
  scalaire final qui vaut deux fois l'énergie élastique sur un trajet linéaire
  et n'a aucun sens de travail après plastification ; il est remplacé par un
  travail interne trapézoïdal accumulé sur les seuls incréments acceptés, exposé
  comme `INTERNAL_WORK`. Elle ajoute aussi le cas **non affine** qui manquait :
  tous mes tests étaient affines, donc l'énergie hourglass y était nulle *par
  construction* et le balayage de β ne prouvait rien
- **`docs/explanation/reduced_integration_hourglass.md` complété** : fonctions de
  forme bilinéaires, matrice `B` en cisaillement ingénieur, les deux règles de
  quadrature et `Σw_g = 4`, exactitude du `2×2` sur maillage régulier (`J`
  constant, intégrande au plus quadratique — c'est **ce qui rend β=1 exact et non
  heuristique**), décomposition des 8 ddl en 3 rigides + 3 déformation constante
  + 2 hourglass, identité `Σ h_a N_a = ξη` dont les deux dérivées s'annulent au
  centroïde, comptage de rang `3 → 5` et `rang(K_hg) = 2`, condensation
  `C^ps = C_aa − C_ab C_bb⁻¹ C_ba` écrite avec la partition Kelvin `a = {0,1,3}`,
  `b = {2,4,5}` et les facteurs `(1,1,1/√2)`
- **Sept références vérifiées une par une contre l'API Crossref** avant d'être
  écrites — titre, revue, volume, pages, année, auteurs. Flanagan & Belytschko
  1981 (l'origine du contrôle en raideur), Belytschko *et al.* 1984, Kosloff &
  Frazier 1978, Belytschko & Bachrach 1986, Zienkiewicz, Taylor & Too 1971, plus
  Belytschko & Bindeman 1993 et Simo & Rifai 1990 comme alternatives non
  retenues. Trois manuels et le précédent Abaqus (`ALLAE`)
- **Réserve de fond écrite dans la doc et dans le contrat** : `r_hg` compare une
  grandeur **d'état** (l'énergie stockée au pas final) à une grandeur **de
  trajet** qui inclut la dissipation plastique. Après plastification le
  dénominateur croît, le numérateur non : le ratio baisse quand le trajet
  s'allonge, à comportement élémentaire inchangé. Les seuils 1 % / 5 % dérivent
  donc avec le nombre d'incréments, et deux runs ne sont comparables que sur des
  trajets comparables. De plus il n'est évalué qu'à l'état final : une excitation
  transitoire qui se décharge est invisible
- **Job `quality` de la CI** : `test_warp_is_deterministic_bit_for_bit` était le
  seul test non marqué à atteindre `cv2`. Le marqueur seul ne suffisait pas —
  `quality` lance `pytest` sans `-m`, donc un test marqué **s'exécute quand
  même**. Le motif du dépôt est marqueur **+** `pytest.importorskip("cv2")` ;
  c'est ce qui a été appliqué. Le job `measurement`, qui échoue sur tout skip,
  passe de 7 à 8 tests sélectionnés sans skip
- Reste sur la PR : la barrière de couverture (`67,19 %` contre `85 %`), très
  antérieure ; figure de diagnostic spatial (§13) ; balayage J2 non affine
  plastique, comparaison SRIX inclinée et benchmark de temps réel avant que
  CPS4R puisse remplacer CPS4 dans une campagne
- 833 tests, zéro échec, zéro skip avec la bibliothèque MFront ; ruff, mypy et
  Sphinx strict propres

### 2026-08-03 (4) — Tangente hourglass mesurée, et CPS4R validé en cristallin

- **Le repli silencieux est supprimé.** `reference_in_plane_tangent_mpa()` est
  exposé par les deux ponts MFront ; le backend `python` garde `C_ps`, qui EST
  sa tangente élastique puisqu'il est isotrope par construction ; tout autre
  backend sans la méthode **lève** au lieu de retomber sur l'isotrope
- **La tangente n'est pas reconstruite, elle est mesurée** : un incrément de
  déformation nul depuis l'état committé laisse tout comportement dans sa
  branche élastique (le cristal prend sa branche gardée sans glissement), donc
  la tangente condensée retournée EST l'opérateur élastique, déjà tourné dans
  le repère global par l'orientation du batch. La sonde est annulée par
  `revert`. Rebâtir depuis `C11/C12/C44` aurait dupliqué l'élasticité qui vit
  déjà dans MFront
- Contrôles : à l'identité `G = C44 = 122000` et
  `C11_ps = 197000 − 125000²/197000 = 117685,3`, soit la condensation
  analytique exacte ; le J2 condensé retombe sur `plane_stress_elasticity` à
  `2,9e-11` ; à 30/45/60 le couplage extension–cisaillement apparaît, ce qu'une
  référence isotrope ne peut pas produire
- **§15.9 fait** : SRIX en CPS4R, orientations identité et inclinée. Un seul
  état constitutif par élément, facteur 4 sur les points matériels, accord avec
  CPS4 à `1e-9` sur le déplacement, contraintes hors plan sous `1e-6`, énergie
  hourglass sous `1e-9`, zéro cutback
- `SolverDiagnostics` expose `element_formulation`,
  `gauss_points_per_element`, `constitutive_material_point_count`,
  `hourglass_energy` et `hourglass_energy_ratio`
- Reste : figure de diagnostic spatial (§13), benchmark de temps réel et cas
  excitant réellement les modes pour choisir β (§16), documentation (§18)
- 826 tests, ruff propre

### 2026-08-03 (3) — CPS4R et contrôle hourglass en raideur : algèbre élémentaire

- **Livré** : abstraction `QuadratureRule` (§6), règles CPS4/CPS4R, stabilisation
  `K_hg = β(K_ref^4pt − K_ref^1pt)` (§9), tangente de référence anisotrope
  validée (§11), et le découplage de `assembly.py` des constantes globales
  (§19.3). 30 tests élémentaires, tous analytiques
- **Pourquoi la forme en différence évite tout projecteur** : tout champ que la
  règle à un point intègre exactement — corps rigides et champs affines, à
  déformation constante — contribue identiquement aux deux termes et donc rien
  à leur différence. Mesuré : force hourglass `3e-17`, énergie `1e-23` sur
  traction, biaxial, cisaillement, rotation et translation
- Équivalence β=1 : **`7,2e-17`** en relatif, contre `1e-11` demandé. C'est une
  identité algébrique, pas une coïncidence — mais elle casse dès que la
  stabilisation est bâtie sur un autre opérateur que le matériau
- Rang 3 non stabilisé → 5 stabilisé, noyau exactement les 3 corps rigides,
  `K_hg` de rang 2 : les deux modes hourglass et rien d'autre
- **La tangente de référence isotrope est fausse pour un cristal orienté** :
  plus de 10 % d'écart sur `K_hg` à 30°, qu'aucun ratio énergétique global ne
  révélerait. D'où sa validation (forme, symétrie, définie positive) et le refus
  d'une dissymétrie réelle, seul le bruit étant symétrisé
- **NON livré, le câblage solveur** : option `element_formulation` en
  configuration (§5), intégration matérielle réduite à un point par élément
  (§7), force hourglass dans le résidu et les réactions (§9), énergie hourglass
  et diagnostics (§12-13), refus `cps4r + non-local` (§4), tests §15.7-15.9,
  benchmark et étude de β (§16), documentation (§18). L'élément existe et est
  vérifié ; il n'est pas encore sélectionnable depuis une configuration
- 803 tests, ruff propre. Non poussé

### 2026-08-03 (2) — Pont MGIS générique et anisotropie cristalline

- Le pont 3D est piloté par le catalogue : plus aucune référence obligatoire à
  `InitialYieldStress`, `EquivalentPlasticStrain` ni `YieldSurfaceRadius` dans
  le cœur. `MFrontVariableSpec.component_count` déclare les familles à 12
  composantes plutôt que de parser `PlasticSlip[7]`
- Nouveau module `core/crystal_orientation.py`. **Convention** :
  `Q_global_to_material`, donc `eps_c = Q eps_g Qᵀ`. **MGIS attend la
  transposée** (matériau→global, aplatie ligne-major) — mesuré contre une
  rigidité cubique tournée à la main, pas contre MGIS. Un seul point du code
  applique cette transposition
- **Défaut de repère trouvé et corrigé** : la condensation s'amorçait sur
  `s0.gradients`, devenu cristallin depuis l'ajout des rotations, et le
  consommait comme global. Invisible à l'identité (les repères coïncident).
  Ce qui l'a démasqué : raffiner les incréments faisait échouer **plus tôt**
  (4,50e-3 → 4,05e-3 → 3,75e-3), l'inverse d'un pas trop grand
- **Piège du champ affine** : `reduced_biaxial_case` impose un champ affine sur
  les quatre bords, qui est solution exacte pour *tout* matériau homogène. Le
  déplacement EF ne peut donc pas dépendre de l'orientation ; la contrainte, si.
  Un test qui aurait affirmé le contraire aurait affirmé une fausseté
- Aucune PEEQ-J2 fabriquée : le champ reste nul pour une loi cristalline, et le
  couplage micromorphique refuse explicitement ces comportements
- Reste ouvert : les 12 composantes ne remontent pas jusqu'à `FEMResult` (elles
  sont accessibles sur le batch) ; orientations EBSD par point de Gauss non
  faites, mais l'API accepte déjà `(n_points, 3, 3)`
- 773 tests, ruff et mypy propres, doc en `-W`. Non poussé

### 2026-08-03 — Loi cristalline SRIX de Forest-Rubin, indépendante du temps

- Deux comportements MFront ajoutés : `Fcc316LMericCailletaud` (galerie TFEL,
  référence de comparaison) et `Fcc316LForestRubinSrix`. **Aucune loi
  cristalline n'existait dans le dépôt** avant cela — MC n'avait vécu que dans
  un bac à sable de chiffrage, or les §9.7 et §10 du cahier des charges en font
  la référence de non-régression, d'où son ajout
- **Indépendance au temps acquise** : même chemin, `dt` varié d'un facteur 1e6,
  écart `0,000e+00` bit à bit sur contraintes, variables internes **et**
  tangente. Contrôle MC obligatoire : il dérive bien de `9,1 MPa`
- **Δε̄ est construit depuis les inconnues, pas depuis `deto`.** Lu sur `deto`,
  il est constant pour le système local et la brique `StandardElasticity`
  produit une tangente amputée d'un terme de rang un : exacte en élastique,
  fausse de `2e-2` à `4e-1` dès que le glissement s'active. Comme
  `feel = deel - deto + Σ dg·m`, les deux lectures sont identiques à
  convergence ; passer par `deel` et `dg` fait descendre l'écart à `7e-7`, soit
  le plancher de troncature des différences finies elles-mêmes
- Deux pièges de jacobien trouvés **par la mesure**, pas par relecture : le
  crochet de Macaulay ne se dérive pas tout seul ici (la pente `Δε̄/R` est
  constante, contrairement à `n·v/f`), et l'ancien garde-fou numérique
  `f > 1.1 K` de MC ne doit pas être recopié. Ce seuil n'était pas une partie de
  la loi de Norton et a depuis été supprimé : la robustesse des pas appartient
  à la line-search et au cutback sélectif communs.
- `R = 18,7819100705 MPa` par l'équation (16), pour `K=12`, `n=11`,
  `1e-3 s⁻¹`. **Transposition analytique, pas identification** : la vitesse de
  référence est un argument obligatoire sans défaut, car celle de notre essai
  DIC n'est pas documentée. Accord `0,32 %` en `[001]`, désaccord `7,1 %` en
  `[111]` et `14,2 %` en `[123]`, tous deux assertés
- **Reste ouvert : le §8/§9.6 contraintes planes n'est pas exercé.** Le pont 3D
  (`mfront.py:886-908`) code en dur `EquivalentPlasticStrain` et
  `YieldSurfaceRadius` et les `assert` non nuls ; une loi cristalline n'a ni
  l'un ni l'autre, donc `MFront3DCondensedPlaneStressBatch` refuse de la
  charger. Le catalogue déclare déjà les bonnes liaisons, il reste à les lui
  faire lire
- 719 tests, aucun skip ; ruff propre ; doc en `-W`

### 2026-07-31 — L'excès de PEEQ après filtrage est une redistribution

- Question fermée : le `+0,64 %` de PEEQ moyen que j'avais consigné sans
  explication est une **redistribution**, pas une hausse d'amplitude
- Le confondant de sous-incréments (40 contre 65) n'en explique que `5,8 %`,
  sous la borne de `20 %` fixée d'avance. Mon affirmation antérieure qu'il était
  « du bon ordre » avait été écrite **sans mesure** et est corrigée dans le
  document du filtre
- Signature : déciles 3–4 négatifs, `217` éléments plastifiés en moins, quatre
  déciles supérieurs portant `94,6 %` de l'excès, queue extrême rabotée
  (`q0,9999` à `-3,55 %`, maximum à `-1,65 %`)
- Lecture : le contenu de bord sous le plancher de bruit semait de la
  plastification marginale éparse et quelques excursions extrêmes isolées ; le
  retirer laisse la déformation se canaliser dans les bandes
- Aucune revendication sur la correction physique : un champ plastique plus
  concentré n'est pas pour autant plus juste. Le filtre reste justifié par le
  bruit mesuré, pas par l'allure du champ qu'il produit
- Analyse sur données archivées uniquement, aucun calcul mécanique

### 2026-07-31 — Le profil DISFlow ne se départage pas sur la donnée

- Test pré-enregistré : quel profil reproduit le mieux le champ archivé ?
- Verdict **ne discrimine pas** : `1,04` de rapport pour un facteur `1,5` exigé.
  L'attente pré-enregistrée d'un 4/1 meilleur n'est pas vérifiée
- Le garde-fou de cohérence archivée s'est déclenché et était utile : le chiffre
  de `1,583 %` portait sur le support P43, pas sur le champ complet. Après
  correction, la recomputation reproduit l'archive exactement
- Le résidu commun est dominé par les confondants non levés, pas par patch/stride
- 4/1 reste primaire, mais la formulation exacte est désormais « primaire par
  provenance documentée, la reproduction ne départageant pas »
- Résultat de méthode : deux profils quasi identiques sur la donnée diffèrent de
  `1,8 marge` sur l'accord EF/DIC, ce qui chiffre la circularité du critère par
  score

### 2026-07-30 — Filtrage modal du bord et pénalisation optionnelle

- Filtre à 3 mode(s) sur l'écart à la rampe : retire `5,3×` sous le bruit,
  Dirichlet dur conservé, extrémités et intérieur bit-à-bit identiques
- Correction en cours de route : une troncature de rang 3 ne préserve pas une
  ligne nulle. Épinglage linéaire explicite ajouté et pré-enregistrement corrigé
  avant toute production
- Gain numérique décisif : zéro cutback et moitié moins d'itérations Newton
- La dépendance au trajet survit au filtre, donc elle n'est pas du bruit
- Accord DIC inchangé : cet observable est désormais montré insensible au
  trajet et au filtre
- Pénalisation ajoutée en option avec `BOUNDARY_MISFIT` comme indicateur
  d'écart mesure/imposé ; élimination toujours par défaut

### 2026-07-30 — Les deux trajets sont indiscernables face à la DIC

- Comparaison symétrique obligatoire : la vue brute reproduirait l'erreur
  d'asymétrie d'observation que le lot V3 avait déjà documentée et corrigée
- Marges de significativité reprises des intervalles de sensibilité au bruit
  DIC déjà mesurés, pas choisies pour l'occasion
- Verdict enregistré « indiscernable » sur les quatre métriques et les deux
  profils. L'attente pré-enregistrée d'un trajet mesuré au moins aussi proche
  n'est pas vérifiée
- `15,8 %` d'écart sur PEEQ ne produisent aucun gain mesurable sur l'observable :
  l'EVM est dominée par la cinématique imposée et DISFlow lisse à `49 px` les
  filaments où les trajets diffèrent
- Résultat d'identifiabilité : cet observable ne peut ni valider ni réfuter
  l'histoire mesurée. La rampe proportionnelle reste défendable à `2,2×` moins
  de travail Newton
- `fem-inhouse export-run-as-campaign` ajoutée pour rejouer un run multipas au
  travers de l'opérateur d'observation archivé

### 2026-07-30 — Dépendance au trajet mesurée sur PEEQ

- Pré-enregistrement écrit avant tout calcul de métrique, avec seuils, veto de
  discrétisation et discriminateur de structure spatiale
- Contrôle proportionnel à 40 incréments produit spécialement : comparer au
  seul archivé à 20 incréments aurait confondu trajet et discrétisation
- Résultat : L2 relative `15,82 %` sur le cœur, excès positif concentré dans
  les bandes (rapport `13,11`), contrôle de discrétisation `78×` plus petit
- Le rochet de bruit est écarté comme terme dominant par la structure spatiale,
  mais sa contribution n'est pas soustraite
- Nouveau systématique à porter sur les campagnes micromorphiques archivées,
  sans renverser leurs classements
- `fem-inhouse compare-path-dependence` ajoutée avec 9 tests

### 2026-07-30 — Blocage multi-pas résolu, histoire mesurée complète

- Les quatre critères pré-enregistrés du correctif passent
- Incrément 4, la transition qui bloquait depuis le début : converge en 4
  itérations, déformation d'essai initiale `5,440e-04` contre `1,855e-02` avant
- Les 40 états s'exécutent : 65 incréments convergés sur 68, 3 cutbacks aux
  incréments 29, 50 et 59, tous aux itérations 10, 4 et 6, donc des dépassements
  Newton ordinaires rattrapés par un cutback, plus les échecs pathologiques à
  l'itération 1
- `max||du||/||du_B|| = 4,295e-01` : les corrections restent plus petites que
  l'incrément de bord qui les pilote, contre `2,696` en divergence avant
- Prochaine étape ouverte : le test de prédiction temporelle conditionnelle
  (identification sur les pas 1–20, évaluation 21–40 à cartes gelées) devient
  exécutable
- La régularisation temporelle de l'étape 1 ne débloque plus rien ; si elle est
  poursuivie, c'est sur ses seuls mérites

### 2026-07-30 — Cause racine du blocage multi-pas : prédicteur élastique corrompu

- Trace Newton observationnelle ajoutée à `run_fem`, câblée jusqu'à
  `--record-newton-trace`, écrite aussi en cas d'échec. Contrainte vérifiée : champs **bitwise identiques** avec et sans
  trace, donc la trace ne modifie pas le chemin numérique
- Le run instrumenté reproduit l'échec archivé avec les mêmes SHA-256 d'entrée
- `FixedCSRAssembler.assemble` réécrit `matrix.data` sur place et renvoie le
  même objet ; `KII_el` et `K_tang` sont donc le même buffer
- Le prédicteur élastique de la branche histoire est résolu sur ce buffer après
  qu'il a été écrasé par la tangente élastoplastique
- Le chemin proportionnel est indemne : son prédicteur est calculé une fois
  avant la boucle. Aucun résultat archivé n'est affecté
- Le correctif et le rejeu sont pré-enregistrés séparément. Tant que le rejeu
  n'est pas terminé, aucune affirmation n'est faite sur la convergence de
  l'histoire mesurée

### 2026-07-30 — Réfutation de l'hypothèse d'outlier de bord mesuré

- Hypothèse pré-enregistrée : l'échec de l'état 3 vers l'état 4 vient d'une
  excursion non physique du bord mesuré. Critère : `|z| >= 3` à l'état 4
- Mesuré : `z = 0,13` sur le coefficient de chargement, `1,66` sur la
  déformation affine. Maximum sur 40 états `1,99`, sous le `~2,7` attendu de
  39 tirages gaussiens. **Hypothèse réfutée, résultat négatif conservé**
- La lecture antérieure à `3,5 sigma` était un artefact de fenêtre courte :
  incréments bruts comparés à cinq voisins d'une série en tendance
- L'instrumentation Newton, différée au profit de cette campagne, redevient la
  piste principale de l'échec multi-pas
- Acquis conservés : bruit `0,047–0,051 px` par état confirmant la borne
  `0,06283 px` par une route indépendante ; bruit affine à `90 %` ; chargement
  de bord en un seul mode lisse à `99,91 %` de l'énergie
- Saint-Venant tranché : la bande protégée par le padding ne porte presque pas
  de bruit, et la bande qui porte le bruit est quasi uniforme et ne décroît pas
- La régularisation temporelle reste défendable mais faiblement : 5 incréments
  sur 40 sous SNR unité, biais d'accumulation plastique `~3,6 %`. Étape 1 à
  re-pré-enregistrer sur cette base ou à différer
- Aucune mécanique exécutée, histoire immuable non modifiée

### 2026-07-26 — P43 et optimisation du chemin chaud micromorphique

- P43 retenue comme prochaine ROI de calibration après inspection des bandes
- Évaluations intermédiaires limitées à PEEQ, sans tangent ni tenseurs 3D
- Une seule évaluation tangentielle par point fixe convergé et une seule
  reconstruction 3D par calcul FEM convergé
- Buffers réutilisables et prédicteur proportionnel préassemblé
- Chronométrages détaillés ajoutés jusqu'à PARDISO
- Équivalence constitutive bit à bit et équivalence EF sous `1e-10` validées
- Structure CSR libre-libre figée et buffers numériques mis à jour en place
- PARDISO piloté en phases 11/22/33 explicites
- J2 vérifié en CSR supérieur `mtype=2`; comportement inconnu en CSR complet
  `mtype=11`, notamment la future plasticité cristalline par défaut
- Contrôle runtime de l'asymétrie tangentielle sans symétrisation artificielle
- Gate P187 : `-10,6 %` de temps processus et `-16,7 %` de pic RSS
  supplémentaires, sans modifier Newton, le point fixe ou la tangente
- Gate symétrique P187 : `-7,1 %` de temps, `-38,0 %` dans PARDISO et
  `-8,7 %` de pic RSS supplémentaires
- Rapports reproductibles ajoutés sous `validation/performance/`

### 2026-07-24 — Documentation Sphinx anglaise avec Diátaxis

- Création d'une landing page Read the Docs présentant le but scientifique,
  le périmètre supporté, les limites et les résultats validés
- Organisation en quatre quadrants Diátaxis : tutoriel guidé, guides
  opératoires, référence des contrats et explications scientifiques
- Documentation détaillée de la chaîne DIC, de la loi J2/Ludwik analytique,
  de MFront/MGIS, de Newton, du partitionnement, des entrées et des sorties
- Génération reproductible de schémas vectoriels en paires SVG/PDF afin
  d'adapter automatiquement le format aux sorties HTML et LaTeX
- Compilation Sphinx stricte sans avertissement et production locale d'un PDF
  de 70 pages avec LuaLaTeX
- Configuration Read the Docs v2 pour publier `htmlzip` et PDF
- Ajout d'un job CI régénérant les figures, contrôlant leur stabilité et
  compilant le HTML avec les avertissements traités comme des erreurs

### 2026-07-24 — Loi MFront nominale et calcul long de l'article

- Passage des valeurs par défaut de la configuration, de l'API basse et de la
  CLI vers `constitutive_backend=mfront` et `hardening_mode=ludwik`
- Construction paresseuse de la table Python uniquement si le backend
  historique est demandé explicitement ; aucun tableau de 1000 points n'est
  créé sur le chemin nominal
- Conservation de la loi tabulée plafonnée à PEEQ `0,2` uniquement pour les
  régressions historiques et la future comparaison Abaqus
- Exécution complète de la partition de coin `510×460` avec les entrées DIC,
  PyPardiso, 20 incréments et huit threads MGIS
- Convergence 20/20 sans cutback en `648,402 s` solveur et `650,08 s` mur
  global, 112 itérations Newton, résidu relatif final `2,207e-8`
- Conservation des six champs, manifeste, logs, temps `/usr/bin/time -v`,
  empreintes, cartes dérivées, aperçu et rapport de comparaison reproductible
- Gain mur de `40,35 %` et gain constitutif de `6,905×` face à l'ancien calcul
  Python tabulé ; différences L2 relatives de `0,72–0,91 %` sur
  `E/PE/PEEQ/S`
- Pic RSS complet de `4 163 308 KiB`, supérieur de `10,49 %` à l'ancien run :
  la suppression de la table est effective, mais le stockage MGIS et le
  système EF sparse dominent la mesure globale
- PEEQ maximal `0,06496` : le plafond historique `0,2` n'aurait pas été atteint
  sur cette partition, mais il est désormais absent du modèle nominal
- Validation par 172 tests avec MGIS réel, 167 tests et 5 skips sans MGIS,
  Ruff, mypy, smoke CLI MFront et préflight de la partition

### 2026-07-24 — Premier backend constitutif MFront/MGIS

- Installation source de TFEL/MFront 5.1.0 et MGIS 3.1 sous
  `/home/jeff/.local`, avec commits et options CMake enregistrés
- Contournement documenté du suffixe de module Python TFEL 5.1.0 en désactivant
  uniquement `TFEL_APPEND_VERSION`, sans patcher les sources
- Ajout d'une loi J2/Ludwik en contrainte plane avec propriétés locales
  `InitialYieldStress`, `HardeningCoefficient` et `HardeningExponent`
- Ajout d'un adaptateur MGIS vectorisé avec conversions Kelvin, tangente
  cohérente et transactions explicites `evaluate/commit/revert`
- Sauvegarde de 200 incréments pour trois trajets dans un NPZ, avec rapport
  JSON, empreintes et figure
- Passage des seuils initiaux de contrainte et PEEQ ; écart de tangente
  `1,02–6,39 %` conservé comme diagnostic avant branchement dans Newton
- Validation complète par Ruff, mypy, recompilation MFront et 165 tests avec
  94,25 % de couverture ; les deux tests MGIS utilisent la bibliothèque réelle
- Maintien explicite du backend Python en production tant que la parité du
  sous-domaine DIC et la loi tabulée exacte ne sont pas validées

### 2026-07-24 — Benchmark constitutif d'une minute

- Ajout d'un pool de threads MGIS explicite et vérification de sa parité avec
  l'intégration série
- Construction d'un cas hétérogène de 200 000 points, 20 incréments et tangente
  cohérente à chaque mise à jour
- Deux répétitions avec inversion de l'ordre des backends pour limiter le biais
  thermique et de cache
- Temps médian Python `12,347 s`, MFront série `13,333 s` et MFront 8 threads
  `3,527 s`
- Gain MFront parallèle de `3,500×` sur Python et `3,780×` sur MFront série
- Durée complète `1 min 03,24 s`, pic RSS `393,45 MiB`, aucun swap
- Conservation des temps bruts, états finaux complets, échantillons de
  tangentes, empreintes, figure et mesure `/usr/bin/time -v`
- Validation complète par Ruff, mypy et 167 tests avec 94,21 % de couverture
- Limitation maintenue : benchmark du noyau constitutif uniquement, sans
  assemblage CPS4, Newton global ni PyPardiso

### 2026-07-24 — Couplage MFront dans Newton

- Ajout de la sélection `python|mfront` dans `SolverConfig`, l'API typée et la
  CLI de partitionnement, avec chemin de bibliothèque et pool MGIS configurés
- Intégration de la contrainte, des variables internes et de la tangente MFront
  aux points de Gauss dans la boucle Newton CPS4
- Garantie transactionnelle : chaque essai repart du dernier état convergé,
  `commit` uniquement après convergence globale et `revert` avant cutback
- Test homogène plastique de bout en bout avec parité Python/MFront de l'ordre
  de `1e-10`, sans cutback
- Campagne DIC réelle `10×10` sauvegardée avec les six champs de chaque
  backend, diagnostics, empreintes, seuils et rapport JSON
- Passage de tous les seuils : L∞ relatif maximal `3,26e-4`; 20 incréments et
  aucun cutback pour les deux backends
- Temps indicatifs sur le crop : Python `1,583 s`, 84 itérations ; MFront
  2 threads `0,669 s`, 66 itérations
- Validation par Ruff, mypy et 172 tests avec la bibliothèque MGIS réelle
- Maintien du backend Python par défaut jusqu'à décision sur la réplication
  exacte de la table Abaqus à 1000 segments et essai d'une partition article

### 2026-07-24 — Première partition à la taille de l'article

- Exécution de la partition de coin 0 sur la grille `10×10` de l'article avec
  padding 150, soit `510×460` éléments résolus et `360×310` éléments de cœur
- Conservation atomique des six champs finaux `U/S/E/PE/PEEQ/RF`, du manifeste,
  des journaux, de la consommation de ressources et de toutes les empreintes
- Convergence des 20 incréments sans cutback en `1088,13 s` solveur, avec
  113 itérations de Newton et un pic RSS processus de `3 768 132 KiB`
- Vérification des déplacements DIC prescrits à `4,16e-17 mm` et de l'équilibre
  global relatif des réactions à `4,39e-14`
- Archivage des cartes `epsilon_vM` DIC/EF, de leur différence, de `S_Mises` et
  d'une synthèse graphique
- Première comparaison exploratoire : RMSE `0,253` et MAE `0,185` points de
  pourcentage, proches en amplitude des `0,220/0,156` du ROI complet publié,
  mais corrélation spatiale faible (`0,016`) ; aucune revendication de parité
  avant raccordement du ROI et vérification des conventions exactes

### 2026-07-24 — Recentrage sur le calcul autonome depuis la DIC

- Déplacement de la comparaison Abaqus vers une phase de validation externe
  différée et non bloquante
- Adoption du pipeline prioritaire `raw DIC → préparation canonique → partitions
  → raccordement → post-traitement`
- Copie sans modification des quatre tableaux scientifiques dans
  `data/raw/case_study`
- Versionnement des grands tableaux par Git LFS
- Ajout d'un manifeste avec empreintes, formes, types, unités et ambiguïtés
- Conservation des deux générateurs historiques sous `references/legacy_abaqus`
- Exclusion du ZIP duplicatif et du HDF5 de plasticité cristalline, qui
  n'apportent aucune entrée supplémentaire au calcul ciblé
- Enregistrement explicite des trois décisions encore nécessaires : complétion
  nodale, facteur `K=380/396 MPa`, traitement des neuf valeurs non finies

### 2026-07-24 — Préparation canonique et smoke test DIC

- Ajout de `fem-inhouse prepare-case`
- Vérification en flux des tailles et empreintes SHA-256 brutes
- Conversion explicite `V → u_x`, `U → u_y` et pixel → millimètre
- Facteur `K=380 MPa` nominal et `396 MPa` historique sélectionnable
- Refus par défaut des valeurs non finies et politique `nearest` explicite
- Complétion nodale `edge-pad-upper` enregistrée dans le manifeste
- Écriture atomique hors du répertoire brut et réutilisation idempotente
- Ajout d'un crop central reproductible pour les contrôles rapides réels
- Préparation réussie du ROI complet en `3601×3101` nœuds et
  `3600×3100` éléments
- Calcul réussi du crop réel `10×10` en 25 partitions, puis raccordement de
  `U`, `S`, `E` et `PEEQ`
- Documentation du chemin complet depuis un clone neuf
- Clone distant isolé vérifié avec téléchargement des quatre objets Git LFS,
  empreintes identiques et préparation réussie d'un crop `4×4`
- CI GitHub verte sur le commit du pipeline autonome

### 2026-07-24 — Conservation exhaustive des calculs coûteux

- Extension des sorties persistantes à tous les champs finaux du solveur :
  `U`, `S`, `E`, `PE`, `PEEQ` et `RF`
- Écriture atomique et empreinte SHA-256 de chaque champ avant validation du
  statut de partition
- Conservation des diagnostics de convergence dans `status.json`
- Activation de Git LFS pour les résultats numériques de référence
- Validation par la suite complète : 156 tests, couverture 95,26 %

### 2026-07-24 — Création

- Audit initial du code existant
- Installation de l'environnement scientifique et de PyPardiso/MKL
- Vérifications élémentaires du noyau numérique
- Lecture de `ArticleSource/ArticleAdil.pdf`
- Recentrage du projet sur la reconstruction cinématique partitionnée
- Extension du planning de 9 à 12 semaines

### 2026-07-24 — Premier lot scientifique et logiciel

- Ajout du contrat scientifique exécutable et documenté
- Ajout de `pyproject.toml`, du paquet `src/fem_inhouse` et de pytest/Ruff
- Formalisation des configurations matériau, maillage et solveur
- Implémentation commune des déformations DIC/EF et de l'invariant plane-stress
- Séparation entre contrainte EF directe et reconstruction depuis la déformation
- Passage de la table historique de 50 à 1000 points
- Ajout d'une assertion PEEQ au test biaxial

### 2026-07-24 — Partitionnement déterministe

- Ajout des grilles équilibrées de 25 et 100 partitions du ROI complet
- Gestion explicite des cœurs, du padding et des bords du domaine
- Extraction locale des champs aux éléments et aux nœuds
- Raccordement à propriétaire unique, indépendant de l'ordre d'exécution
- Écriture du champ global au format `.npy` mappé en mémoire
- Ajout d'un manifeste JSON déterministe et de la documentation associée
- Validation par 62 tests avec 98 % de couverture lignes et branches combinées

### 2026-07-24 — API solveur et noyau testable

- Déplacement du noyau historique dans le paquet avec point d'entrée compatible
- Ajout de `CaseStudyConfig` comme API publique à la place des 19 paramètres
- Ajout de résultats typés et nommés, avec contrôle des valeurs non finies
- Validation des dimensions, cartes matériau, pseudo-temps et domaines physiques
- Échec explicite si PyPardiso/MKL est absent du calcul de production
- Alignement de la loi tabulée sur la grille `0`, `1e-6`, puis jusqu'à `0.2`
- Ajout des tests élémentaires, du patch affine, du retour plastique et de la
  tangente par différences finies
- Verrouillage exact de l'environnement Linux/Python 3.12
- Validation par 82 tests, 94 % de couverture totale et aucun avertissement

### 2026-07-24 — Workflow partitionné reprenable

- Résolution autonome de chaque zone de calcul paddée avec configuration locale
- Écriture atomique des seuls champs `U`, `S`, `E` et `PEEQ`
- Manifeste immuable avec empreintes des entrées, du code et de la configuration
- Reprise automatique avec détection des fichiers manquants ou corrompus
- Raccordement hors mémoire uniquement lorsque toutes les partitions sont valides
- Validation de la reprise et du raccordement par 87 tests, couverture 95 %

### 2026-07-24 — Exemple exécutable et intégration continue

- Ajout des commandes `backend`, `validate`, `example` et `layout`
- Ajout d'un cas équibiaxial réduit avec seuils déclarés avant exécution
- Sauvegarde de résultats auto-décrits et tutoriel de reproduction
- Construction et inspection réussies du wheel Python
- Ajout d'une CI GitHub avec environnement exact, Ruff et seuil de couverture 85 %
- Validation locale par 92 tests avec 95,04 % de couverture

### 2026-07-24 — Portabilité des données historiques

- Remplacement du `test_config.py` externe manquant par un contrat versionné
- Suppression des chemins Windows personnels dans les scripts conservés
- Configuration des données et résultats uniquement par variables d'environnement
- Validation des formes, valeurs finies et domaines des quatre champs d'entrée
- Documentation explicite des noms de fichiers `.npy` attendus
- Validation par 97 tests avec 95,20 % de couverture

### 2026-07-24 — Métriques et parité de partition

- Ajout de RMSE, MAE, erreur signée, L2 relative et corrélation spatiale
- Ajout d'un ratio de gradient spécifique aux interfaces de raccordement
- Parité vérifiée entre résolution monolithique et quatre partitions homogènes
- Comparaison vérifiée sans padding et avec padding d'un élément
- BGE exact maintenu bloqué : l'article ne donne pas la formule complète et le
  script d'analyse source n'est pas livré
- Validation complète par 104 tests avec 95,36 % de couverture

### 2026-07-24 — Robustesse du solveur

- Vérification des retours plastiques uniaxial, équibiaxial et en cisaillement
- Vérification de la saturation de la table plastique au-delà de `ep = 0.2`
- Stabilité vérifiée pour 5, 10 et 20 incréments
- Convergence vérifiée sur un damier hétérogène de paramètres
- Équilibre global des réactions intégré au seuil de l'exemple
- Échec de convergence forcé et diagnostiqué après réduction de pas
- Validation complète par 111 tests avec 96,14 % de couverture

### 2026-07-24 — Première campagne de performance

- Mesure homogène tabulée avec PyPardiso et 20 incréments
- 10k éléments : 5,01 s et 163 MiB
- 50 176 éléments : 10,60 s et 557 MiB
- 99 856 éléments : 21,87 s et 1,04 GiB
- Utilisation multithread observée entre 349 % et 552 % CPU
- Point 350k reporté pour éviter un OOM avec 3,7 GiB disponibles et swap saturé
- Protocole, limites et conditions de reprise documentés

### 2026-07-24 — Extraction du noyau EF

- Extraction du maillage rectangulaire structuré dans `core.mesh`
- Extraction du CPS4, de la quadrature et de l'élasticité dans `core.element`
- Extraction de l'assemblage sparse et des forces internes dans `core.assembly`
- Validation explicite des géométries, Jacobien, paramètres et formes matricielles
- Boucle Newton conservée comme dette isolée à l'issue de ce lot
- Validation complète par 117 tests avec 96,26 % de couverture

### 2026-07-24 — Extraction du modèle constitutif

- Extraction de l'invariant de von Mises plane-stress dans `core.constitutive`
- Extraction des écrouissages analytique et tabulé avec contrats d'entrée
- Extraction du retour plastique vectorisé et de la tangente cohérente
- Utilisation directe de ce module par le solveur et l'API publique du cœur
- Conservation de l'alias historique `_vm` dans `fem_pixel.py` uniquement
- Validation complète par 123 tests avec 96,32 % de couverture

### 2026-07-24 — Isolation du solveur non linéaire

- Déplacement de l'incrémentation, de Newton-Raphson et du cutback dans
  `core.nonlinear`
- Branchement direct de l'API publique typée sur ce module
- Réduction de `core.solver_legacy` à une couche de compatibilité historique
- Conservation du test de non-convergence par injection du solveur linéaire
- Validation complète par 123 tests avec 96,33 % de couverture

### 2026-07-24 — Diagnostics de convergence structurés

- Ajout de `SolverDiagnostics` au résultat public typé
- Enregistrement du backend, du temps, des incréments, cutbacks et itérations
- Enregistrement du résidu final et du critère de convergence réellement actif
- Émission d'événements `logging` structurés du début à la fin du calcul
- Inclusion des diagnostics dans le `report.json` de l'exemple reproductible
- Validation complète par 123 tests avec 96,66 % de couverture

### 2026-07-24 — Contrôle statique du paquet

- Ajout de mypy aux dépendances de développement verrouillées
- Correction des types des gradients, empreintes par blocs et emplacements
  de champs partitionnés
- Ajout du contrôle mypy à la CI après Ruff
- Validation sans défaut des 25 fichiers du paquet

### 2026-07-24 — Instrumentation des performances

- Chronométrage séparé de l'initialisation et de l'assemblage élastique
- Cumul des temps de retour constitutif, tangentes/assemblages et PyPardiso
- Chronométrage de la construction des sorties et de l'écriture des partitions
- Profil hétérogène 10k : 31,948 s, 78 itérations de Newton, aucun cutback
- Factorisation et substitutions encore regroupées par l'appel PyPardiso

### 2026-07-24 — Convention des réactions

- Définition explicite des réactions comme forces internes sur les DDL prescrits
- Vérification des signes sur les quatre bords du patch affine
- Vérification des résultantes analytiques horizontales et verticales
- Documentation des unités et de l'épaisseur implicite de 1 mm
- Maintien de l'épaisseur Abaqus exacte comme donnée externe encore absente

### 2026-07-24 — Patch test en cisaillement

- Ajout d'un champ affine de cisaillement simple sur maillage 4×3
- Vérification de la convention de cisaillement d'ingénieur `gamma12`
- Vérification de `S12 = G gamma12` et des composantes normales nulles
- Vérification simultanée de `U1`, `U2` et de PEEQ nulle

### 2026-07-24 — CLI partitionnée et job array

- Ajout d'une commande unique pour lister, résoudre, reprendre et raccorder
- Chargement mappé des quatre champs `.npy` et inférence de la taille du ROI
- Ajout du point d'entrée `--partition-id` adapté aux tâches indépendantes
- Ajout d'un modèle Slurm pour les grilles de 25 et 100 partitions
- Fichiers temporaires rendus uniques pour les écritures atomiques concurrentes
- Documentation du lancement, de la reprise et du raccordement hors mémoire

### 2026-07-24 — Qualité statique de tout le dépôt

- Extension de Ruff aux scripts historiques de comparaison et visualisation
- Formatage mécanique sans modification des formules scientifiques
- Exceptions limitées aux noms de variables scientifiques et imports de compatibilité
- Passage de la CI de chemins sélectionnés à `ruff check .`
- Validation locale : Ruff, mypy et 125 tests réussis

### 2026-07-24 — Distribution et citation

- Déclaration PEP 561 du paquet public avec `py.typed`
- Vérification du contenu du wheel construit localement
- Ajout de la construction du wheel à la CI
- Ajout de `CITATION.cff` depuis le titre et les auteurs de l'article source
- Licence laissée explicitement ouverte avant publication publique

### 2026-07-24 — Recouvrement des zones localisées

- Ajout d'une sélection indépendante par quantile supérieur
- Ajout des scores de Jaccard et Dice
- Ajout du rappel de la zone de référence et de la précision de la prédiction
- Conservation des seuils et effectifs pour interpréter les ex æquo
- Tests des recouvrements identique, partiel, masqué et des contrats invalides

### 2026-07-24 — Provenance de la référence scientifique

- Ajout d'un manifeste versionné pour le PDF fourni
- Enregistrement du titre, des auteurs, de la taille et du SHA-256
- DOI maintenu explicitement nul car absent du manuscrit fourni
- Ajout d'un test empêchant une modification silencieuse de la référence

### 2026-07-24 — Rapports automatiques de champs

- Ajout de seuils typés pour RMSE, MAE, corrélation et recouvrement
- Ajout d'une décision globale reproductible sans ajustement après calcul
- Ajout d'une carte signée `prédiction - référence` avec masque et NaN explicites
- Ajout de la commande `compare-fields` et d'un code retour exploitable en CI
- Documentation explicite de l'exigence de co-enregistrement préalable

### 2026-07-24 — Réduction mémoire de la tangente

- Suppression des tenseurs `C_ep` et `C B` matérialisés pour tous les points
- Assemblage par corrections plastiques sur la matrice élastique, par blocs
- Parité avec la formulation dense vérifiée à `rtol=1e-13`
- Réduction théorique de 1 568 à 800 octets globaux par élément
- Mesure A/B 10k : poste tangent -22,4 %, pic RSS processus -3,2 %

### 2026-07-24 — Décisions et revue numérique

- Ajout d'ADR sur le périmètre, PyPardiso et le raccordement des cœurs
- Ajout d'un guide de contribution limité au cas d'étude
- Exigence écrite d'un second relecteur pour toute formule numérique
- Ajout d'un modèle de PR avec preuves mathématiques et performance
- Protection de branche laissée inactive pour ne pas bloquer le dépôt mono-auteur

### 2026-07-24 — Actualisation des actions CI

- CI complète validée sur installation fraîche, wheel inclus
- Mise à jour de `actions/checkout` et `actions/setup-python` vers la version 7
- Suppression attendue de l'annotation de dépréciation Node.js 20

### 2026-07-24 — Reconstruction des tenseurs 3D en contraintes planes

- Ajout d'une couche vectorisée de post-traitement du seul état 2D convergé
- Conservation stricte du maillage CPS4, de Newton, du tangent et des sorties
  historiques
- Reconstruction analytique Python par élasticité plane-stress et
  incompressibilité plastique J2
- Extraction MFront native depuis `AxialStrain`, `ElasticStrain` et `Stress`
- Conservation du résidu numérique `S33` MFront sans remplacement artificiel
- Ajout des sorties `S_3D`, `E_3D`, `EE_3D`, `PE_3D` et
  `S33_RESIDUAL_MPA`
- Extension du résultat typé, des partitions, du raccordement et du chargeur
  de campagnes anciennes
- Ajout des invariants 3D et séparation explicite de `EVM_HISTORICAL`
  et `EVM_RECONSTRUCTED_3D`
- Validation des trajets proportionnels, du déchargement et du chargement non
  proportionnel
- Campagne DIC 10×10 sauvegardée avec non-régression des six anciens champs
- Documentation Diátaxis anglaise reconstruite sans erreur en HTML strict et
  en PDF de 78 pages
- Validation finale avec le vrai backend MGIS/MFront : 199 tests réussis ;
  Ruff sans défaut et mypy sans défaut sur le paquet et le script de campagne

### 2026-07-25 — Condensation d'une loi MFront 3D en contraintes planes

- Ajout d'un contrat commun `PlaneStressMaterialBatch` utilisé par le Newton
  global, sans connaissance de J2 ni des variables internes MGIS
- Compilation de la loi J2/Ludwik identique sous les hypothèses
  `PlaneStress` et `Tridimensional`
- Vérification de l'ordre Kelvin 3D MGIS `[11,22,33,12,13,23]` par
  métadonnées et six essais élastiques indépendants
- Résolution locale transactionnelle de
  `[epsilon33,gamma13,gamma23]` et condensation de la tangente par complément
  de Schur
- Ajout du résidu vectoriel `[S33,S13,S23]` et des diagnostics locaux au point
  de Gauss, sans rupture du champ historique `S33_RESIDUAL_MPA`
- Suppression du fallback J2 implicite pour un comportement MFront ne
  déclarant pas la capacité correspondante
- Tests de parité sur trajets matériels, tangent par différences finies,
  échec local sans pollution d'état, maillage homogène 4×4 et DIC 10×10
- Campagne immuable
  `validation/reference_data/mfront_3d_condensed_dic_10x10_v1` sauvegardée
- Validation avec MGIS/MFront réel : 206 tests réussis ; Ruff, mypy et
  contrôle des différences Git sans défaut

### 2026-07-25 — Benchmark EF comparatif des trois backends

- Ajout d'un pilote reproductible exécutant `python`,
  `mfront-native-plane-stress` et `mfront-3d-condensed-plane-stress` dans des
  processus frais et un ordre alterné
- Mesure du temps mur complet par GNU `time`, du temps solveur, du temps
  constitutif et du pic RSS sur le même crop DIC central 100×100
- Trois répétitions par backend, 20 incréments, deux threads MKL et deux
  threads MGIS ; neuf convergences sans cutback ni échec local
- Sauvegarde systématique de tous les champs, diagnostics, journaux,
  mesures de ressources, configuration et rapport agrégé
- Médianes temps mur / RSS : Python `134,36 s / 248,96 MiB`, MFront natif
  `27,03 s / 269,65 MiB`, MFront 3D condensé
  `83,43 s / 320,30 MiB`
- Confirmation que MFront natif et condensé sont équivalents à la précision
  numérique et que Python respecte les tolérances scientifiques déclarées

### 2026-07-25 — Diagnostic de non-localité par filtre de Helmholtz

- Ajout d'un filtre DCT élémentaire à flux nul, sans matrice globale ni
  modification du solveur mécanique
- Ajout des invariants numériques, de la comparaison sparse directe et d'une
  non-régression exacte pour `ell=0`
- Ajout d'une campagne CLI atomique retrouvant domaine résolu, cœur et padding
  depuis les métadonnées
- Reconstruction commune d'EVM depuis les déplacements DIC et FEM, puis
  filtrage du seul champ FEM
- Séparation stricte de PEEQ et d'EVM DIC dans les métriques d'amplitude
- Ajout des métriques de diffusivité, quantiles égaux, seuils absolus DIC,
  sélection Pareto et modes exploratoire/confirmatoire
- Sauvegarde de tous les champs, rapports, tableaux, figures et empreintes de
  la campagne réelle sur la partition article 0
- Diminution de `49,45 %` de RMSE et L2 relative à `58,88 µm`, mais
  corrélation finale `0,0926` et pics trop atténués
- Conclusion limitée à « hypothèse de largeur spatiale partiellement soutenue
  sur cette partition exploratoire », sans identification de longueur matériau
- Validation complète avec MGIS/MFront réel : 230 tests réussis ; Ruff et mypy
  sans défaut

### 2026-07-25 — Sélection P48 et confirmation tenue à l'écart P42

- Pré-enregistrement de P48 comme unique partition de sélection avant calcul
- Convergence de P48 sur 402 600 éléments en 22 min 16 s de processus, zéro
  cutback et 7,51 GiB de RSS maximal
- Sélection commune de `58,88 µm` par corrélation, IoU top-10 et seuils
  absolus DIC
- Réduction de `64,61 %` de RMSE/L2 sur P48 et corrélation finale `0,6160`
- Pré-enregistrement de P42 comme partition tenue à l'écart avec longueur et
  seuils figés
- Conservation d'une première tentative P42 interrompue par un SIGTERM externe
  avant toute écriture partielle
- Relance P42 inchangée dans une unité utilisateur isolée, puis convergence en
  24 min 45 s, zéro cutback et 7,71 GiB de RSS maximal
- Réussite de tous les seuils confirmatoires sur P42 : corrélation finale
  `0,7036`, réduction L2 `65,43 %`, gains d'IoU quantile et absolu
- Conclusion d'étape 1 relevée à « hypothèse de largeur spatiale soutenue »,
  sans interprétation de `58,88 µm` comme longueur interne matérielle
- Sauvegarde exhaustive des deux calculs mécaniques, champs filtrés, figures,
  journaux, rapports, seuils et empreintes

### 2026-07-26 — Consolidation vérifiable de la documentation publique

- Passage du registre documentaire au schéma 2 : chaque preuve primaire
  déclare désormais des assertions JSON Pointer exécutées avant génération
- Ajout d'une preuve déterministe de la définition de table historique
  (1000 lignes, grille et valeurs de Ludwik), explicitement séparée de toute
  revendication de parité EF Abaqus
- Séparation des claims : définition Abaqus-oriented réimplémentée,
  parité Abaqus/Table–InHouse/Table non démontrée, et cohérence
  InHouse/Table–InHouse/MFront vérifiée
- Remplacement des métadonnées de tracé par le JSON numérique consolidé comme
  preuve primaire du mécanisme micromorphique ; provenance graphique conservée
  comme source secondaire
- Génération automatique de quatre tableaux courts depuis les rapports :
  points matériels, petit cas EF, redistribution PEEQ et état
  d'identifiabilité Newton-25
- Remplacement de la première figure scientifique par une comparaison
  strictement locale : DIC EVM, FEM locale, erreur signée et PEEQ locale
- Correction du guide d'identification avec une configuration réellement
  versionnée et explication des deux étapes F1
- Parcours scientifique d'accueil complété jusqu'à l'identification, les
  preuves actuelles et la portée prédictive
- Références CLI et solveur sparse enrichies avec options, valeurs par défaut,
  contrats, erreurs, phases PARDISO et diagnostics
- Tests documentaires étendus pour vérifier l'échec sur assertion sémantique,
  en plus de la génération déterministe

## Registered SRIX/Méric slip-system comparison

The final-state post-processing comparison is implemented in
`src/fem_inhouse/validation/crystal_slip_metrics.py` and regenerated with
`scripts/compare_srix_meric_slip_maps_p43.py`. It uses the registered P43
100x100, 16-increment archives, averages the two TRI2 states per pixel, and
writes JSON, CSV and documentation figures under
`validation/_generated/performance/srix_meric_p43_m100_16_slip_maps/`.

The archived comparison finds the same principal system and top three systems,
with `S95` Jaccard `0.800`, fraction-vector variation distance `0.2565`, and
normalized total-field cosine `0.9862`. This supports shared dominant
mechanisms with redistribution, not a pure global amplitude rescaling.

The source archives contain final per-system fields only; no incremental
activation history or signed-slip history is inferred. The comparison remains
field-authorized but not performance-authorized, and retains the limitations
of the analytically transposed SRIX `R`, homogeneous orientation, undocumented
physical DIC time, and the unqualified temporal accuracy of the Méric
16-increment path.

## MFront architectural refactor

The staged refactor of the MFront backend is pushed through commits c20ca45,
1ce6a8a, affb32c, 404dc91 and eae0afa.
The compatibility facade remains at src/fem_inhouse/core/mfront.py, while
the implementation is split into:

- mfront_runtime.py: MGIS loading, parameters, introspection and Kelvin
  conversions;
- mfront_state.py: snapshots and public timing records;
- mfront_native.py: native 2D bridge;
- mfront_3d.py: raw 3D bridge and explicit rotations;
- mfront_condensation.py: external plane-stress closure, Schur and blocks;
- mfront_gps/adapter.py: GPS adapter;
- mfront_gps/substepping.py: unchanged GPS substep policy;
- mfront_gps/composite_tangent.py: unchanged composite FD policy;
- mfront_gps/diagnostics.py: non-production shadow tangent.

The extraction was intentionally algorithm-preserving. The exact MFront
qualification environment was used for the targeted tests:

~~~text
52 passed
Ruff: passed
mypy: passed
~~~

The architectural reference is docs/reference/numerics/mfront_architecture.md.
Existing facade imports, including diagnostic private helpers, remain
supported. Unrelated benchmark files present in the worktree were not
included in the refactor commits.

The post-refactor M20 and M100 replays are archived in:

- validation/_generated/performance/mfront_refactor_m20_substepping_path.json
- validation/_generated/performance/mfront_refactor_m100_gps_fd.json
- validation/_generated/performance/mfront_refactor_m100_gps_fd.fields.npz

The M100 GPS+FD replay retained 58 Newton iterations, 8 accepted increments,
192 FD points, 1152 FD trajectories and a final residual of 5.34e-9. Its
measured elapsed time was 44.98 s with four MFront threads, one FFTW thread,
and one Krylov BLAS thread. The M20 replay retained 46 Newton iterations for
the condensed reference and 52 for GPS. The archived performance JSON is a
qualification artifact, not a new constitutive calibration.

## StructuralPlaneStress3D feasibility gate

The first generic-GPS feasibility gate is documented in
`validation/structural_plane_stress_mfront_feasibility.md` and was pushed in
commits `a6c457b` and `2cc86cf`. The inspected installation is unmodified
TFEL/MFront 5.1.0 (`deee4cd`) under `/home/jeff/.local`.

The public headers expose `BehaviourBrickFactory` registration and
`MFRONT_ADDITIONAL_LIBRARIES`, and `@Import` is available. A reproducible
external-brick probe is at
`validation/mfront/structural_plane_stress_brick_probe.cxx`, driven by
`scripts/probe_mfront_brick_plugin.sh`. The loader reaches the external module
but the installed library does not export the normal brick-base implementation
symbols, so an external first-level brick is not currently loadable. This is an
ABI/visibility limitation, not evidence that the StructuralPlaneStress3D
equations are infeasible.

The safe genericity claim at this gate is only an explicit additional
elastic-residual/Jacobian adapter contract. No TFEL fork has been created. A local
`@Integrator` prototype now compiles against the installed TFEL 5.1.0 and
demonstrates that `fzeros` and `jacobian` can be transformed after the
StandardElasticity residual initialization and before the generated Newton
solve. It is at
`validation/mfront/StructuralPlaneStressIntegratorHookProbe.mfront`, with the
reproducible check in
`scripts/probe_structural_plane_stress_integrator_hook.sh`. This remains an
identity-orientation elastic hook probe, not a qualified generic backend. The
script also performs one MGIS point integration and observes a maximum
transverse stress of about `8.9e-15`. The next step is to add structural
rotation and a live one-point tangent/closure oracle, without changing the
qualified GPS or 3D-condensation backends. A second fixed-rotation elastic
probe is now available at
`validation/mfront/StructuralPlaneStressRotatedElasticProbe.mfront`, driven by
`scripts/probe_structural_plane_stress_rotated_hook.sh`; it reaches about
`2.3e-14` maximum rotated transverse traction and `4.9e-19` in-plane strain
error. It uses the repository Bunge convention `Q=global_to_material`,
`T=Q.T`, and leaves `deto` unrotated. The earlier `0c12ac7` fixed matrix and
rotated-gradient result is historical hook evidence only, not the physical
rotation proof. The corrected probe now also reconstructs its converged
Jacobian and returns `T D X_e`; its live Schur comparison is `6.95e-16` and
 central-FD errors for steps `1e-5,1e-6,1e-7` are approximately
`1.77e-14,4.78e-13,5.50e-12`. Its six auxiliary strain entries are located
from MGIS metadata. This qualifies only the rotated elastic probe, not yet
per-point orientation properties or Méric. A J2 probe now exists at
`validation/mfront/StructuralPlaneStressJ2Probe.mfront`, with
`scripts/probe_structural_plane_stress_j2.sh`; it transforms all seven local
Jacobian columns without naming the plastic variable. The plastic-point
maximum transverse stress is `7.99e-14`, and FD tangent errors for
`1e-5,1e-6,1e-7` are `3.94e-7,3.94e-9,3.12e-11`. This is still a prototype
contract proof, not yet an independent condensation qualification. A SRIX
closure probe is now available at
`scripts/probe_structural_plane_stress_srix.sh`; it generates a temporary
variant from the raw SRIX law, applies the same generic `fzeros`/`jacobian`
row transformation, and does not modify the production MFront source. With
the qualified Bunge rotation it reaches `9.30e-15` maximum transverse
traction and `1.74e-18` in-plane kinematic error. The same generated tangent
code stores the transformed Jacobian before state promotion and passes central
FD checks from `2.46e-6` at `h=1e-5` down to `2.48e-10` at `h=1e-7`. The same
generator also passes on the raw
`Fcc316LMericCailletaud` behaviour, using a smaller rate-dependent probe
increment: maximum transverse traction `6.77e-15` and in-plane error
`1.42e-19`. Its tangent FD errors range from `3.97e-15` to `2.45e-13` over
the same steps. This is a one-step generic closure/tangent proof for both
laws, not yet a same-state raw-3D Schur qualification; the production
behaviours remain unchanged. The validation generator currently uses an
18x18 auxiliary Jacobian buffer, which must be replaced by a dimension-safe
reusable mechanism before industrialisation.

### 2026-08-10 — Chantier documentaire : couche débutant et options non documentées

La couche d'accueil manquante est créée :
`docs/how-to/run_316l_crystal_plasticity.md` donne un chemin court pour
quelqu'un qui n'a jamais utilisé MFront, avec un unique choix de backend et un
bloc Python exécuté verbatim par un test. La page de choix de backend
dupliquée est réconciliée : `choose_an_mfront_backend.md` devient une souche de
redirection, `choose_mfront_backend.md` porte le contenu.

**Deux défauts réels trouvés en vérifiant la doc contre le code, pas en la
relisant.**

Le premier est une affirmation fausse que j'avais écrite moi-même :
`configuration.md` et la page de choix promettaient que le backend condensé
*refuse* les options `gps_*`, « de sorte qu'une configuration ne peut pas en
porter une sans effet ». Mesuré : seul `gps_shadow_tangent` est refusé.
`gps_composite_fd_tangent`, `gps_composite_fd_step` et `gps_failure_diagnostics`
sont acceptés puis **silencieusement ignorés**. Les deux pages disent maintenant
ce que le code fait. Poser le garde manquant côté code changerait le
comportement de configurations existantes, y compris d'archives rejouables :
c'est une décision à prendre, pas un effet de bord de la doc.

Le second est un test rouge dans `main`, sans rapport avec mon travail :
`test_structural_plane_stress_same_state_schur` lance son script de
qualification en sous-processus sans la racine du dépôt sur `PYTHONPATH`, alors
que le script importe `scripts.diagnose_gps_tangent_blocks`. Sous pytest la
racine vient de `pythonpath = ["."]`, dont un sous-processus n'hérite pas. Le
test passe l'environnement explicitement et repasse au vert.

Douze clés de `constitutive_options` sont désormais documentées, contre cinq
avant : les quatre clés de sélection de loi et d'orientation
(`crystal_orientation`, `parameter_set`, `parameters`, `paired_parameter_set`),
`condensation_block_size`, et les trois arrivées avec les campagnes récentes
(`srix_smoothing_epsilon`, `srix_smoothing_exponent`, `gps_failure_diagnostics`).
Le lissage Charbonnier est décrit dans `explanation/forest_rubin_srix.md` sous
sa forme réellement implémentée — une norme généralisée d'exposant `n = 11` par
défaut, et non la forme racine carrée que décrit encore le commentaire du
`.mfront`. Le point qui compte : `epsilon = 0` n'est pas une limite mais une
**branche distincte**, donc la configuration par défaut intègre exactement la
loi historique, sans erreur de régularisation à défalquer.

Pour que cette dérive ne se reproduise pas par simple vigilance,
`test_every_accepted_constitutive_option_is_documented` confronte les clés que
`plane_stress_material.py` consomme à celles que la référence cite. Contrôle
négatif effectué : une clé fictive est bien détectée.

Les temps M100 cités (`62,38` / `74,05` / `58,38 s` pour condensé / GPS /
GPS+FD, à 57 / 85 / 58 itérations de Newton) sont tracés jusqu'à leurs trois
artefacts `*_runtime_blas1`, tous à huit incréments, `mfront_threads: 4` et
BLAS mono-fil, exécutés sous `2defce9`. Ils remplacent la campagne
`gps_tangent_variants_m100`, dont la variante `gps_composite` n'avait pas
convergé. Le comparatif M20 à trois backends est ajouté : le backend structurel
générique reproduit la loi GPS écrite à la main à `9e-17` en déplacement pour
le même nombre d'itérations.

Neuf erreurs `mypy` subsistent sous `src/`, toutes antérieures et hors de ce
chantier, qui n'a touché aucune ligne de calcul.

Complément du même jour. Deux jeux de temps M100 coexistaient dans la doc sans
se nommer : `62,38 / 74,05 / 58,38 s` sous `2defce9`, et
`56,72 / 51,65 / 54,56 s` sous `c8af766`. Ils ne se contredisent pas, ils ne
mesurent pas la même chose — le premier est le seul à comparer
`gps_composite_fd_tangent` à lui-même (avec et sans), le second est le seul à
couvrir les trois backends au même commit. La page de choix les étiquette
désormais comme tels et avertit explicitement de ne pas lire les valeurs
absolues en travers des deux jeux, ce qui mesurerait le travail intercalaire et
non les backends. Le comparatif courant retenu est celui de `c8af766` : environ
`10 %` d'écart entre les trois routes, accord `1,2e-16` en déplacement entre la
loi GPS écrite à la main et la fermeture structurelle générique, mêmes `192`
points sous-pas. Le choix ne se joue donc pas sur le coût.

`reference/api.md` nommait la façade `core.mfront` sans dire de quoi elle est la
façade ; les modules de mise en œuvre et le registre `MFRONT_BEHAVIOURS` y sont
maintenant listés. Enfin, la page d'accueil débutant était rangée sous
« Extend » : elle ouvre désormais l'index des how-to et la racine propose un
parcours qui commence par elle.

### 2026-08-10 — `sign(0)=0` : correction de sous-gradient, pas variante de loi

Le lissage Charbonnier et la régularisation compacte `δ` ont rempli leur rôle
de diagnostic, mais le résultat qui compte est le contrôle `sign(0)=0`. Il est
maintenant qualifié, et la conclusion est nette : **ce n'est pas une
modification de la loi.**

Fait structurel d'abord, lisible dans la source : `dg_abs_derivative` n'est
consommé qu'aux lignes 303, 304 et 312 de `Fcc316LForestRubinSrix.mfront`,
toutes écrivant `dfg_ddg`. Il n'atteint ni `feel`, ni `fg`, ni
`@UpdateAuxiliaryStateVariables`. Ce qui entre dans le résidu,
`dg_abs_regularized`, vaut exactement `abs(dg)` dès que `δ = 0`, quelle que
soit la valeur du sous-gradient.

**Un seuil préenregistré a été dépassé et n'a pas été déplacé.** H1 exigeait un
accord ≤`1e-11` entre les deux conventions ; le mesuré est `6,1e-11`. Le seuil
était mal spécifié : je l'avais posé en supposant erreur-solution ≈
tolérance-résidu, ce qui ignore le conditionnement du Jacobien local, très
mauvais à ces états de cusp. Cette explication ne valait rien tant qu'elle ne
risquait rien, donc elle a été testée par une prédiction falsifiable — un
balayage de `epsilon`, non préenregistré.

Le balayage est le résultat décisif de la campagne. L'écart `sign(0)=0` suit la
tolérance proportionnellement, sans plancher : `9,94e-08` à `1e-10`,
`1,33e-09`, `1,45e-11`, `1,24e-13` à `1e-16`. L'écart de `δ=1e-5` vaut
`1,355e-04` aux quatre tolérances, à quatre chiffres significatifs. Même racine
d'un côté, racine différente de l'autre.

L'origine des écarts de champs M200 est également fermée, au point matériel.
Sur les 380 états, en reproduisant la séquence de sous-pas du pont :
convention changée à partition égale → `6e-11` ; partition retirée à convention
égale → jusqu'à `1,5e-2`. Huit ordres de grandeur. Les `1e-4` de la campagne
sont cet effet dilué en L2 sur un maillage où la plupart des points n'ont
jamais sous-pas. À noter : l'historique s'en sort avec deux divisions sur 378
des 380 points — l'échec est net, pas profond.

Le contrôle du test de non-régression a d'abord échoué, et c'était utile :
`δ` agit sur l'incrément par pas, pas sur le glissement cumulé, donc un chemin
en huit gros pas ne l'active jamais et les deux variantes coïncidaient à
`2,7e-14`. Le test aurait passé pour la mauvaise raison. Il utilise maintenant
64 pas de `0,004`, où `δ` vaut `8,6e-4` contre `4,3e-11` pour le sous-gradient.

**Le défaut n'a pas été basculé.** Le passage de `-1.` à `0.` est une ligne et
supprimerait 978 points sous-pas, mais si la racine est inchangée les
*partitions* ne le sont pas : un rejeu sous défaut basculé ne reproduirait les
champs archivés qu'à `~1e-4`, et toutes les comparaisons existantes
acquerraient silencieusement ce plancher. C'est une décision sur le contrat de
reproductibilité de l'archive, pas une question numérique.

Neuf erreurs de lint dans les scripts de la session concurrente ont été
réparées mécaniquement pour rendre le gate vert.

### 2026-08-10 — MFront livre lui-même les quatre blocs tangents couplés

Le contrat `generic_implicit_sensitivity_contract.md` concluait qu'un export
TFEL supplémentaire était nécessaire, MGIS n'exposant ni le résidu local ni son
Jacobien. **Le constat sur MGIS est exact, la conséquence était fausse.** L'hôte
n'a jamais eu besoin de `F_z` et `F_q` : il a besoin des dérivées, et MFront
les calcule déjà en interne.

Cinq faits vérifiés sur cette machine, aucun supposé : le DSL
`ImplicitGenericBehaviour` existe ; `PoroPlasticity.mfront` de TFEL 5.1.0
couple deux gradients et deux forces ; `@TangentOperatorBlocks` nomme les
dérivées croisées voulues ; `getIntegrationVariablesDerivatives_*` les tire du
Jacobien implicite convergé ; MGIS publie `tangent_operator_blocks`.

`validation/mfront/MicromorphicJ2GenericBlocksProbe.mfront` applique ce patron
au J2/Ludwik micromorphique : paire mécanique `(eto, sig)` et seconde paire
`(chi, pobs)` où `pobs` est la déformation plastique équivalente. L'appariement
est un **dispositif d'interface, pas une conjugaison énergétique** — MFront ne
l'exige pas, et rend exactement les quatre blocs demandés.

Les quatre blocs coïncident avec des différences finies centrées : les deux en
`eto` chutent d'un facteur 100 exact entre `h=1e-6` et `1e-7`, l'`O(h²)` d'une
différence centrée, puis plafonnent sur le bruit de soustraction ; les deux en
`chi` sont déjà au plancher à `1e-6`, donc bornés à `~1e-10`. Coût sur 10 000
points : `177,82 ms` avec les quatre blocs contre `159,59 ms` sans tangent,
soit **11 %**, là où la route par différences finies coûte `9×`.

Un piège, consigné parce qu'il a coûté trois tentatives. Sous `Implicit`, la
brique `StandardElasticity` pose `deel = deto` avant le Newton local. Sous
`ImplicitGenericBehaviour` **rien ne le fait**, et le défaut `deel = 0` laisse
`sig` à la contrainte engagée — nulle au premier incrément plastique d'un point
vierge, donc `3 dev(sig)/(2 seq)` divise par zéro. Le symptôme trompe : tous les
pas élastiques passent, tous les pas au-delà de la limite échouent à n'importe
quelle amplitude, ce qui ressemble à un résidu faux. Isolé par bissection — un
Jacobien numérique échouait à l'identique, écartant la linéarisation, et une
variante à un seul gradient aussi, écartant le mécanisme à deux gradients.

Reste avant production, et c'est délibérément hors de cette sonde : elle est
tridimensionnelle alors que la loi de production est en contraintes planes ;
elle abandonne la brique `StandardElastoViscoPlasticity`, donc le retour radial
est écrit à la main et devra être qualifié contre le comportement actuel ; et
elle est J2, SRIX n'ayant pas été essayé.

**Trois tests étaient rouges dans `main` à mon arrivée**, tous dus à
`c461fa6`, qui a rendu `sign(0)=0` canonique, supprimé le paramètre et mon test
de non-régression. Deux étaient mes tests d'accueil, périmés par la réécriture
de la page : le backend recommandé est passé au structurel et le bloc Python
vérifié avait disparu. Bloc restauré et de nouveau exécuté par la suite.

Le troisième était le leur, et il apprend quelque chose. `sign(0)=0` **divise
par deux** le nombre d'incréments nécessaires pour capter le renversement :
somme des glissements `0,02048` à 10 incréments contre `0,01713` à 20, `0,01714`
à 40 et `0,01716` à 80. Le seuil du constat passe de vingt à dix, et à dix
l'écart n'est plus exactement nul mais `1,46e-3`. Mon propre travail de
qualification ne couvrait que des états de chargement monotone : le basculement
du défaut n'est donc pas neutre au renversement, il y est favorable, ce que
personne n'avait mesuré.

### 2026-08-13 — Reprise après 78 commits de la session concurrente

Relevé d'état, sans modification du code : la session concurrente écrivait dans
l'arbre pendant ce relevé (`srix_generic.py`, `plane_stress_material.py`,
`compare_srix_generic_real_ebsd_material.py` non committés, ajout du pool de
threads MGIS au pont Generic).

**Ce qui a avancé.** L'architecture `ImplicitGenericBehaviour` que la sonde J2
du 10 août avait établie a été portée jusqu'à SRIX. Le verrou identifié —
`getIntegrationVariablesDerivatives_*` n'accepte pas le tableau `g[Nss]` des
douze incréments de glissement comme bloc de sortie — a été contourné par une
variable d'intégration scalaire `chilocal` contrainte par
`chilocal + dchilocal = chi + dchi` : les douze résidus de glissement dépendent
de `chilocal + theta*dchilocal`, donc la dérivée gradient-vers-tableau est
remplacée par la dérivée de la contrainte scalaire. Les quatre blocs 3D passent
un contrôle par différences finies centrées à `3,02e-9`, et la condensation
contraintes planes (élimination des composantes Kelvin transverses `zz, xz, yz`)
à `7,45e-10`. Deux erreurs de transcription avaient été trouvées et corrigées au
passage : contribution plastique ajoutée deux fois à `feel`, et cisaillements
Kelvin du tenseur élastique posés à `G` au lieu de `2G`.

Le backend `mfront-srix-generic-plane-stress` est enregistré en opt-in strict
(identifiant `fcc_forest_rubin_srix_generic_validation` + bibliothèque dédiée).
Sur le crop EBSD P43 réel `(1610:1613, 1075:1078)`, écart Generic/historique de
`9,1e-16` en contrainte et `1,4e-16` en glissement cumulé au point matériel ;
`3,4e-7` en déplacement à travers le solveur global imbriqué.

Côté solveur, ADR 0010 fige la stratégie imbriquée comme **référence de
robustesse** et classe le monolithique `(u, chi)` en candidat sous
qualification. Sur P43 M100 J2 à `1e-6`, imbriqué `49,3 s`, monolithique
`51,0 s`, étagé vrai `54,9 s` : le monolithique n'est pas encore un gain.

**Ce qui est cassé, et depuis quand.** Suite complète : **61 échecs, 1453
passés, 1 ignoré**. Cause unique, introduite par `f013d25` (13 août 16h27) :

```
@MaterialProperty stress Hchi;
Hchi.setEntryName("MicromorphicCouplingModulus");
```

ajouté à `mfront/Fcc316LForestRubinSrix.mfront`. Une `@MaterialProperty` MGIS
n'a pas de valeur par défaut : toute construction qui ne la fournit pas meurt à
`buildEvaluators`. Seul `mfront_3d.py` a été mis à jour. Restent découverts :

- les 59 tests MGIS directs de `test_forest_rubin_srix.py`,
  `test_srix_canonical.py` et `test_srix_symmetry_and_plane_stress.py` ;
- la route structurelle : `Fcc316LForestRubinSrixStructuralPlaneStress` est
  **générée** depuis la même source et hérite donc de la propriété (vérifié :
  `mps = [MicromorphicCouplingModulus, Q11..Q33]`), mais sa fiche de catalogue
  déclare `material_properties=()` et `MFrontNativeGeneralisedPlaneStressBatch`
  n'accepte aucune propriété matériau. D'où l'échec du bloc Python d'accueil et
  de `qualify_structural_plane_stress_same_state_schur.py`.

Le garde-fou de `plane_stress_material.py:563` fournit bien des zéros, mais il
teste la fiche *avant* la bascule structurelle et `crystal_material_properties`
n'est transmis qu'à la fabrique condensée 3D.

Correctif minimal établi par mesure directe — poser
`MicromorphicCouplingModulus = 0` restaure l'intégration à l'identique
(`126,7897 MPa` sur le plateau canonique) :

1. déclarer la propriété sur la fiche `..._structural_plane_stress` et
   transmettre `crystal_material_properties` à la fabrique GPS **après** la
   bascule — le comportement GPS `Fcc316LForestRubinSrixGps` a sa propre source,
   ne déclare que `Q11..Q33`, et lui passer la propriété échouerait ;
2. faire poser la propriété à `mfront_gps/adapter.py` avec la même précaution de
   durée de vie que les tampons `Q` (`ExternalStorage` garde un pointeur) ;
3. fournir `0.0` dans les trois fichiers de tests MGIS directs.

Non appliqué : la session concurrente édite ces fichiers en ce moment.

Lint : deux `E501` dans `scripts/generate_srix_generic_3d.py` (lignes 29 et 48).

`Claude.md` n'avait pas été mis à jour depuis `7a2ca21`, soit 78 commits.

### 2026-08-14 — Réparation de la régression `MicromorphicCouplingModulus`

Toujours d'actualité au moment de reprendre : **61 échecs, 1454 passés**, cause
inchangée. La session concurrente avait committé son travail entre-temps
(arbre propre) sans toucher à ce point.

Le diagnostic de la veille était juste mais **incomplet d'une moitié**. `f013d25`
n'a pas ajouté une, mais **deux** entrées d'interface à
`mfront/Fcc316LForestRubinSrix.mfront` :

```
@MaterialProperty stress Hchi;        → MicromorphicCouplingModulus
@ExternalStateVariable strain chi;    → NonlocalEquivalentPlasticStrain
```

MGIS ne donne de valeur par défaut **ni aux propriétés matériau ni aux
variables d'état externes**. N'avoir corrigé que la première a fait passer les
tests d'un échec à l'autre, au même endroit — `buildEvaluators` — avec un nom
différent. C'est le seul intérêt de la note : la symétrie des deux mécanismes
n'était pas dans mon relevé initial.

Le défaut de conception derrière : ces deux manques ne se manifestent **pas à la
construction** mais à la première intégration, ce qui les rend invisibles
jusqu'au premier pas plastique.

Trois corrections :

1. `mfront_behaviours.py` — la fiche `..._structural_plane_stress` déclare
   maintenant les deux entrées. Son comportement est **généré** depuis la même
   source SRIX par `generate_structural_plane_stress.sh` et en hérite ; le
   catalogue le disait muet.
2. `mfront_gps/adapter.py` — pose les deux à zéro, avec la même précaution de
   durée de vie que les tampons `Q` (`ExternalStorage` garde un pointeur).
   Piloté par les **métadonnées compilées** et non par la fiche : c'est la
   synchronisation manuelle fiche/générateur qui avait échoué. Zéro n'est pas
   un bouche-trou, c'est la valeur documentée à laquelle la loi redonne
   exactement la réponse locale historique, et ce backend est local — le
   couplage non local y est refusé en amont.
3. Les cinq points de construction MGIS directs des trois fichiers de tests
   SRIX, via un helper local par fichier.

Vérifié : **1515 passés, 1 ignoré**, zéro échec. Ruff propre après réparation de
deux `E501` dans `scripts/generate_srix_generic_3d.py` — sortie du générateur
vérifiée octet pour octet identique avant/après. Doc `-W` verte.

Restent 16 erreurs mypy, toutes antérieures et inchangées par ces corrections
(mesuré des deux côtés du diff) ; l'essentiel est dans `coupled_newton.py`, en
cours côté session concurrente.

### 2026-08-14 — Réduction de l'espace plastique guidée par l'observabilité DIC

Le diagnostic P43 M20 impose un nouveau jalon avant M50/M200 : le problème
inverse plein champ en `Delta p(x,t)` n'est pas identifiable avec la DIC seule.
Avec le prior Ludwik, l'oracle reste proche de Ludwik ; avec `prior_weight=0`,
la solution devient instable, plus coûteuse et l'accord DIC se dégrade
(`experimental_oracle_p43_m20_prior_000`). Le champ libre ne doit donc pas être
interprété comme une vérité expérimentale.

Le prochain objectif est, sans hyper-réduction spatiale, de remplacer le champ
libre par `Delta p_n = Delta p_L,n + Phi_p a_n`. `u`, les contraintes, tous les
points matériau et le résidu d'équilibre restent plein champ. RID, DEIM et ECSW
sont hors périmètre jusqu'à preuve qu'un espace réduit identifiable existe.

La référence de linéarisation est le run M20 priorisé ; elle n'est pas une
vérité physique. Pour chaque état :

    K = d(B^T sigma)/du       G_p = d(B^T sigma)/d(Delta p)
    S_p = -K^{-1} G_p         O = W_D S_p

`W_D` est le whitener DIC existant. Le module
`identification.plastic_observability` introduit les actions matrix-free de
`G_p`, `G_p.T`, `O` et `O.T`; aucune matrice globale dense n'est formée. Les
solveurs transposés utilisent explicitement `K.T` et ne supposent pas la
symétrie du tangent.

Le même module fournit maintenant un `PlasticMetric` explicite, normalisé par
un `reference_scale` RMS et pouvant ajouter le terme de différences voisines,
ainsi que `generalized_modes`, qui appelle `eigsh` sur les `LinearOperator` de
`A_obs` et `H_p`. La valeur de référence et le poids spatial restent à
calibrer ; ils ne doivent pas être présentés comme des paramètres physiques.

Suite obligatoire :

1. tests d'adjoint de `G_p` et `O`, puis tests par différences finies ;
2. métrique SPD `H_p` documentée (amplitude/régularité du prior) ;
3. opérateur `A_obs = sum O.T O` et problème `A_obs phi = lambda H_p phi`,
   sans SVD de snapshots ;
4. validation des modes sur des états tenus à l'écart ;
5. intégration d'une paramétrisation réduite dans l'oracle en conservant
   positivité, transactions et cutbacks ;
6. reproduction de l'oracle plein avec le prior Ludwik ;
7. seulement ensuite, test `prior_weight=0` dans l'espace observable.

Le rang ne sera pas choisi par une énergie POD. Il sera justifié par le spectre,
la stabilité des modes, la validation hors échantillon et la robustesse
non-linéaire. Le critère scientifique est le nombre de directions plastiques
réellement observables par cette expérience, puis la question de savoir si,
dans cet espace, la DIC préfère une solution différente de Ludwik.

Une première qualification M20 est disponible dans
`validation/_generated/performance/experimental_oracle_p43_m20/observability/`.
Elle utilise les états `[0, 10, 20, 29, 39]`, le whitener P43 corrigé du mode
DC et `H_p=I` provisoire. Les deux premières valeurs propres sont environ
`4.835e7` et `4.027e7`; les contrôles d'adjoint donnent
`4.2e-16` pour `G_p` et `4.0e-10` pour `O`. Ces valeurs ne sont pas encore une
conclusion physique : la métrique spatiale doit être calibrée avant de fixer le
rang ou de lancer l'oracle réduit sans prior.

Avec la métrique normalisée par le RMS des incréments plastiques non nuls
(`p_ref = 2.53965e-4`), le spectre M20 à 10 modes est :
`[2494.8, 2078.1, 1844.4, 1535.2, 455.1, 435.4, 369.4, 357.2, 340.9,
332.1]`. Il ne montre pas encore un gap interprétable : la base reste
instantanée, les états sont seulement `[0,10,20,29,39]` et la métrique spatiale
n'est pas activée. Le calcul rank-20 a été trop coûteux dans la configuration
actuelle et n'a produit aucun artefact ; il faut optimiser les applications
répétées de `O/O.T` avant d'en déduire quoi que ce soit.

Les figures montrent que `H_p=I` produit des premiers modes en damier, malgré
des réponses `S_p phi` lisses : c'est un artefact de normalisation/haute
fréquence, pas un mode plastique physique. Avec le même `p_ref` et
`spatial_weight=100`, les modes deviennent lisses ; ce poids est uniquement un
test de suppression du Nyquist et ne constitue pas encore une calibration.
Aucun oracle réduit ne doit être lancé avant cette calibration.

Un premier opérateur `M_D` spectral a été ajouté à partir de la moyenne
isotrope des gains sinusoïdaux V4. Avec `H_p` amplitude-only, le spectre
M20 devient environ `[0.142, 0.0604, 0.00948, 0.00253]` pour les quatre
premiers modes : les damiers ne dominent plus les premiers modes plastiques.
La réponse blanchie `W_D M_D S_p phi` reste visuellement bruitée, ce qui est
attendu avec l'inverse du PSD ; cela devra être évalué quantitativement. Cette
version de `M_D` est une approximation isotrope, pas encore la chaîne DIC
complète ni une MTF directionnelle validée.

La normalisation automatique par Ludwik donne `p_ref=2.96197e-4` sur la
trajectoire utilisée. Avec `M_D` et `H_p` amplitude-only, les quatre premières
valeurs propres sont `[0.1931, 0.0821, 0.01289, 0.00344]`, donc le rapport
`lambda_2/lambda_3` vaut environ `6.37`. Une sélection d'états décalée
`[4,14,23,32,39]` donne `[0.1596, 0.0860]`; les angles principaux du sous-espace
de rang 2 avec `[0,10,20,29,39]` sont `6.8°` et `12.0°`. C'est une indication
positive, mais insuffisante pour figer `r=2` ou lancer l'oracle réduit.

La construction sur les 40 états donne
`lambda=[1.3153, 0.8071, 0.03914, 0.02336]`, soit un gap
`lambda_2/lambda_3=20.6`. Le sous-espace de rang 2 formé avec les 40 états
forme des angles principaux de `7.7°` et `4.3°` avec celui des cinq états.
C'est nettement plus robuste. Il reste à valider la traduction non linéaire de
cette base avec le prior avant tout `prior_weight=0`.

### 2026-08-14 — Couplage non local inerte, et le mur de l'incrément plastique

Deux corrections indépendantes, la première invalidante.

#### Le couplage non local SRIX n'avait jamais agi

Mesure de départ, sur le cas P43 réduit hétérogène : la sensibilité de la
solution au module de couplage, de `Hchi = 0` à `Hchi = 100`, valait
**`0,000e+00` bit à bit** sur la voie legacy. Le batch construit avec
`nonlocal_coupling_modulus_mpa=100.0` contenait `MicromorphicCouplingModulus =
0.0` en tout point.

Deux canaux mènent la même grandeur au batch condensé : la voie J2 passe
`micromorphic_coupling_modulus_mpa`, la fabrique cristalline passe l'entrée par
`material_property_values`. Le repli à zéro de `mfront_3d.py` écrasait le second
**inconditionnellement**. La demande était acceptée, enregistrée au manifeste, et
jetée. Et la fabrique cristalline ne remplissait `crystal_material_properties`
que dans le cas local, donc les deux canaux étaient vides à la fois.

Le test d'équivalence Generic/legacy était vert **parce que les deux côtés
étaient découplés**. C'est le mode d'échec du test du δ : un garde-fou qui ne
s'active jamais. `9ac1b31` n'avait rien cassé — en rendant le couplage effectif
côté Generic, il a révélé que le legacy ne l'était pas.

Après correction, les deux backends répondent identiquement (`8,232e-08` mm) et
leur écart tombe de `5,4e-3` à **`8,3e-12`**. L'équivalence devient vraie.

Le test ajouté n'est pas un test d'accord mais de **sensibilité** : allumer le
module doit déplacer la réponse. Un accord entre deux backends ne peut pas
distinguer « tous deux corrects » de « tous deux inertes ».

**Conséquence à trancher** : tout ce qui a été qualifié « non local SRIX » avant
aujourd'hui l'a été avec `Hchi = 0`, y compris ce que décrit
`docs/reference/numerics/srix_nonlocal_source.md` et le pas P43 M100. Le champ χ
était bien calculé et injecté ; seul son effet en retour était nul. Les
campagnes concernées sont à rejouer.

#### L'incrément plastique : ce n'était pas un problème de solveur

Le replay directionnel butait à l'état 21, point 117, sur
`driven J2 local line search failed`, insensible à la continuation locale.

En contraintes planes, `C` et `M` commutent. Dans leur base propre commune le
système local 3×3 se réduit à **une équation scalaire**,
`phi(q) = Σ m_i t_i²/(q+a_i)² = 1`, avec `phi` strictement décroissante et
convexe. D'où : racine unique encadrée par `[0, q_trial]`, aucun rôle pour une
recherche linéaire, et surtout une condition d'existence en forme close —
`Δp < Δp_max = sqrt(Σ t_i²/(c_i² m_i))`.

C'est toute l'histoire du point 117 : `Δp = 2,935356e-05` contre
`Δp_max = 2,934566e-05`, dépassement de 0,027 %. J2 associé relaxe la contrainte
déviatorique vers l'origine et l'atteint à `Δp` fini ; au-delà aucun état à
`q > 0` n'existe. L'ancien solveur rencontrait un mur de **non-existence** et le
rapportait comme un accident de conditionnement, ce qui a orienté l'enquête vers
le suivi de branche.

Détail consigné parce qu'il m'a coûté une itération : Newton depuis `q = 0` est
monotone et ne dépasse jamais, mais loin du mur il progresse d'un facteur ~1,5
par itération depuis un premier pas d'ordre `a/2`, donc en `log(q_trial/a)`
itérations. Ma première version a échoué à l'état 4, point 281 — un échec qui
ressemblait à celui qu'il venait de remplacer. Corrigé par un départ au retour
radial classique `q_trial - a_eff`, exact quand les deux valeurs de relaxation
coïncident, et un encadrement avec repli par bissection.

Vérifications : résidu `1,2e-15` sur 20 000 états aléatoires balayés jusqu'à
99,9 % du mur ; accord `1e-12` avec le retour radial en cisaillement pur ; et
les états 10 et 20 du diagnostic directionnel reproduits **bit pour bit**. Le
nouveau solveur ne donne pas une autre réponse, il donne la même sans condition.

Enfin, en parcourant l'histoire archivée de l'oracle contre la borne
(`scripts/diagnose_admissible_delta_p_wall.py`), **les 40 états restent en deçà
du mur**, pire ratio `0,871` à l'état 27 et `0,831` à l'état 21. Le dépassement
appartient donc aux états **perturbés** par la sonde directionnelle, qui déplace
la direction d'écoulement et donc la borne. Avec projection sur
`[0, 0,999 Δp_max)`, le replay va au bout des quatre états, gain directionnel
`< 0,61 %` partout — le résultat négatif tient désormais sur 4 états sur 4.

La projection n'est délibérément **pas** câblée dans le matériau : elle a joué
3221 fois avec un ratio demandé allant jusqu'à `6,64`, ce qui est une
information sur la sonde et non un arrondi à masquer. Voir
`validation/driven_j2_admissible_increment.md`.

### 2026-08-15 — L'oracle tensoriel : de l'observabilité à l'excitation réelle

Nuit de travail autonome, quatre commits. Le fil : le spectre d'observabilité
dit ce que l'instrument *pourrait* voir ; il fallait demander aux données ce
qu'elles contiennent réellement.

**Matrix-free d'abord** (`712a87e`). Le prototype dense n'était pas un
algorithme : à M100 l'opérateur de forçage seul ferait `19602 × 60000`, soit
9,4 Go. `A = W_D M_D K⁻¹ BᵀC H^{-1/2}` s'applique par les opérateurs de champ
existants, avec `K` récupéré en sparse par **coloriage de son pochoir nodal
`3×3`** — dix-huit applications, à n'importe quelle taille, sans rien savoir des
internes de l'élément. M20 tombe de plusieurs minutes à `0,09 s`, M100 coûte
`5 s`. Validé avant usage : six valeurs singulières à `2,2e-7` du dense, angles
principaux du rang 2 à `8,5e-7°`, adjoint à `5,3e-14`.

Le balayage M20→M100 **détruit la belle structure de rang 2** : l'écart après le
mode 2 passe de `330` à `1,1` dès M40. La falaise était un effet de fenêtre.
À M100, soixante modes montrent une décroissance lente (`σ₁/σ₆₀ = 6,7`), sept
modes au-dessus d'un sigma, aucun au-dessus de trois. Réduction réelle, mais
troncature au bruit, pas rang structurel.

**Puis la projection des vrais résidus** (`893a9e1`). Deux corrections de
construction : le transfert s'applique au **modèle seul** (la mesure est déjà
passée par l'instrument), et la référence élastique ne demande aucun bloc de
couplage au bord — `u_el,int = u_DIC,int − K⁻¹ f_int` exactement. Comme `W_D`
blanchit, les projections sont **directement des z-scores**.

Blanchisseur vérifié avant toute lecture : norme blanchie de vrai bruit sur
`√(composantes intérieures)` = `1,109` à M20, `1,025` à M100.

À M20 le champ mesuré est élastique à une fraction du bruit près — `0,005σ` par
nœud à l'état 1, `0,543σ` à l'état 40. À M100 la signature est **détectée** :
test nul propre (`1,11σ` sur vingt modes à l'état 1), puis croissance monotone
jusqu'à `167σ`. Et les amplitudes sont physiquement justes : `a_j = c_j/σ_j`
donne `2,1e-3` à `6,6e-3` contre `5,67e-3` cumulé archivé.

**Enfin la séparation hétérogénéité / plasticité**, sans EBSD. Les motifs des
résidus normalisés sont parallèles tôt (cos `0,93–0,98` entre états 5, 10, 20)
et tournent tard (cos `0,27` entre 5 et 40). Un sous-espace de **rang 3** ajusté
sur les états 3-20 capture `99,70 %` de leur variance et annule tout le résidu
pré-plastique : `≤ 0,09` fois le bruit, aucun mode au-dessus de `1,3σ` jusqu'à
l'état 20. Une seconde composante apparaît **entre les états 20 et 25** et monte
à `21,6σ`.

Conséquence forte : l'eigenstrain équivalent des modes dominants tombe de
`2,1e-3…6,6e-3` à `1,1e-4…5,8e-4` après correction. **Neuf dixièmes de
l'amplitude « plastique » apparente étaient de l'hétérogénéité élastique.**
C'est le piège d'Eshelby rendu quantitatif : un eigenstrain tensoriel libre
reproduit exactement une inclusion élastique, donc rien dans l'espace plastique
ne peut les distinguer — seule la dynamique temporelle le peut.

Anomalie consignée sans explication : le résidu corrigé décroît de `2,015`
(état 35) à `1,131` (état 40).

Voir `validation/dic_excitation_of_observable_plastic_modes.md` et
`validation/tensor_plastic_observability_m20.md`.

**Dettes non soldées** : le mécanisme de projection Δp à committer, la baseline
Ludwik à rejouer.

#### Suite de la nuit : anatomie des modes, et le troisième dépouillement

Les deux dettes sont soldées. Le mécanisme de projection Δp est devenu une
option documentée `--admissible-fraction` (off par défaut), et la baseline
Ludwik a été rejouée sur **sa propre** trajectoire avec archivage des
déplacements manquants.

Ce rejeu **corrige ma conclusion précédente** : je mesurais l'histoire Ludwik
contre le mur en utilisant les déplacements de l'*oracle*, donc les incréments
d'une solution contre les états d'une autre.

| | trajectoire oracle | Ludwik rejouée |
|---|---:|---:|
| états touchant le mur | `0` / 40 | **`20` / 40** |
| pire `Δp / Δp_max` | `0,871` | **`3,509`** |

Le dépassement n'appartient donc pas seulement aux états perturbés par la
sonde : la trajectoire baseline elle-même sort du domaine à la moitié de ses
états, dont l'état 21. C'est pourquoi clipper la baseline seule suffisait.

**Anatomie des modes M100.** Tous les modes dominants sont concentrés **au
bord** : avec une bordure de 15 pixels, l'intérieur couvre `49 %` de l'aire mais
n'en porte que `0,094` au pire et `0,197` en médiane. Sous conditions de
Dirichlet, un eigenstrain proche du bord a le meilleur bras de levier sur le
déplacement intérieur, donc l'opérateur classe ces directions en tête.
Observables mathématiquement, mais c'est la condition aux limites qui parle.
Modes dominés par le cisaillement (parts moyennes `0,24 / 0,22 / 0,54`).

**Et la reconstruction ne tombe pas où le matériau plastifie.** Pic
`7,08e-3` contre un pic mesuré de `1,59e-2` — le bon ordre. Mais corrélation
avec la carte de déformation équivalente DIC de seulement `+0,149`, et part dans
le décile supérieur `0,134` contre `0,10` au hasard. En masquant une bande de 15
nœuds dans l'observation, la corrélation passe à `−0,150` : deux géométries
indépendantes, compatibles avec zéro.

C'est le **troisième dépouillement** de la même mesure, et chacun a retiré une
couche : les coefficients bruts ressemblaient à de la plasticité à la bonne
amplitude ; retirer le sous-espace d'hétérogénéité en a ôté neuf dixièmes ;
l'anatomie montre que ce qui reste vit au bord et non dans la bande.

**Conclusion à ce stade** : la détection à `21,6σ` est réelle, mais l'appeler
une reconstruction du champ plastique n'est pas soutenu par les données.

#### Test EBSD : le défaut n'est pas de l'hétérogénéité élastique cristalline

L'élasticité de référence est remplacée par la vraie : à chaque pixel, le
tenseur cubique **déjà déclaré par la loi FCC du dépôt** (`E = 99950,3 MPa`,
`ν = 0,388`, `G = 122000 MPa`, anisotropie de Zener `3,39`) est tourné en 3D par
l'orientation EBSD puis condensé exactement,
`C_ps = C_aa − C_ab C_bb⁻¹ C_ba`. Jauge, chaîne de mesure, conditions aux
limites, crop et diagnostics inchangés. Aucune plasticité cristalline, aucun
recalage.

Chaîne vérifiée d'abord : un cristal isotrope se condense sur
`plane_stress_elasticity` **à n'importe quel angle d'Euler**, à `4e-16` — ce qui
teste ensemble la rotation 3D, la convention de Voigt et le complément de Schur.
Symétrie cubique préservée à `7e-17`. Et sur une frontière affine à 1 %,
l'extension isotrope fluctue de `6e-20 mm` — le zéro exact que la théorie
impose — contre `6,18e-5 mm` RMS pour l'EBSD, soit `0,66 σ` DIC.

**Le premier crop était un mauvais test.** À `(1610, 1075)` un seul grain couvre
`70,4 %` de la fenêtre. Sous Dirichlet quasi affine, une raideur uniforme même
anisotrope rend le même champ. Crop rebalayé pour la diversité de grains et
`(1580, 1030)` retenu : grain dominant `20,6 %`, `7,2` grains effectifs.

**Sur ce vrai polycristal, le résidu ne bouge pas** : `5,873` → `5,875` à l'état
40, et le contrôle à orientations permutées est identique. Ce n'est donc pas une
question de mauvais arrangement spatial.

**La raison est l'orthogonalité, pas la petitesse.** La correction EBSD vaut
`2,6 %` du résidu à l'état 20 et `3,0 %` à l'état 40 — non négligeable — mais son
cosinus avec le résidu vaut `+0,0015` et `+0,0057`. Retrancher une composante
orthogonale ne peut pas réduire une norme, et la norme monte bien du
`√(1+ε²)` prévu. Seul `1 %` de la correction tombe dans le sous-espace précoce
de rang 3.

**Le confondeur principal identifié cette nuit est donc éliminé.** L'équivalence
d'Eshelby reste vraie en principe, mais l'élasticité cristalline réelle de cette
éprouvette ne produit pas le défaut observé. Ce que représente le sous-espace
précoce de rang 3 reste ouvert.

#### Le défaut précoce n'était pas de la mécanique — artefact de FFT périodique

Après avoir éliminé l'élasticité cristalline (orthogonale, cos `+0,006`) puis
**toute** hétérogénéité élastique effective linéaire — six canaux de Kelvin sans
dimension par pixel, un seul champ pour dix-huit états, angles principaux avec
le sous-espace précoce de `81,7°`, `88,4°`, `89,4°` — il restait à regarder les
trois motifs eux-mêmes.

Le premier porte `97,7 %` de la variance précoce. Tous trois placent `90` à
`98 %` de leur énergie dans une bande de huit nœuds couvrant `29 %` des nœuds,
sont à `73–83 %` sur `x`, et un ajustement affine en laisse `99 %` inexpliqué.
Une bande de bord, ni translation ni rotation ni gradient.

C'est la signature d'une FFT périodique. `DICSpectralTransfer.apply` filtre par
`fftn`, qui traite le crop comme périodique ; or un champ de déplacement sur une
fenêtre est dominé par une rampe affine, discontinue au raccord périodique.

Appliqué à un champ **purement affine**, qu'un passe-bas doit laisser intact :
`8,92e-4 mm`, soit **`9,49 σ` DIC**, avec `89,6 %` de l'erreur dans la bande de
bord et `91,7 %` sur `x`. Un champ constant passe à `1,9e-17` — la normalisation
est bonne, c'est spécifiquement la rampe.

Retirer la part affine avant filtrage et la remettre après est **exact** (un
champ affine est invariant sous tout passe-bas) et laisse les hautes fréquences
filtrées à l'identique : l'erreur tombe à `7,4e-18 mm`.

Reconstruit ainsi, le résidu perd **57 à 71 %** à tous les états — `5,873` →
`1,709` fois le bruit à l'état 40 — et passe **sous le bruit avant l'état 20**.
L'« hétérogénéité élastique précoce » cesse largement d'exister, ce qui explique
enfin pourquoi aucune élasticité ne pouvait en rendre compte.

Ce qui survit à l'état 40 vaut `1,71` fois le bruit, faible avant l'écoulement
et croissant après : bien meilleur candidat pour un signal inélastique.

`apply` est laissé inchangé — il définit tous les résultats archivés — et la
correction est disponible en `apply_without_wrap`, sur choix de l'appelant.
**Tout chiffre produit contre l'ancien transfert, y compris la fonctionnelle de
l'oracle, porte cet artefact et doit être recalculé.**

### 2026-08-15 (suite) — L'oracle plastique libre : ce qui est acquis, ce qui ne l'est pas

Session très longue sur `agent/plastic-observability`. Plusieurs conclusions
successives sont tombées sur des **défauts de méthode**, chacun consigné avec le
piège plutôt qu'effacé. Le fil est plus instructif que le résultat.

**Trois artefacts trouvés et corrigés.** `DICSpectralTransfer.apply` filtre par
FFT périodique : sur une rampe affine il fabrique `9,49 σ` d'erreur de bord, et
cela portait 57 à 71 % du « défaut mécanique ». Le résidu était comparé au bruit
**brut** alors qu'il vaut `(I − E P_b)n` : le bruit propagé est 18× plus petit,
donc mes rapports étaient sous-estimés d'autant. Et le gradient de la pénalité de
dissipation prenait l'adjoint d'une somme cumulée au lieu d'une différence —
faux à 98 %, optimiseur immobile, ce qui se lisait comme « la contrainte ne coûte
rien ».

**Ce qui est acquis.** Un eigenstrain plastique d'amplitude réaliste reproduit
**exactement** chaque incrément mesuré : `p_RMS = 1,374e-3`, pic `4,70e-3`,
contre un cumulé archivé de `5,67e-3`. L'exactitude est garantie d'avance (`A`
est surjectif sur les champs à bord nul), donc **seul le prix informe** — et il
est plausible.

**Ce qui ne l'est pas.** Le sous-espace partagé de rang 16 est de la
**compression, pas de la prédiction** : en leave-one-out une base construite sur
trois états ne reproduit le quatrième qu'à 26–63 %, et passer du rang 4 au rang
32 ne gagne qu'un dixième. La trajectoire libre erre `4,48×` plus qu'elle
n'avance et se dissipe à 48/52 — donc la longueur de trajet `6,2e-3`, proche du
`5,67e-3` archivé, est une **coïncidence d'échelle** due au zigzag, pas une
accumulation.

**Le nombre manquant** est le prix de l'admissibilité : l'erreur DIC de la
meilleure histoire vérifiant `D_kq ≥ 0`. Deux tentatives ont échoué, pour deux
raisons distinctes. Une pénalité quadratique sur `min(D,0)` a un minimiseur
trivial en zéro. Et un active-set **à ajout seul** est faux dès que les coupes
dépassent le nombre d'inconnues : traiter chaque inégalité active comme une
égalité sur-détermine et force la même solution triviale.

#### Ordre de travail arrêté avec l'encadrement

1. **Kelvin/Mandel** comme convention interne obligatoire — `[11, 22, √2·12]`
   pour contraintes **et** déformations, Voigt-ingénieur ne survivant qu'aux
   interfaces MFront/MGIS avec conversion au bord. `src/fem_inhouse/core/kelvin.py`
   pose la représentation et fixe les deux pièges vérifiés contre le code :
   `strain()` rend de l'ingénieur donc la ligne Kelvin est `B_shear/√2` **et non**
   `√2·B_shear` ; et `C^K = 2G` là où l'ingénieur a `G`. La migration du cœur
   reste à faire, avec rejeu des qualifications — le fit exact devrait être
   invariant, mais rang, QR/Krylov tronqués, normes de trajectoire et
   leave-one-out utilisaient une géométrie dépendante de la convention.
2. **QP dissipatif** par convexification séquentielle : `σ` gelée, QP creux avec
   cutting-plane, contraintes **reconstruites** à chaque boucle extérieure (une
   coupe bâtie sur `σ^(j)` n'est pas valide pour `σ^(j+1)`), départ à `a = 0`
   qui est faisable par construction. **Aucun solveur QP n'est installé** — ni
   OSQP, ni quadprog, ni cvxpy ; c'est un ajout de dépendance à décider.
3. **Replay Ludwik TwoSubcell**, avec deux histoires constitutives indépendantes
   par pixel et jamais d'interpolation depuis l'EBI.

#### Terminologie à corriger dans les notes

Parler de **contrainte dure de dissipation positive**, pas de « plasticité
thermodynamiquement admissible » : même avec `σ:Δε_p ≥ 0` partout, on n'a imposé
ni surface de charge, ni normalité, ni consistance. Ce qu'on démontrerait est
l'existence d'une **histoire d'eigenstrain localement dissipative** compatible
avec la DIC et l'équilibre.

### 2026-08-15 (fin) — Le déblocage : comparer par opérateur d'observation commun

**Le blocage annoncé était faux, et c'est la chose la plus utile de cette fin de
session.** J'avais écrit qu'un replay Ludwik exigeait un nouveau solveur sur
`TwoSubcellDiagnostic2D`, les cinématiques n'étant pas comparables. C'est vrai
des **champs internes aux points d'intégration**. Ce n'est pas vrai de la
cinématique nodale.

Vérifié : `Spectral2DResult` expose `displacement`, un champ **nodal** sur le
même `StructuredGrid2D`, et `TwoSubcellDiagnostic2D.strain()` prend précisément
un champ nodal. On peut donc définir un opérateur d'observation commun et
comparer `B_obs u_DIC` à `B_obs u_Ludwik^EBI` — sans interpolation, sans
conversion d'état matériau, sans mélange de points d'intégration. C'est le
principe de l'identification intégrée : les représentations internes n'ont pas à
coïncider, seule l'observation doit être la même.

Plusieurs jours de dérive évités. Le solveur EBI existant suffit.

**Ce qu'il reste à câbler** : `solve_ebi_dirichlet_plane_stress` veut un
`HookeanPlaneStressMaterialBatch` avec **un état par pixel**. Le seul appelant du
dépôt (`tests/unit/spectral2d/test_ebi.py`) utilise un stub élastique, donc il
n'existe pas de matériau Ludwik J2 prêt pour ce protocole. La route probable est
`create_plane_stress_material_batch` avec `mfront_behaviour_id="ludwik_j2"` et
`point_count = pixels²`, à condition qu'il expose `elastic_tangent_in_plane_mpa`.
C'est le premier point à vérifier.

#### La métrique à produire

`E_L = ‖ε_Ludwik − ε_DIC‖ / ‖ε_élastique − ε_DIC‖` aux états 25/30/35/40, plus
les erreurs par composante Kelvin et les cartes. Pas de blanchisseur dans
l'objectif principal.

#### Puis la hiérarchie qui remplace le rang

L'inverse change de rôle : il n'explique plus `ε_DIC − ε_élastique` mais
`r_L = ε_DIC − ε_Ludwik`, et `δε_p = 0` devient **la solution Ludwik** au lieu de
« pas de plasticité ». Trois modèles emboîtés, dans une base Kelvin orthonormée
pour la jauge plastique `Q^T G_p Q = I` :

* **A** — amplitude seule, `δε_p = δp·n_L` : Ludwik a-t-il la bonne direction et
  la mauvaise quantité ?
* **B** — plus les deux directions transverses : faut-il vraiment sortir de J2 ?
* **C** — tenseur libre, borne supérieure seulement.

La décomposition amplitude / direction est le résultat visé, pas « un champ
plastique ». Et la dissipation redevient traitable : Ludwik vérifie déjà
`σ:Δε_p^L > 0`, donc la correction dispose d'une marge au lieu de devoir
fabriquer toute la trajectoire.

#### Mis en pause explicitement

Le QP dissipatif libre — il mélange le prix de la dissipation et l'incapacité du
rang 16 à représenter les états non vus, donc il ne mesure pas ce qu'on veut. Le
raffinement du rang partagé. Tout nouveau travail sur le blanchisseur. Et
Méric/SRIX tant qu'on ne sait pas ce que Ludwik rate exactement.

### 2026-08-15 — Ludwik sur l'histoire mesurée : E_L > 1

Note complète : `validation/ludwik_on_the_measured_p43_history.md`.
Artefacts : `results/ludwik-two-state-replay-p0043/`.

**La brique existait déjà.** Aucun solveur à écrire, et pas même besoin de
l'opérateur d'observation commun imaginé la veille : `newton_two_state.py`
fournit `solve_two_state_dirichlet_plane_stress`, qui construit
`TwoSubcellDiagnostic2D` et `TraditionalTwoStateTriangleBatch` — deux histoires
constitutives indépendantes par pixel — et l'inverse construit la même
cinématique. Mesure, simulation et inverse partagent nativement le layout
`(nx, ny, 2, 3)`. Ce qui manquait était un artefact, pas une capacité : le
benchmark TRI2 existant pilote ce solveur avec une rampe proportionnelle sur un
autre crop, jamais avec l'histoire DIC.

**Le résultat est négatif et net.** `E_L` vaut 1,64 / 1,70 / 2,07 / 2,91 aux
états 25/30/35/40 : Ludwik dégrade l'accord en déformation d'un facteur 1,6 à
2,9 par rapport à ne rien faire. L'écart nodal reste à 0,1 % pour les deux
modèles — un crop en Dirichlet total est presque déterminé par son bord, toute
la discrimination est dans la déformation.

**L'amplitude est bonne, la distribution est fausse.** À l'état 40 la
déformation équivalente moyenne de Ludwik tombe à 2 % de la mesure, mais son
coefficient de variation vaut 0,77 contre 0,22 mesuré, et sa corrélation avec la
DIC est de 0,229 quand la solution **élastique**, sans aucune plasticité,
atteint 0,645. Ludwik corrèle à −0,569 avec la carte de limite d'élasticité, la
mesure seulement à −0,196 : le modèle localise dans les pixels mous de cette
carte, l'éprouvette non.

**Aucun recalage d'amplitude ne peut réparer cela.** La correction
`c = ε_L − ε_el` est orthogonale au défaut `g = ε_DIC − ε_el` : `cos(c,g)` vaut
+0,006 à +0,038 selon l'état, le meilleur facteur global est 0,005–0,02, et à
cet optimum `E_L = 0,999`. Deux produits scalaires suffisent à fermer
l'hypothèse amplitude au niveau global. Ils ne ferment **pas** l'hypothèse
amplitude ponctuelle : un champ `δp(x)` le long de `n_L` a vingt mille degrés de
liberté et peut annuler la correction là où le modèle localise à tort. C'est
maintenant l'expérience décisive, plus une formalité.

**Le garde-fou à lever avant d'interpréter.** Le défaut élastique vaut 0,29 de
la norme mesurée alors que l'accord nodal est à 0,1 % : une grande part est de
la texture à l'échelle du pixel. La référence de bruit propagé `(I − E P_b) n`
existe mais n'a pas été appliquée ici. Si le défaut élastique est largement du
bruit, « orthogonal » parle en partie du bruit et pas de Ludwik. `E_L > 1` et
l'écart de CV n'en dépendent pas — ce sont des propriétés du champ simulé.

**Deux incidents.** La politique « optimized » du benchmark TRI2
(Eisenstat-Walker + référence par itération de Newton), qualifiée sur 8
incréments proportionnels, ne converge pas à l'incrément 38 sur 40 incréments
d'histoire mesurée ; la politique conservatrice passe. Le solveur gagne un
`increment_observer` optionnel livrant déplacement, contrainte et déformation
plastique de chaque incrément convergé — le résultat ne garde que le dernier, et
le `progress_callback` ne transporte que des scalaires.

Suite : 1584 passés, 1 ignoré (les 7 SRIX-generic exigent
`SRIX_GENERIC_MFRONT_BEHAVIOUR_LIBRARY=build/srix-generic/src/libBehaviour.so`).

### 2026-08-16 — Point complet : l'identification plastique pilotée par la DIC

**Document de reprise à froid : `validation/dic_driven_plastic_identification.md`.**
Il contient tout — problème, chaîne mécanique, données et leur emplacement,
acquis, réfuté, hypothèse courante, jalons, et les pièges. À lire en premier
après cette entrée.

Résumé de ce qui a changé aujourd'hui.

**L'histoire DIC plein champ existe enfin** : 41 états 3600×3100, recalculés
depuis les 42 TIFF avec un paramétrage convergé (`finest_scale` 0, patch 4,
stride 1, alpha 15, epsilon 0,01, **100 itérations**). Le budget d'itérations
était le facteur dominant, pas le lissage ni la fenêtre. Les champs `U_40`/`V_40`
reçus portent la grille de patches non effacée : **ils ne sont pas convergés**,
et il ne faut pas chercher à les reproduire.

**Le plancher de bruit est mesuré** pour la première fois, via l'image répétée
de l'état final : 0,148 px, et 0,100 en EVM. Le signal dépasse le bruit
jusqu'à 2 px. Cela valide rétroactivement le défaut élastique de 0,29 sur lequel
repose le verdict Ludwik.

**Ludwik dégrade** : `E_L` = 1,64 → 2,91. Amplitude juste, distribution fausse,
correction orthogonale au défaut. La carte de limite d'élasticité localise là où
l'éprouvette ne localise pas.

**Aucune représentation réduite globale ne fonctionne.** POD par bande de
pyramide laplacienne : erreur de holdout 0,562 / 0,507 / 0,544 / 0,320 au rang
31, avec une erreur d'entraînement de **0,000** dans les quatre bandes. Le
repère de 0,115 que j'utilisais était un artefact de normalisation ; toute
comparaison qui s'y référait est nulle. Autoencodeur convolutionnel et neural
field échouent de même.

**Trente-deux états ne peuvent pas soutenir un holdout temporel** — 31 modes,
32 snapshots, aucune marge. La puissance statistique de ce jeu est spatiale.

**La question a donc changé** : non plus « le champ est-il de faible dimension »
mais « existe-t-il une règle locale spatialement transférable qui, couplée à
l'équilibre et à la thermodynamique, reproduit les hétérogénéités ». Ce qui doit
être compact n'est plus le champ mais le générateur.

Trois garde-fous à ne pas perdre : l'inpainting local n'est pas une loi locale
et exige une ligne de base ; P43 qualifie le logiciel, pas l'hypothèse ; et ne
jamais donner `(x,y)` au réseau, sinon il apprend une carte de l'éprouvette.

Et le rappel qui gouverne tout : **`A` est surjectif sur les champs à bord nul**,
donc ajuster la DIC est garanti et ne prouve rien. Ce qui brise la
dégénérescence est le partage des poids, pas l'équilibre.

### 2026-08-23 — SRIX-REGM : jumeau exact positif, transfert DIC négatif

Point de reprise unique : `validation/srix_regm_worklog.md`. La méthode rejoue
causalement SRIX sur la cinématique imposée, assemble le défaut faible intérieur
et calcule `delta_u = -K0^-1 f`, sans Newton global dans l'objectif.

Le jumeau exact M8 passe : vérité/initial/identifié = `1.474e-13`, `3.143e-8`,
`1.412e-13 mm`, erreur projetée `0.248 %`, gain mesuré `43x` sur une trajectoire
forward. M20/M100 confirment que MFront domine le coût et que `K0^-1` est déjà
négligeable. Ne pas développer de FFT maintenant.

Le Gate 4 est en revanche **négatif**. Après le transfert DIC qualifié, la
vérité vaut `2.132e-7 mm` mais un point très éloigné descend à `1.300e-7 mm`.
Avec bruit mesuré et whitening : `1.741e-3` contre `1.346e-3 mm`. Le transfert
préserve quatre directions sensibles mais ne préserve pas l'équilibre et
décale le minimum hors de la vérité. Voir
`validation/srix_regm_transfer_noise_results.md`.

Le classement exact de 20 lois contre 20 vrais solves FEMU passe : Spearman
`0,866`, Pearson logarithmique `0,878`, top-5 `3/5`, avec un gain médian
`4,94x`. Mais le classement après observation est **négatif** : le transfert
seul donne `0,326`, `0,276`, `2/5`. Le niveau bruité passe formellement alors
que son coût FEMU a un coefficient de variation de seulement `9,1e-5` : le
bruit domine, il ne sauve pas l'échec sans bruit. La règle pré-enregistrée
exigeait les deux niveaux.

**NO-GO avant P43.** Ne lancer ni P43-A ni M100 et ne publier aucun paramètre
SRIX identifié avec cet objectif. Point de reprise :
`validation/srix_regm_worklog.md`, résultats décisifs dans
`validation/srix_regm_femu_observed_ranking_results.md`. Toute reformulation
doit d'abord restaurer le minimum vrai sur le jumeau puis repasser le classement
observé. Ne pas développer le reconditionnement séquentiel avant d'avoir réglé
l'incohérence de l'opérateur d'observation.

### 2026-08-24 — SRIX-REGM : placement de l'opérateur d'observation

Une ablation pré-enregistrée a été exécutée sans nouveau solveur mécanique sur
les 20 candidats existants. Voir
`validation/srix_regm_observation_placement_preregistration.md`,
`validation/srix_regm_observation_placement_results.md` et le JSON primaire
`validation/reference_data/srix_regm_observation_placement_v1/report.json`.

Résultats contre le même classement FEMU observé :

- replay brut + score périodique : Spearman `0,950` ;
- replay brut + score affine-preserving : `0,940` ;
- replay transféré + score identité : `0,338` ;
- replay transféré + score périodique : `0,290` ;
- replay transféré + score affine-preserving (chemin actuel) : `0,326`.

Le biais à la vérité du replay transféré vaut `4,067e-7 mm`, contre une
dispersion paramétrique de `3,495e-8 mm`, soit un rapport `11,64`. Le défaut
dominant est donc l'injection de `O(u*)` avant le replay SRIX, qui modifie la
trajectoire constitutive non linéaire. L'application de `O` au pseudo-
déplacement est secondaire sur ce jumeau ; la variante périodique n'améliore
pas la conclusion scientifique.

**Décision :** le NO-GO avant P43 reste inchangé. Toute reformulation future
doit conserver une histoire mécanique latente pour le replay constitutif et
qualifier séparément sa reconstruction depuis la DIC sur le jumeau. Ne pas
lancer P43 ni ajuster SRIX avant de repasser le classement observé.

### 2026-08-24 — Borne supérieure twin par modes cinématiques latents

Le test suivant a utilisé uniquement le jumeau exact M8. La différence
`u* - O(u*)` a été décomposée par POD sur les états, puis réintroduite par rang
avant le replay SRIX. Le score est resté `O(delta_u)` et aucun nouveau FEMU
n'a été lancé.

La corrélation observée REGM/FEMU évolue ainsi : `k=0: 0,326`, `k=3: 0,577`,
`k=4: 0,708`, `k=5: 0,859`, rang complet `162: 0,940`. Les cinq premiers modes
portent `99,9897 %` de l'énergie de la différence et donnent aussi un
Pearson logarithmique `0,888` et un recouvrement top-5 `4/5`.

Ce résultat soutient une hypothèse de cinématique latente de faible dimension,
mais c'est une **borne supérieure** : la base et ses coefficients utilisent
`u*`, inconnu sur P43. Il ne s'agit pas encore d'une reconstruction DIC. Le
prochain gate doit construire les modes à partir de `O`, `K0` et du modèle de
bruit uniquement, puis repasser le même test twin sans utiliser `u*` pour
définir la base. Le NO-GO P43 reste inchangé.

### 2026-08-24 — Test de projection mécanique avant replay SRIX

Une première voie de reconstruction sans vérité latente a été testée sur le
jumeau transféré. À chaque état, le replay SRIX du preset fournit la correction
existante `-K0^-1 B^T sigma`; cette correction est ajoutée à l'histoire observée
avec amortissement `0,25`, `0,50` ou `1,00`, une ou deux fois. Les bords restent
inchangés et l'état constitutif est rejoué causalement.

Le résultat est négatif pour cette formulation simple : Spearman reste entre
`0,326` et `0,341`, le recouvrement top-5 reste `2/5`, alors que le résidu à la
vérité diminue de `2,132e-7` à `1,381e-7 mm`. La correction rend donc la
cinématique plus équilibrée selon la loi de référence, mais ne restaure pas
l'information nécessaire au classement des paramètres.

Artefacts :
`validation/srix_regm_mechanical_projection_preregistration.md`,
`validation/srix_regm_mechanical_projection_results.md` et
`validation/reference_data/srix_regm_mechanical_projection_v1/report.json`.

**Décision :** rejeter `u_observed + damping * (-K0^-1 R)` comme méthode de
production. Une projection contrainte par l'écart d'observation, le résidu
mécanique et la covariance DIC reste éventuellement testable sur un jumeau,
mais P43 demeure bloqué.

### 2026-08-24 — Géométrie locale d'information REGM/FEMU

Le diagnostic suivant a comparé les Jacobiennes en coordonnées
`(log(tau0), log(R), log(Q), log(b))` au point vrai du jumeau M8 : REGM sur la
cinématique exacte, REGM après transfert DIC, et FEMU directe observée obtenue
par huit résolutions perturbées. Les spectres normalisés sont :

- REGM exact : `1, .422, .0324, 4.65e-5` ;
- REGM observé : `1, .337, .0178, 1.27e-5` ;
- FEMU observée : `1, .542, .407, .0679`.

Le conditionnement FEMU vaut `14,7`, contre `2,15e4` et `7,90e4` pour REGM.
L'angle entre les sous-espaces dominants de dimension deux REGM exact/FEMU est
`67,2 degrés`, alors qu'il n'est que `0,81 degré` entre REGM exact et REGM
observé. La chaîne DIC déforme donc la géométrie REGM, mais REGM exact ne
reproduit déjà pas la géométrie locale de la FEMU.

La corrélation FEMU `Q/b` est forte (`0,933`) sans être une disparition
complète : la quatrième direction reste à `6,8 %` de la première. Il faut donc
éviter de conclure trop vite que P43 ne porte que deux paramètres. Le rapport
complet est `validation/srix_regm_information_geometry_results.md`; aucun
calcul P43 n'est autorisé avant une reformulation qui repasse ce gate.

### Tangente algorithmique : résultat et suite obligatoire (2026-08-24)

Le diagnostic `validation/reference_data/srix_regm_algorithmic_tangent_v1/report.json`
remplace `K0` par la tangente algorithmique consistante de SRIX à chaque état,
sans Newton global. Résultat : spectre normalisé `(1, .37594, .03469,
8.62e-5)`, conditionnement `1.16e4`, angle principal avec FEMU `73.9 deg`.
Le `K0` fixe donnait `(1, .42199, .03240, 4.65e-5)`, `2.15e4` et `68.4 deg`.
La tangente seule ne rétablit donc pas les directions FEMU manquantes.

Suite autorisée, et uniquement sur le twin : tester un rejeu séquentiel à une
correction par incrément. À chaque incrément, utiliser la tangente d'essai pour
calculer une correction, réévaluer SRIX sur le déplacement corrigé, puis
committer avant l'incrément suivant. Ne pas converger Newton, ne pas lancer
P43, et comparer les quatre géométries `REGM-K0`, `REGM-Kalg`, `SREGM-1Newton`
et `FEMU`. Si cette variante ne rejoint pas la géométrie FEMU, arrêter les
surrogates REGM et documenter le NO-GO.

### Rejeu séquentiel : gate final des surrogates REGM (2026-08-24)

Le diagnostic `validation/reference_data/srix_regm_sequential_one_newton_v2/report.json`
applique exactement cette variante sur le twin M8 : une seule correction
tangentielle, réévaluation SRIX, commit causal, puis incrément suivant. Le
spectre obtenu est `(1, .56251, .05764, 2.30e-4)`, conditionnement `4.35e3`,
angle principal FEMU `45.24 deg`, mais angles de rang deux `67.91 deg` et
`12.38 deg`. Les directions faibles FEMU restent absentes.

Conclusion : amélioration réelle mais gate négatif. La chaîne `REGM-K0` →
`REGM-Kalg` → `SREGM-1Newton` ne reproduit pas la géométrie locale FEMU.
Ne pas lancer P43 ni présenter un paramètre SRIX identifié. Toute suite doit
être une méthode de sensibilité tangentielle validée ou un objectif réduit
explicitement justifié, avec nouveau jumeau et corrélation FEMU avant les
données expérimentales.

### Correction de l'observable séquentielle (2026-08-24)

Le rejeu séquentiel a été audité après le résultat v2 : le script score la
correction du dernier incrément, alors que la comparaison à une Jacobienne FEMU
doit utiliser l'écart de déplacement accepté cumulé à chaque endpoint. Un
rejeu v3 conserve les deux observables, sans modifier v2 :
`validation/reference_data/srix_regm_sequential_one_newton_v3/report.json`.

Résultats : correction seule `(1, .56251, .05764, 2.30e-4)`; écart cumulé
`(1, .46460, .09381, 2.17e-4)`; FEMU `(1, .54152, .40668, .06787)`. Le cumul
relève le troisième mode mais pas le quatrième et son angle principal FEMU est
`74.67 degrés` (contre `45.24 degrés` pour la correction seule). La correction
de scoring était nécessaire, mais le rejeu séquentiel reste un échec du gate
de géométrie; P43 demeure interdit. Voir
`validation/srix_regm_sequential_one_newton_cumulative_results.md`.

### Clôture REGM et prochain gate FEMU direct (2026-08-24)

`E-SRIX-REGM-009` clôt définitivement la voie REGM comme voie
d'identification. Le test cumulatif était le bon observable, mais il utilisait
encore le discretisé mécanique REGM :
`TensorPlasticObservabilityOperator`, `weak_equilibrium_residual` et
`_assemble_sparse_stiffness`. Il ne différenciait donc pas le résidu du
solveur FEMU M8.

Le prochain gate est `E-SRIX-FEMU-DIRECT-001`. Il doit réutiliser exactement
le chemin matrix-free de `solve_two_state_dirichlet_plane_stress` :
`TraditionalTwoStateTriangleBatch.tangent_action_into`, la divergence de
`TwoSubcellDiagnostic2D`, le packing des DOFs libres, l'extension de Dirichlet
et le Krylov/EBI du forward. Aucune routine REGM ne doit intervenir dans la
sensibilité globale. La première comparaison est l'égalité des quatre
colonnes à la Jacobienne FEMU FD archivée, avant toute SVD.

### Audit du chemin adaptatif et gate FD à chemin figé (2026-08-24)

L'audit `validation/srix_femu_fd_adaptive_path_audit.md` montre que la FD
FEMU archivée n'est pas une dérivée à chemin discret fixe : la base accepte 338
incréments, contre 326 pour `Q+` et 328 pour `Q-`, avec des noeuds internes
différents. Elle reste une provenance utile, mais ne peut pas être l'oracle
primaire d'une sensibilité tangentielle.

Le nouveau driver
`scripts/qualify_srix_femu_fixed_path_gate.py` construit donc une FD sur la
séquence `LoadPathStep` acceptée par la base. Les essais h=`3e-3`, `1e-3` et
`1e-4` n'ont pas encore produit d'oracle complet : les trajectoires perturbées
échouent respectivement aux incréments 18, 5 et 12 du chemin figé, même avec
Newton-80, vingt réductions de line-search et un prédicteur initial issu de la
base. Ce résultat est enregistré comme blocage numérique, pas comme NO-GO
scientifique de la sensibilité directe.

Conséquence : le rejeu propre `validation/reference_data/srix_femu_direct_sensitivity_v2/report.json`
contre la FD adaptative (erreurs de colonnes `0.942, 0.967, 0.997, 0.998`) ne
constitue pas un gate valide. Le
prochain travail autorisé est de rendre l'oracle commun numériquement robuste
(chemin raffiné commun ou diagnostic de branche), puis seulement de comparer
les colonnes brutes. P43, optimisation et dérivée analytique MFront restent
interdits.

Une première tentative de chemin commun uniformément raffiné (deux sous-pas
par incrément accepté) à h=`1e-3` a également bloqué à l'incrément 407/676.
L'artefact est `validation/reference_data/srix_femu_fixed_path_gate_ref2_h1e3_v1/report.json`.
La prochaine investigation doit donc qualifier la branche Newton et la
continuité du chemin, plutôt que réduire encore aveuglément le pas FD.

### Chemin commun synchronisé : premier diagnostic (2026-08-24)

Le driver `scripts/qualify_srix_femu_common_path_gate.py` implémente l'union
des fractions acceptées par les neuf trajectoires puis la bisection synchronisée
des intervalles en échec. Le premier lancement M8 a terminé les directions
jusqu'à `b_plus`, mais `b_minus` est resté plus de 40 minutes dans le solveur
adaptatif sans fournir son chemin. Il a été interrompu proprement : ce n'est
pas une conclusion sur la sensibilité directe, mais un diagnostic de coût/
branche du chemin adaptatif.

Le driver possède maintenant un timeout par trajectoire et produit un rapport
`blocked_adaptive_trajectory_timeout` au lieu de rester indéfini. Aucun chemin
commun ni aucune comparaison FD n'est déclaré tant que les neuf trajectoires ne
sont pas disponibles.

### Recherche de chemin commune à trois niveaux (2026-08-24)

Le prochain rejeu du gate utilise trois politiques explicitement séparées dans
`scripts/qualify_srix_femu_common_path_gate.py` :

* `_seed_config` est exploratoire uniquement (tolérance `1e-5`, Newton 12,
  croissance `2`, seuil de line-search difficile `0.25`, timeout 60 s par
  trajectoire) ;
* `_path_search_config` est fail-fast et strict sur l'équilibre (tolérance
  `1e-6`, Newton 12, six réductions de line-search) ;
* `_oracle_config` est la seule configuration scientifique (tolérance `1e-6`,
  vérification finale, Newton 80, vingt réductions de line-search).

Le défaut historique du contrôleur reste inchangé :
`line_search_difficult_threshold=1.0`. Le seuil `0.25` n'est autorisé que pour
le seed et ne peut donc pas modifier les résultats de production.

Les fractions seed sont mises en cache avec validation stricte de la SHA Git,
de la bibliothèque MFront, de l'historique de bord, du maillage, des threads,
du pas FD et de la configuration seed, sous
`validation/reference_data/srix_femu_common_path_cache/`. Le seed `b_minus`,
identifié comme trajectoire très coûteuse, est ignoré par défaut : il est
qualifié directement dans la recherche de chemin commun, puis dans le rejeu
oracle strict. Utiliser `--include-b-minus-seed` seulement pour une expérience
exploratoire explicite.

La recherche bissecte un seul intervalle en échec à la fois et ne rejoue pas les
neuf variantes après chaque bisection. Les limites sont dix bisections locales,
`1/65536` par intervalle et un budget global configurable. Un échec de seed ou
de recherche reste un diagnostic de branche/coût ; aucune comparaison FD ni
aucun résultat scientifique n'est déclaré avant le rejeu strict des neuf
variantes sur une partition identique.

Le rejeu v7 a finalement passé ce gate sur M8 après reprise du chemin strict
qualifié : erreurs L2 des quatre colonnes `1.656e-3`, `6.964e-4`, `6.105e-4`
et `6.166e-4`, cosinus tous supérieurs à `0.9999989`. Cela valide la
différentiation directe contre une FD construite sur le **même chemin discret**.
Le spectre commun reste `(1, 0.1871, 0.04053, 5.35e-5)`, différent du spectre
de l'ancienne FD adaptative `(1, 0.542, 0.407, 0.0679)` : ce résultat ne
réhabilite pas l'ancienne géométrie et n'autorise toujours ni P43 ni
l'identification. L'artefact principal est
`validation/reference_data/srix_femu_common_path_gate_v9/` (rapport propre,
`dirty=false`).

### PATH-002 : raffinement bloqué (2026-08-24)

Le gate `E-SRIX-FEMU-PATH-002` a été implémenté pour comparer les chemins
emboîtés 57, 114 et 228 pas. Le niveau 57 est recalculé avec succès, mais le
forward de base échoue au pas 34 du chemin 114, sur l'intervalle de fraction
`[0.236328125, 0.23828125]`. Les essais isolés avec 80, 120 et 160 itérations
Newton échouent au même endroit : ce n'est pas un simple plafond d'itérations.

Le résultat est enregistré comme `blocked_path_level`, sans extrapolation du
spectre et sans autorisation d'identification :
`validation/reference_data/srix_femu_path_convergence_v2/`. Le prochain travail
doit diagnostiquer cette branche ou définir une subdivision locale
préréférencée qui converge avant de comparer les limites 57/114/228.

### BRANCH-002A : diagnostic local autour de f≈0.237 (2026-08-24)

Le diagnostic `scripts/qualify_srix_femu_branch_local.py` teste le parent
`[0.234375, 0.23828125]` avec cinq positions de midpoint (`alpha=0.25, 0.40,
0.50, 0.60, 0.75`) et deux prédicteurs globaux. Les cinq partitions locales
convergent, y compris `alpha=0.50`, qui correspond au demi-pas bloqué dans le
raffinement global 114. L'échec 114 ne peut donc pas être attribué à ce
demi-pas isolé : il dépend de l'histoire obtenue après le raffinement des
autres intervalles.

À l'endpoint `f=0.23828125`, les écarts relatifs au chemin 57 sont de
`8.09e-6`–`1.09e-5` en déplacement, `7.30e-5`–`9.77e-5` en contrainte et
`4.97e-4`–`6.66e-4` en déformation plastique. Les partitions convergentes
restent proches, mais leur dispersion n'est pas nulle (`2.80e-6`, `2.47e-5`,
`1.70e-4`). Il n'y a donc pas de branche constitutive distincte démontrée,
mais la convergence par raffinement global n'est pas établie.

Les prédicteurs extrapolé et coarse-endpoint échouent respectivement aux
incréments 44 et 18. Ils ne copient pas l'état constitutif coarse et ne
permettent pas de conclure sur une bifurcation matérielle. Les tableaux SRIX
bruts `g/p/a` ne sont pas encore exposés par `TwoStateIncrementFields`.

Artefacts : `validation/srix_femu_branch_local_results.md` et
`validation/reference_data/srix_femu_branch_local_v3/` (rapport et figure,
`dirty=false`).
Statut : `path_convergence_authorized=false`, `identification_authorized=false`
et `p43_authorized=false`. Ne pas lancer de campagne 228, d'identification ou
de P43 avant une qualification de convergence et un diagnostic constitutif
local plus complet.

### BRANCH-002B : localisation causale de la divergence (2026-08-24)

Le gate `scripts/qualify_srix_femu_branch_causal.py` expose désormais les
observables SRIX déjà disponibles dans le bridge MFront : `plastic_slip`,
`equivalent_plastic_slip`, `back_strain` et `elastic_strain`. Il compare les
endpoints communs 57/114 et teste les préfixes dont les `k` premiers intervalles
sont raffinés (`k=8,16,24,32,40,48,57`).

Ce diagnostic a d'abord révélé un bug de plomberie : une initialisation pleine
du champ `initial_displacement` annulait aussi les valeurs de bord du premier
pas. Le solveur ne doit appliquer cette initialisation qu'aux inconnues
intérieures. La correction est couverte par un test unitaire ; les campagnes
historiques restent inchangées.

Avec l'initialisation corrigée, le chemin 57 échoue à l'incrément 5
(`f=0.15625`) et le chemin entièrement raffiné à l'incrément 28
(`f=0.2421875`). Seuls quatre endpoints communs sont donc comparables. La
divergence est négligeable à `f=0.03125` et `0.0625`, puis devient visible à
`f=0.09375` (`1.90e-3` en contrainte, `0.331` en `g`) ; un changement
d'activité est observé à `f=0.125`. Il s'agit d'un candidat de transition
d'ensemble actif, pas encore d'une bifurcation constitutive démontrée.

Tous les préfixes testés échouent dans la configuration corrigée. Le statut
reste donc `unresolved` : aucune reprise de PATH-002, identification ou P43
n'est autorisée. Artefacts :
`validation/srix_femu_branch_causal_preregistration.md`,
`validation/srix_femu_branch_causal_results.md` et
`validation/reference_data/srix_femu_branch_causal_v5/` (`dirty=false`).

### COMMON-PATH-001R : re-baselining après correction des Dirichlet (2026-08-24)

Un bug sémantique a été corrigé dans `solve_two_state_dirichlet_plane_stress` :
`initial_displacement` ne doit jamais remplacer les DOF de bord prescrits au
premier incrément. Le prédicteur est maintenant appliqué uniquement aux DOF
intérieurs ; les bords viennent toujours de `boundary_state`. Le test
`test_initial_displacement_guess_does_not_cancel_first_boundary_step` verrouille
ce contrat, ainsi que l'équivalence du cas sans prédicteur pour un petit cas
élastique.

Le cache `validation/reference_data/srix_femu_common_path_cache/` a été
explicitement invalidé par `fixed_path_initialization_contract=2`. Les chemins
v9 et PATH-002 v2 ne sont pas supprimés, mais sont superseded pour toute
interprétation scientifique. Leur spectre, notamment `(1, 0.187, 0.0405,
5.35e-5)`, ne doit plus être cité comme propriété du forward corrigé.

Le nouveau gate est preregistré dans
`validation/srix_femu_common_path_rebaseline_preregistration.md`. Il reconstruit
un chemin commun depuis une proposition non qualifiée, par bisection locale
fail-fast, puis exige la convergence stricte de la base et des huit perturbations
avant de recalculer la Jacobienne directe et sa FD commune. Aucun P43 ni aucune
identification n'est autorisé avant ce passage. Le résultat courant est suivi
dans `validation/srix_femu_common_path_rebaseline_results.md`.

Le rejeu corrigé v16 était limité par son budget de 12 bisections. La reprise
v17 (`validation/reference_data/srix_femu_common_path_gate_v17/`) a utilisé v16
comme proposition non qualifiée, inséré 25 nœuds et obtenu un chemin commun de
94 pas. Les neuf trajectoires convergent sous l'oracle strict (`dirty=false`,
commit `387af84`).

La Jacobienne directe est qualifiée contre la FD du même chemin : erreurs L2
relatives `(3.95e-4, 9.71e-4, 7.94e-5, 7.96e-5)` et cosinus tous supérieurs à
`0.9999995`. Le spectre corrigé est `(1, 0.18020, 0.04029, 6.32e-5)` et le
conditionnement `1.58e4`. Cela valide la différentiation directe sur ce chemin
discret, mais pas encore la convergence de la géométrie quand le chemin est
raffiné. L'identification et P43 restent interdits jusqu'au gate de convergence
de chemin.

### PATH-002R : convergence imbriquée après re-baselining (2026-08-24)

Le gate `E-SRIX-FEMU-PATH-002R` est archivé dans
`validation/reference_data/srix_femu_path_convergence_v3/` et préréférencé dans
`validation/srix_femu_path_convergence_rebaseline_preregistration.md`. Il
compare uniquement un forward de base et une Jacobienne directe aux niveaux
L0/L1/L2 ; aucune nouvelle FD globale n'est utilisée.

L0 est le chemin v17 de 94 pas. L1 impose les 94 midpoints et converge à 188
pas. L2 impose ensuite les midpoints de L1 et converge à 392 pas après 16
réparations locales strictes. Le forward observé varie de `2.176e-4` entre L0
et L1, puis `9.514e-5` entre L1 et L2.

Le gate principal L1→L2 est toutefois négatif : les colonnes `log(tau0)` et
`log(R)` varient encore de `3.67 %` et `4.20 %`, et l'angle maximal du
sous-espace de rang 3 vaut `2.606°` (seuils préenregistrés : 2 % et 2°). Les
trois premiers rapports singuliers changent de `0 %`, `0.95 %` et `3.11 %` ; le
quatrième reste de l'ordre de `6.2e-5` et aligné à `0.999987` sur le contraste
`Q-b`.

Le forward est donc proche de la convergence, mais la géométrie différentielle
ne l'est pas encore selon les seuils fixés. Le résultat est documenté comme
négatif, sans ajustement des seuils. Identification et P43 restent interdits.

### PATH-002S : extension L3 finale (2026-08-24)

L'extension finale a été pré-enregistrée dans
`validation/srix_femu_path_convergence_extension_preregistration.md`. Elle
réutilise L0--L2 qualifiés et ne recalcule que L3. Le chemin L3 converge avec
809 pas effectifs, dont 25 réparations locales.

Le forward L3 est donc encore stable, mais la construction de la Jacobienne
directe échoue pendant le replay d'une histoire constitutive shadow :
`MFrontIntegrationError: 3D MFront integration failed with status -1`. Il n'y a
donc pas de métrique L2→L3 de sensibilité et le gate est bloqué au stade
constitutif, sans conclusion de convergence ou de non-identifiabilité.

Ce résultat est archivé dans
`validation/reference_data/srix_femu_path_convergence_v4/report.json` et
`validation/srix_femu_path_convergence_extension_results.md`. C'était le dernier
niveau de raffinement pré-enregistré : aucun L4 ne doit être lancé. La prochaine
étape autorisée est un diagnostic ciblé du replay shadow/MFront, puis seulement
une décision sur l'identification.

### SHADOW-003 : localisation du blocage L3 (2026-08-24)

Le diagnostic pré-enregistré dans
`validation/srix_femu_shadow_diagnostic_preregistration.md` a instrumenté
séparément les deux phases du calcul direct : `fixed_current_strain` et
`history_advance`. Le chemin L3 exact de 809 incréments converge pour le
forward, mais le premier shadow fautif est maintenant localisé :

- incrément accepté 271 ;
- fraction `[0.232177734375, 0.2322998046875]` ;
- paramètre `tau0`, signe `minus` ;
- phase `fixed_current_strain` ;
- MFront `status -1`.

La phase `history_advance` n'est pas atteinte dans ce run. Sur L2 (392 pas),
les trois valeurs diagnostiques `h = 0.003`, `0.0015` et `0.001` passent toutes
avec 1568 résolutions GMRES ; réduire `h` ne répare donc pas le phénomène.
Cette étude ne compare pas les matrices des trois essais et n'adopte aucun
nouveau `h`.

L'artefact machine-readable est
`validation/reference_data/srix_femu_shadow_diagnostic_v1/report.json` et la
note d'interprétation est
`validation/srix_femu_shadow_diagnostic_results.md`. Le résultat est classé
comme limitation locale du replay shadow sur chemin très raffiné, pas comme
échec du forward mécanique. Aucun L4, aucune identification et aucun P43 ne
sont autorisés ; la suite possible est un rejeu local de `tau0−` avec
télémétrie MFront ou le provider de sensibilité constitutive analytique.

Le sous-gate `SHADOW-003B` a ensuite rejoué le préfixe L3 jusqu'au pas 271. Avec
`h=0.003`, `tau0−` échoue encore au même appel ; avec `h=0.0015` et `h=0.001`,
le pas 271 passe. Le contrôle L2 complet est favorable aux deux valeurs : par
rapport à `h=0.003`, l'erreur maximale de colonne est respectivement `0.204 %`
et `0.246 %`, avec des cosinus minimaux `0.9999981` et `0.9999973`; le spectre
normalisé et le conditionnement (`1.60e4`) restent stables. Ces valeurs sont
donc des candidates pour un rejeu L3 complet, mais aucune n'est adoptée avant
ce rejeu. Les matrices et le détail sont dans
`validation/reference_data/srix_femu_shadow_diagnostic_v1/l2_jacobian_h_sweep.*`.

Le gate `SHADOW-003C` a exécuté les deux replays L3 complets sur le chemin de
809 pas. `h=0.0015` termine en 227,44 s et `h=0.001` en 210,25 s, avec 3236
résolutions GMRES chacun. Les deux Jacobiennes sont cohérentes : erreur
maximale entre colonnes `0,102 %`, cosinus minimal `0,99999957`, angle de rang
3 `0,136°`. Le pas `h=0.0015` est donc validé comme candidat principal et
`h=0.001` comme contrôle.

Le gate de convergence L2→L3 reste néanmoins négatif selon les seuils
PATH-002S, uniquement sur l'angle du sous-espace de rang 3 : `2,290°` pour
`h=0.0015` et `2,195°` pour `h=0.001`, contre un seuil de `2°`. Le forward
(`4,71e-5`), les erreurs de colonnes (`<2 %`), les cosinus et les spectres
passent. La stabilité en `h` est donc acquise, mais la convergence de la
géométrie L2→L3 ne l'est pas ; aucune identification ni P43 n'est autorisée.
Le résultat est archivé dans
`validation/reference_data/srix_femu_shadow_h_l3_v1/report.json` et
`validation/srix_femu_shadow_h_l3_results.md`.

La stabilité L3 autorise l'adoption de `h=0.0015` comme pas principal du
shadow FD ; `h=0.001` reste le contrôle. Cette adoption ne lève pas le blocage
PATH-002S : son angle de rang 3 reste légèrement au-dessus du seuil
pré-enregistré.

### P43-SYNTH-001/A : premier smoke test synthétique (2026-08-25)

Le premier test synthétique P43 autorisé après SHADOW-003C utilise un crop M20
réel (`[1610:1630,1075:1095]`), l'EBSD P43 réel, l'histoire DIC réparée, une
partition fixe de 32 incréments, une observation identité et aucun bruit. Il
ne concerne pas le P43 expérimental. Le script est
`scripts/qualify_srix_p0043_synthetic_smoke.py`, le preregister est
`validation/p0043_synthetic_identification_preregistration.md` et le résultat
machine-readable est
`validation/reference_data/p0043_synthetic_identification_v1/report.json`.

Avec la vérité `(tau0,R,Q,b)=(40,18.781910,10,3)`, un départ
`(42,17.842815,10.8,2.76)` atteint un RMS whitened de `1.269e-13` contre
`8.850e-8` initial. Après 7 forwards et 7 Jacobiennes (`438.9 s`), les
paramètres sont `(40.000004,18.781912,10.006888,2.997907)`. L'optimiseur
signale `maximum number of function evaluations exceeded` car la limite
préenregistrée de six évaluations est atteinte ; ce n'est pas une preuve de
convergence globale multi-départs.

La SVD finale est `(1,0.135725,0.036116,1.031e-4)`, conditionnement `9696.6`.
La corrélation `Q/b` vaut `0.999999997` et le quatrième vecteur est le contraste
`Q-b`. La réponse synthétique et les trois directions robustes sont donc
retrouvées dans ce smoke test, mais `Q` et `b` ne sont pas revendiqués comme
identifiés séparément. La suite autorisée est uniquement P43 synthétique :
départs plus éloignés, puis tests à trois et quatre paramètres et mismatch
contrôlé. Aucun transfert/bruit DIC ni P43 expérimental n'est autorisé à ce
stade.

Le gate `P43-SYNTH-002B` est lancé sur quatre départs déterministes éloignés
(`±20--25 %`) avec `max_nfev=15`; son artefact sera
`validation/reference_data/p0043_synthetic_multistart_v1/report.json`. Une
montée M100 est planifiée automatiquement après la création de ce rapport :
le meilleur jeu M20 initialise le crop enregistré `[1580:1680,1030:1130]`,
sans changer la vérité SRIX ni l'histoire. Le script est
`scripts/qualify_srix_p0043_synthetic_scaleup.py` et la limite initiale est de
quatre évaluations pour éviter de transformer le diagnostic nocturne en
campagne non bornée. Le résultat sera dans
`validation/reference_data/p0043_synthetic_scaleup_v1/`; aucune conclusion sur
le P43 expérimental n'est autorisée.

### Résultats P43-SYNTH-002B et P43-SYNTH-003 (2026-08-26)

Les quatre départs éloignés M20 ont convergé. Les RMS finaux sont `5.26e-17`,
`4.98e-13`, `4.84e-17` et `7.25e-17`. Trois départs retrouvent la vérité à
la précision numérique ; B2 termine à `Q=9.973054`, `b=3.008216` avec un
résidu quasi nul, ce qui confirme la vallée faible `Q-b`. Les projections de
l'erreur restent faibles dans les trois directions fortes et se concentrent
sur le quatrième vecteur singulier.

Le scale-up M100 enregistré de façon indépendante a ensuite convergé en 3
évaluations et `18282.4 s` (~5.08 h), en partant du meilleur jeu M20. Le RMS
passe de `3.38e-16` à `3.25e-18`, avec
`(tau0,R,Q,b)=(40.000000,18.781910,10.000000,3.000000)`. Sa SVD est
`(1,0.41608,0.05539,1.4307e-4)` et `rho(Q,b)=0.9999999999`. Cette réussite
est synthétique uniquement ; C (paramètres réduits), bruit, transfert DIC et
P43 expérimental restent à faire.

### E-SRIX-P43-SYNTH-SVD-001 : C1/C3 M20 (2026-08-26)

L'infrastructure générique `svd_parameter_basis.py` et ses trois tests unitaires
sont ajoutées. C1 utilise la base SVD M20 à la vérité, avec rang fixé à 3 et
`eta=eta_ref+V3 z`; `Q` et `b` restent donc tous deux variables. Les quatre
départs B1--B4 convergent avec RMS `8.65e-18`, `3.32e-16`, `9.56e-18` et
`2.45e-17`; l'erreur rang-3 maximale est `1.61e-9` en log-coordonnées.

Le spectre M20 est `(1,0.135725,0.036116,1.031e-4)`. L'alignement de `v4` avec
`log(Q)-log(b)` vaut `0.999977`, celui de `v3` avec `log(Q)+log(b)` vaut
`0.998844`. C3 profile `z4` sans réoptimiser `z1..z3` : le RMS reste entre
`8.65e-18` et `4.53e-11` sur `z4∈[-0.3,0.3]`, tandis que `Q` varie de `8.10`
à `12.35` et `b` de `3.71` à `2.42`. La paramétrisation SVD rang 3 est donc
qualifiée sur le jumeau M20 et la direction `Q/b` est pratiquement une jauge
non observable. C2 dynamique et C4 Q-fixé/b-fixé restent optionnels ; aucun
M100 supplémentaire ni P43 expérimental n'est autorisé.
