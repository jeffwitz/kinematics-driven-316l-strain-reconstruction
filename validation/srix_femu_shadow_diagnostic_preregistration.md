# E-SRIX-FEMU-SHADOW-003 — pré-enregistrement

## Objet

Localiser un échec de la Jacobienne FEMU directe par histoires constitutives
ombres sur le niveau L3. Le forward FEMU L3 est évalué séparément ; aucune
conclusion de convergence des sensibilités ne sera tirée avant ce diagnostic.

## Calcul autorisé

- chemin L3 déjà construit et strictement qualifié pour le forward ;
- aucune modification de SRIX, des tolérances MFront, du solveur global ou de
  l'opérateur d'observation ;
- phases distinguées : `fixed_current_strain` puis `history_advance` ;
- paramètres : `tau0`, `R`, `Q`, `b` en coordonnées logarithmiques ;
- pas initial : `h = 3e-3` ;
- tests diagnostiques supplémentaires : `h = 1.5e-3` et `h = 1e-3` ;
- aucun calcul P43, aucune identification et aucun niveau L4.

## Provenance à enregistrer

Pour chaque appel shadow : incrément accepté, fractions début/fin, incrément
de temps, paramètre, signe, phase et statut. En cas d'échec, arrêter au
premier appel fautif et conserver le message MFront ainsi que le chemin et les
états de base disponibles.

## Critère de décision

Le résultat principal est l'identification non ambiguë du premier appel fautif.
Un changement de `h` est uniquement diagnostique. Il ne pourra être adopté que
si la Jacobienne L2 reste stable dans une étude séparée ; sinon l'échec sera
classé comme limitation du fournisseur shadow par différences finies.
