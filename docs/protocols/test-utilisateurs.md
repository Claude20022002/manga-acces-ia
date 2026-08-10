# Protocole de test utilisateurs — Manga Access AI

Ce protocole encadre les sessions de test avec des utilisateurs aveugles
ou malvoyants, prévues en Phase 5+ pour valider l'accessibilité réelle
du pipeline (audio + timeline + lecteur), pas seulement sa conformité
technique WCAG. Il ne couvre que la préparation et le déroulement des
sessions — l'exécution effective et l'analyse des résultats sont hors
périmètre de ce document (cf. `docs/phases/phase-5.md`, section 5.4).

## 1. Objectif du protocole

Valider, avec de vrais utilisateurs, les hypothèses suivantes :
- L'audio généré (dialogue + sfx + description de scène) permet de
  comprendre le contenu d'une planche de manga sans support visuel.
- Le lecteur (`player.html`) est navigable entièrement au clavier, avec
  un lecteur d'écran, sans assistance sighted.
- La timeline JSON (segment courant affiché en `aria-live`) aide
  effectivement à se repérer dans l'audio, plutôt qu'elle ne gêne.
- La distinction dialogue/sfx (heuristique de la tâche 5.1) est
  perceptible et utile à l'oreille, pas seulement correcte sur le papier.

## 2. Prérequis bloquant avant tout recrutement

Avant de solliciter le moindre participant, vérifier en interne la
compatibilité de `player.html` avec au moins deux lecteurs d'écran de
référence parmi NVDA, JAWS, VoiceOver, Orca (clavier complet, chaque
`aria-live` correctement annoncé, aucun piège au clavier). Un bug
d'accessibilité détectable en interne ne doit jamais être découvert
par un participant — ça reporte le coût du test sur la personne qu'on
est censé servir. Toute régression trouvée ici bloque le recrutement
jusqu'à correction.

## 3. Recrutement

- Passer par des associations plutôt que du recrutement grand public :
  par exemple Valentin Haüy, CFPSAA, ou équivalents locaux selon la
  zone géographique des participants.
- Viser une diversité de lecteurs d'écran et de plateformes (NVDA sur
  Windows, VoiceOver sur macOS/iOS, Orca sur Linux) — le comportement
  de `player.html` peut varier d'un lecteur d'écran à l'autre.
- Nombre de participants : pas de cible chiffrée imposée par ce
  protocole (budget $0, recrutement non rémunéré probable — cf. §9) ;
  privilégier la diversité des profils à la taille de l'échantillon.
- La question d'un éventuel dédommagement des participants est à
  trancher séparément, hors budget $0 du pipeline logiciel lui-même.

## 4. Consentement et éthique

- Le formulaire de consentement doit être lui-même accessible : pas de
  PDF scanné ni d'image de texte. HTML accessible, document texte
  balisé, ou consentement oral enregistré avec accord explicite du
  participant.
- Droit de retrait à tout moment, sans justification à donner, sans
  conséquence.
- Anonymisation des données collectées ; pas d'obligation de préciser
  la nature ou le degré de son handicap au-delà de ce que la personne
  choisit spontanément de partager.
- Enregistrement (audio de l'entretien, logs d'interaction) uniquement
  avec accord explicite et séparé du consentement général de participer.

## 5. Environnement de test

- Privilégier le matériel et les logiciels d'assistance propres au
  participant plutôt qu'un poste de laboratoire imposé — la
  configuration personnelle (raccourcis, verbosité du lecteur d'écran,
  navigateur) fait partie de ce qu'on teste réellement (validité
  écologique).
- Le test à distance (partage d'écran ou d'audio, appel) est
  envisageable et réduit la charge de déplacement pour le participant ;
  à privilégier sauf préférence contraire du participant.

## 6. Scénarios de tâches

Chaque scénario s'appuie sur `player.html` et un audio réellement généré
par le pipeline (jamais de mock ou de contenu simulé) :

1. **Chargement** : charger un fichier `.opus` sans assistance visuelle
   de l'observateur (mesure : réussite/échec, temps, nombre de tentatives).
2. **Lecture/pause au clavier** (barre d'espace).
3. **Navigation** : avancer/reculer de 5s et 10s, changer la vitesse de
   lecture.
4. **Timeline** (si disponible) : charger le fichier `.timeline.json`,
   identifier à l'oreille la transition entre un effet sonore (sfx) et
   un dialogue, en s'appuyant sur ce qu'annonce `#segment-display`.
5. **Compréhension du contenu** : résumer oralement ce qui se passe
   dans la planche après écoute complète — mesure de compréhension du
   contenu manga, pas seulement de l'ergonomie de l'interface.

## 7. Ce qu'on mesure

Uniquement via des moyens eux-mêmes accessibles — jamais de
questionnaire visuel (pas de Likert papier, pas de formulaire non
balisé) :

- Taux de réussite par tâche, temps pris, nombre d'erreurs de navigation.
- Charge perçue et retours qualitatifs recueillis **oralement**, via un
  entretien semi-directif enregistré (avec consentement séparé, §4).
- Exactitude du résumé de contenu (tâche 5 du §6) comparée au contenu
  réel de la planche testée.

## 8. Restitution des résultats

Compte-rendu texte accessible, avec alternative textuelle systématique
à tout tableau ou graphique produit — cohérent avec la philosophie
« accessibilité d'abord » du projet, appliquée jusqu'au compte-rendu du
test lui-même, pas seulement au produit testé.

## 9. Limites du protocole

- Échantillon probablement restreint : budget $0, pas de recrutement
  payant à grande échelle.
- Le retour qualitatif est prioritaire sur toute prétention à la
  significativité statistique — ce protocole produit des indices
  d'utilisabilité, pas une étude contrôlée.
