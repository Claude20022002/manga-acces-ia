# Manga Access AI — CLAUDE.md

## Vision du projet
Pipeline multimodal pour rendre les mangas japonais accessibles aux personnes
aveugles et malvoyantes. Entrée : CBZ/images. Sortie : audio navigable + timeline JSON.

## Philosophie technique
- Accessibilité d'abord. Jamais de décision qui sacrifie l'accessibilité.
- Budget $0. Pas de cloud payant, pas d'API externes, tout tourne en local.
- Pas d'entraînement. On utilise des modèles pré-entraînés jusqu'à preuve du besoin.
- CPU uniquement. Machine : Ryzen 5 5500U, 12 Go RAM, AMD iGPU inutilisable en ML.
- Les modèles tournent séquentiellement (chargement → traitement → déchargement).

## Stack décidée (ne pas remettre en question sans raison explicite)
- Structure manga  : Magiv2 (ragavsachdeva/magiv2 — licence NC recherche, intentionnel)
- OCR japonais     : manga-ocr (kha-white, Apache-2.0)
- Sens/traduction  : Qwen3-VL-4B-Instruct (Apache-2.0, Q4 quantisé)
- TTS              : Kokoro-82M via kokoro-onnx (Apache-2.0, MIT)
- Audio            : pydub + FFmpeg (LGPL build)
- Backend          : FastAPI + SQLite
- Packaging        : uv (pyproject.toml)
- OS cible         : Ubuntu 24.04

## Contrat inter-modules : MangaPage JSON v1
Le schéma est dans src/manga_access/schemas/manga_page.py (Pydantic).
C'est le seul moyen de communication entre les backends. Ne jamais bypasser.

## Architecture des backends
Chaque backend est une classe abstraite dans backends/.
Implémentation concrète séparée. La logique métier ne doit jamais importer
directement un modèle — toujours passer par l'interface du backend.

## Règles de code
- Python 3.11, type hints partout, pas de Any sans commentaire explicatif
- Chaque fonction publique a une docstring
- Tests dans tests/ avec pytest, un fichier de test par module
- Pas de print() en dehors des scripts de benchmark — utiliser logging
- Pas de secrets dans le code — variables d'environnement via .env (non commité)

## Ce que Claude NE doit PAS faire
- Installer des dépendances lourdes sans demander (torch, transformers seuls = 4+ Go)
- Commiter sur main directement
- Proposer des alternatives à la stack décidée sans justification technique précise
- Écrire du code de production avant que les tests de la phase courante passent

## Phase actuelle : 0 — Socle
Objectif : environnement reproductible + harnais de test + baseline naïve
Voir docs/phases/phase-0.md pour le détail des tâches.