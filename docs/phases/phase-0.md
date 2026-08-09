# Phase 0 — Socle

## Objectif
Environnement reproductible + harnais de test + baseline naïve.
À la fin de cette phase, on doit pouvoir lancer une page manga
à travers manga-ocr et Kokoro et obtenir un fichier audio,
même sans structure ni attribution locuteur.

## Tâches

### 0.1 pyproject.toml
- Nom : manga-access-ai
- Version : 0.1.0
- Python : >=3.11
- Dépendances runtime :
    manga-ocr
    kokoro-onnx
    pydub
    fastapi
    uvicorn
    pydantic>=2.0
    Pillow
    loguru
- Dépendances dev :
    pytest
    pytest-cov
    ruff
    mypy
- Config pytest : testpaths = ["tests"]
- Config ruff : line-length = 88, target = py311

### 0.2 Schéma MangaPage JSON v1
Fichier : src/manga_access/schemas/manga_page.py
Modèles Pydantic à créer :
  - TextElement (id, type: Literal["dialogue","narration","sfx","thought"],
    bbox, text_original, speaker_id optionnel, confidence float)
  - Panel (id, order, bbox, elements: list[TextElement],
    scene_description optionnel)
  - Character (id, voice_id, name optionnel, cluster_confidence float)
  - MangaPage (schema_version, source dict, reading_direction,
    characters: list[Character], panels: list[Panel])
Tout champ incertain = Optional avec None par défaut.
Ajouter une méthode to_json() et un classmethod from_json() sur MangaPage.

### 0.3 Interfaces abstraites des backends
Fichier : src/manga_access/backends/base.py
  - OCRBackend (ABC) : méthode recognize(image: PIL.Image, bbox: tuple) -> TextElement
  - TTSBackend (ABC) : méthode synthesize(text: str, voice_id: str) -> bytes
Les implémentations concrètes viennent en Phase 1.

### 0.4 Harnais de test
tests/test_schemas.py :
  - test_manga_page_roundtrip : crée un MangaPage, sérialise, désérialise, compare
  - test_text_element_types : vérifie que les Literal["dialogue"...] rejettent les types invalides
  - test_optional_fields : vérifie que les champs Optional acceptent None

tests/test_backends.py :
  - test_ocr_backend_is_abstract : vérifie qu'OCRBackend ne peut pas être instancié
  - test_tts_backend_is_abstract : idem pour TTSBackend

### 0.5 Baseline naïve (script autonome, pas dans src/)
Fichier : benchmarks/baseline_naive.py
Script qui prend un dossier d'images manga en argument,
lance manga-ocr sur chaque image (sans Magiv2, sans ordre, sans attribution),
produit un fichier texte brut et mesure le temps par page.
Ce n'est PAS du code de production. C'est notre point de comparaison.

### 0.6 docs/sessions et README minimal
README.md : titre, description 3 lignes, section "Install" avec les commandes uv.
.env.example : fichier vide commenté (aucun secret pour l'instant,
mais structure en place).

## Critères de succès
- uv sync sans erreur
- pytest passe (tous les tests de 0.4)
- ruff check src/ tests/ sans erreur
- python benchmarks/baseline_naive.py fonctionne sur 1 image test

## Ce qui n'est PAS dans cette phase
- Magiv2 (Phase 1)
- Qwen3-VL (Phase 2)
- API FastAPI (Phase 2)
- Audio timeline (Phase 2)
- Traduction (Phase 3)
