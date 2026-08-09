# Manga Access AI

Pipeline multimodal pour rendre les mangas japonais accessibles aux
personnes aveugles et malvoyantes. Entrée : CBZ/images de manga japonais.
Sortie : audio navigable accompagné d'une timeline JSON, généré entièrement
en local sur CPU, sans service cloud payant.

## Install

Prérequis : [uv](https://docs.astral.sh/uv/) et Python >= 3.11.

```bash
uv sync
```

## Prérequis système

Deux dépendances système, hors gestion `uv`/`pyproject.toml` :

- **FFmpeg** — requis par `pydub` pour décoder/encoder l'audio (assemblage
  des segments TTS, export `.opus`) :
  ```bash
  sudo apt install ffmpeg
  ```
- **espeak-ng** — requis par `kokoro-onnx` pour la phonémisation :
  ```bash
  sudo apt install espeak-ng
  ```

## Poids des modèles

Les poids Magiv2 et manga-ocr se téléchargent automatiquement depuis
HuggingFace au premier chargement (`from_pretrained`), aucune action
manuelle requise.

Kokoro (`kokoro-onnx`) ne télécharge rien automatiquement — les poids
doivent être récupérés manuellement depuis les releases GitHub du projet :

```bash
mkdir -p models/kokoro
wget -P models/kokoro https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx
wget -P models/kokoro https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Variante int8 (~88 Mo) choisie pour la contrainte CPU/RAM du projet,
cohérente avec la quantisation retenue ailleurs dans la stack. `models/`
est ignoré par git (`.gitignore`) — téléchargement local à refaire à
chaque nouvel environnement.

## Dépendances et reproductibilité

### Dépendances cachées de Magiv2

Magiv2 utilise `trust_remote_code=True` et charge en interne
plusieurs modèles (ConditionalDETR, TrOCR, ViT-MAE) dont les
dépendances ne sont pas déclarées dans le README officiel.

Ces dépendances ont été découvertes uniquement à l'exécution
et sont documentées ici pour la reproductibilité :

| Paquet | Raison |
|---|---|
| einops | Manipulation de tenseurs (modelling_magiv2.py) |
| pulp | Optimisation linéaire (assignation personnages) |
| scipy | Calculs scientifiques internes |
| matplotlib | Visualisation (importé même si non utilisé) |
| shapely | Géométrie des bboxes |
| networkx | Graphe d'association texte/personnage (modelling_magiv2.py) |
| sentencepiece | Tokenizer TrOCR (chargé même si OCR inutilisé) |
| timm | Backbone ResNet-50 de ConditionalDETR |

Ces paquets sont déclarés dans `pyproject.toml` et installés
automatiquement via `uv sync`. Aucune action manuelle requise.

### Contrainte de version transformers

`transformers>=4.45.0,<5.0` — le plafond `<5.0` est requis
car transformers 5.x a cassé la conversion automatique
lent→rapide du tokenizer TrOCR utilisé par Magiv2.
Testé et validé avec transformers 4.57.6.

### Note sur la RAM

Magiv2 requiert ~4-6 Go de RAM pendant l'inférence.
Sur une machine avec 10-12 Go de RAM, les modèles
doivent être chargés séquentiellement :
charger Magiv2 → inférence → décharger → charger manga-ocr.
Le pipeline respecte cette contrainte par construction.

## Tests

```bash
uv run pytest
```
