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

## Tests

```bash
uv run pytest
```
