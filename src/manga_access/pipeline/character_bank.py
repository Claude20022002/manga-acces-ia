"""Chargement d'une character_bank (portraits de référence) pour l'identification nominative Magiv2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_character_bank(json_path: str | Path) -> dict[str, Any]:
    """Charge une character_bank depuis `json_path` au format `{"images": np.ndarray[], "names": str[]}`.

    Le fichier JSON sur disque contient des chemins d'images (clé
    "image_paths", résolus relativement au dossier du JSON s'ils sont
    relatifs) et une liste de noms parallèle (clé "names") — jamais de
    pixels sérialisés. Retourne le dict attendu par
    `StructureBackend.detect_chapter` / `ChapterProcessor.process`, avec les
    images chargées en `np.ndarray` RGB.
    """
    json_path = Path(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    names: list[str] = data["names"]
    image_paths: list[str] = data["image_paths"]
    if len(names) != len(image_paths):
        raise ValueError(
            f"character_bank invalide ({json_path}) : "
            f"{len(names)} nom(s) pour {len(image_paths)} image(s)"
        )

    base_dir = json_path.parent
    images = []
    for raw_path in image_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = base_dir / path
        images.append(np.array(Image.open(path).convert("RGB")))

    return {"images": images, "names": names}
