"""Résolution ordonnée des images d'un dossier manga (ordre de lecture)."""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_images(folder: Path, limit: int | None = None) -> list[Path]:
    """Retourne les images de `folder`, triées par nom (ordre de lecture), tronquées à `limit`.

    Tri alphabétique déterministe : c'est ce que `demo.py` (CLI) et
    `api/jobs.py` (page_index -> image servie) doivent utiliser tous les
    deux, à l'identique, pour qu'un `page_index` de la timeline désigne
    toujours la même image des deux côtés — d'où l'extraction dans ce
    module partagé plutôt qu'une définition locale par appelant.
    """
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"aucune image trouvée dans {folder}")
    return images[:limit] if limit is not None else images
