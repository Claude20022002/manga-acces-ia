#!/usr/bin/env python3
"""Construit une character_bank (portraits de référence nommés) pour Magiv2.

Prend un mapping JSON {nom_personnage: chemin_image} et un dossier
d'images de référence, valide que chaque image existe et s'ouvre, puis
écrit un fichier au format attendu par
`manga_access.pipeline.character_bank.load_character_bank`
(consommé par `ChapterProcessor.process(..., character_bank=...)`).

Usage:
    python scripts/create_character_bank.py \\
        --images-dir data/character_refs/naruto \\
        --mapping data/character_refs/naruto/mapping.json \\
        --output data/character_banks/naruto.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


def build_character_bank(images_dir: Path, mapping_path: Path, output_path: Path) -> None:
    """Valide `mapping_path` (nom -> image relative à `images_dir`) et écrit `output_path`.

    Lève `ValueError` si une image référencée est introuvable ou illisible —
    on refuse d'écrire une character_bank partiellement invalide plutôt que
    de laisser échouer silencieusement le matching à l'inférence.
    """
    mapping: dict[str, str] = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not mapping:
        raise ValueError(f"mapping vide : {mapping_path}")

    names: list[str] = []
    image_paths: list[str] = []
    for name, relative_path in mapping.items():
        image_path = images_dir / relative_path
        if not image_path.is_file():
            raise ValueError(f"image introuvable pour {name!r} : {image_path}")
        with Image.open(image_path) as img:
            img.verify()
        names.append(name)
        image_paths.append(str(image_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"names": names, "image_paths": image_paths}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir", type=Path, required=True, help="Dossier des images de référence"
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        required=True,
        help="Fichier JSON {nom_personnage: chemin_image_relatif}",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Fichier character_bank de sortie (JSON)"
    )
    args = parser.parse_args()

    try:
        build_character_bank(args.images_dir, args.mapping, args.output)
    except (ValueError, OSError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"character_bank écrite : {args.output}")


if __name__ == "__main__":
    main()
