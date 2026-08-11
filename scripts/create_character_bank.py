#!/usr/bin/env python3
"""Construit une character_bank (portraits de référence nommés) pour Magiv2.

Prend une suite de paires <image> <nom> en arguments positionnels, valide
que chaque image existe et s'ouvre, puis écrit un fichier au format attendu par
`manga_access.pipeline.character_bank.load_character_bank`
(consommé par `ChapterProcessor.process(..., character_bank=...)`).

Usage:
    python scripts/create_character_bank.py \\
        --output data/character_banks/naruto.json \\
        naruto.png "Naruto" sasuke.png "Sasuke" sakura.png "Sakura"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


def build_character_bank(entries: list[tuple[str, str]], output_path: Path) -> None:
    """Valide chaque paire (chemin_image, nom) de `entries` et écrit `output_path`.

    Les chemins sont résolus en absolu avant écriture : la character_bank
    reste valide quel que soit le dossier depuis lequel elle est chargée
    ensuite (load_character_bank ne résout les chemins relatifs que par
    rapport à son propre dossier). Lève `ValueError` si une image référencée
    est introuvable/illisible, ou si `entries` est vide — on refuse d'écrire
    une character_bank partiellement invalide plutôt que de laisser échouer
    silencieusement le matching à l'inférence.
    """
    if not entries:
        raise ValueError("aucune paire <image> <nom> fournie")

    names: list[str] = []
    image_paths: list[str] = []
    for image_arg, name in entries:
        image_path = Path(image_arg)
        if not image_path.is_file():
            raise ValueError(f"image introuvable pour {name!r} : {image_path}")
        with Image.open(image_path) as img:
            img.verify()
        names.append(name)
        image_paths.append(str(image_path.resolve()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"names": names, "image_paths": image_paths}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pairs",
        nargs="+",
        metavar="IMAGE NOM",
        help="Paires alternées <image> <nom> (ex: naruto.png Naruto sasuke.png Sasuke)",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Fichier character_bank de sortie (JSON)"
    )
    args = parser.parse_args()

    if len(args.pairs) % 2 != 0:
        parser.error("nombre d'arguments positionnels impair : attendu des paires <image> <nom>")
    entries = list(zip(args.pairs[0::2], args.pairs[1::2]))

    try:
        build_character_bank(entries, args.output)
    except (ValueError, OSError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"character_bank écrite : {args.output}")


if __name__ == "__main__":
    main()
