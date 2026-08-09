#!/usr/bin/env python3
"""Baseline naïve : OCR brut sur un dossier d'images, sans structure ni attribution.

Script de benchmark, PAS du code de production. Sert de point de comparaison
pour mesurer l'apport de Magiv2 (structuration) et de l'attribution de
locuteur dans les phases suivantes.

Usage:
    python benchmarks/baseline_naive.py <dossier_images> [--output <fichier.txt>]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from manga_ocr import MangaOcr
except ImportError as exc:
    print(
        "Erreur : le paquet 'manga-ocr' n'est pas installé.\n"
        "Lance 'uv sync' pour installer les dépendances du projet avant "
        "d'exécuter ce script.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_images(folder: Path) -> list[Path]:
    """Retourne les fichiers image du dossier, triés par nom."""
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def run_baseline(folder: Path, output_path: Path) -> None:
    """Lance manga-ocr sur chaque image du dossier et écrit un texte brut.

    Aucune structuration (Magiv2), aucun ordre de lecture, aucune attribution
    de locuteur : chaque image est traitée indépendamment, dans l'ordre
    alphabétique des fichiers.
    """
    images = find_images(folder)
    if not images:
        print(f"Aucune image trouvée dans {folder}", file=sys.stderr)
        raise SystemExit(1)

    mocr = MangaOcr()

    lines: list[str] = []
    total_start = time.perf_counter()

    for image_path in images:
        page_start = time.perf_counter()
        text = mocr(Image.open(image_path))
        elapsed = time.perf_counter() - page_start

        print(f"{image_path.name}: {elapsed:.2f}s")
        lines.append(f"=== {image_path.name} ({elapsed:.2f}s) ===")
        lines.append(text)
        lines.append("")

    total_elapsed = time.perf_counter() - total_start
    print(
        f"\nTotal : {len(images)} page(s) en {total_elapsed:.2f}s "
        f"({total_elapsed / len(images):.2f}s/page en moyenne)"
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Résultat écrit dans {output_path}")


def main() -> None:
    """Point d'entrée CLI du script de baseline."""
    parser = argparse.ArgumentParser(
        description="Baseline naïve : OCR brut sans structure ni attribution."
    )
    parser.add_argument("folder", type=Path, help="Dossier contenant les images manga")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/outputs/baseline_output.txt"),
        help="Fichier texte de sortie (défaut : data/outputs/baseline_output.txt)",
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"Erreur : {args.folder} n'est pas un dossier valide.", file=sys.stderr)
        raise SystemExit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_baseline(args.folder, args.output)


if __name__ == "__main__":
    main()
