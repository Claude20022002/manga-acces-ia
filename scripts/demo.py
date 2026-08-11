#!/usr/bin/env python3
"""Démonstration bout-en-bout : dossier d'images manga -> audio navigable.

Enchaîne le pipeline production (ChapterProcessor -> NarrativeScript par
page, fusionnés -> assemble_audio) sur un dossier d'images, avec des logs
colorés par étape. Script de démonstration, pas du code de production
(comme benchmarks/) : produit exactement demo.opus/demo.txt/
demo.timeline.json dans --output-dir.

Usage:
    python scripts/demo.py data/manga_jpg \\
        [--character-bank data/character_banks/mon_manga.json] \\
        [--output-dir data/outputs/demo] \\
        [--narration-lang fr] \\
        [--pages 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

try:
    from manga_access.backends.kokoro_backend import KokoroBackend
    from manga_access.backends.magiv2_backend import Magiv2Backend
    from manga_access.backends.manga_ocr_backend import MangaOCRBackend
except ImportError as exc:
    print(
        f"Erreur : le paquet '{exc.name}' n'est pas installé.\n"
        "Lance 'uv sync' pour installer les dépendances du projet avant "
        "d'exécuter ce script.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

from manga_access.pipeline.audio_assembler import (
    assemble_audio,
    save_timeline,
    save_transcript,
)
from manga_access.pipeline.chapter_processor import ChapterProcessor
from manga_access.pipeline.character_bank import load_character_bank
from manga_access.pipeline.image_source import find_images
from manga_access.pipeline.narrative_builder import build_narrative_script
from manga_access.schemas.narrative_script import NarrativeScript


def merge_narrative_scripts(scripts: list[NarrativeScript], source: dict) -> NarrativeScript:
    """Fusionne les scripts narratifs d'un chapitre en un seul, pour un unique fichier audio.

    Préfixe `id` et `panel_id` par page (`p{index}-...`) : ces identifiants
    sont générés localement à chaque page (ex. "panel-0" existe sur chaque
    page) et entreraient en collision une fois fusionnés — panel_id en
    particulier doit rester unique globalement pour que le regroupement par
    panel de enrich_script() (narration_builder.py) ne mélange pas des
    dialogues de pages différentes qui partageraient le même panel_id.
    """
    merged_segments = []
    for page_index, script in enumerate(scripts):
        for segment in script.segments:
            segment.id = f"p{page_index}-{segment.id}"
            segment.panel_id = f"p{page_index}-{segment.panel_id}"
            merged_segments.append(segment)
    return NarrativeScript(source=source, segments=merged_segments)


def run_demo(
    images_dir: Path,
    output_dir: Path,
    character_bank_path: Path | None,
    narration_lang: str,
    pages: int | None,
) -> None:
    """Exécute le pipeline complet sur `images_dir`, écrit les 3 fichiers de sortie dans `output_dir`."""
    image_paths = find_images(images_dir, limit=pages)
    logger.info(f"{len(image_paths)} page(s) trouvée(s) dans {images_dir}")

    character_bank = None
    if character_bank_path is not None:
        character_bank = load_character_bank(character_bank_path)
        logger.info(f"character_bank chargée : {character_bank['names']}")

    logger.info("🔍 Détection structure...")
    logger.info("📖 Reconnaissance texte...")
    t0 = time.perf_counter()
    processor = ChapterProcessor(Magiv2Backend(), MangaOCRBackend())
    pages_result = processor.process(image_paths, character_bank=character_bank)
    logger.info(
        f"Structure + texte : {len(pages_result)} page(s) traitée(s) en "
        f"{time.perf_counter() - t0:.2f}s"
    )

    scripts = [build_narrative_script(page) for page in pages_result]
    merged_script = merge_narrative_scripts(
        scripts, source={"folder": str(images_dir), "page_count": len(pages_result)}
    )

    logger.info("🎤 Synthèse vocale...")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / "demo.opus"
    t0 = time.perf_counter()
    timeline = assemble_audio(
        merged_script, KokoroBackend(), audio_path, narration_lang=narration_lang
    )
    elapsed = time.perf_counter() - t0

    save_transcript(merged_script, output_dir / "demo.txt")
    save_timeline(timeline, output_dir / "demo.timeline.json")

    logger.success(
        f"✅ Audio généré : {elapsed:.1f}s pour {len(timeline.segments)} segments -> {audio_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images_dir",
        type=Path,
        help="Dossier d'images manga (ordre alphabétique = ordre de lecture)",
    )
    parser.add_argument(
        "--character-bank",
        type=Path,
        default=None,
        help="character_bank JSON optionnelle (identification nominative)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/outputs/demo"),
        help="Dossier de sortie (défaut: data/outputs/demo)",
    )
    parser.add_argument("--narration-lang", default="fr", help="Langue de narration (défaut: fr)")
    parser.add_argument(
        "--pages", type=int, default=None, help="Limite le nombre de pages traitées (défaut: toutes)"
    )
    args = parser.parse_args()

    try:
        run_demo(
            args.images_dir,
            args.output_dir,
            args.character_bank,
            args.narration_lang,
            args.pages,
        )
    except (ValueError, OSError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
