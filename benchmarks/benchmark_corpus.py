#!/usr/bin/env python3
"""Benchmark corpus complet : pipeline bout-en-bout sur toutes les pages d'un dossier.

Script de benchmark, PAS du code de production. Traite le corpus en 3 passes
séquentielles (structuration -> OCR -> synthèse audio) : chaque backend est
chargé une seule fois pour tout le corpus puis déchargé avant la passe
suivante, conformément à la contrainte mémoire du projet (12 Go RAM, un seul
modèle lourd en RAM à la fois) et pour éviter 28x le coût de rechargement.
Une page en erreur n'interrompt pas les suivantes.

Usage:
    python benchmarks/benchmark_corpus.py [dossier_images] [--output <fichier.csv>]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image

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

import io

from pydub import AudioSegment

from manga_access.pipeline.audio_assembler import _detect_lang, save_transcript
from manga_access.pipeline.narrative_builder import build_narrative_script
from manga_access.pipeline.scene_descriptor import describe_panel
from manga_access.schemas.manga_page import BBox, Character, MangaPage, Panel
from manga_access.schemas.narrative_script import NarrativeScript

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_SILENCE_BETWEEN_SEGMENTS_MS = 300


@dataclass
class PageResult:
    """Métriques collectées pour une page du corpus."""

    page: str
    panels_count: int = 0
    texts_detected_count: int = 0
    texts_lost_count: int = 0
    segments_count: int = 0
    audio_duration_s: float = 0.0
    processing_time_s: float = 0.0
    status: str = "OK"
    error_message: str = ""


@dataclass
class _PageState:
    """État intermédiaire d'une page, porté entre les 3 passes du benchmark."""

    image_path: Path
    result: PageResult
    panels: list[Panel] = field(default_factory=list)
    text_bboxes: list[BBox] = field(default_factory=list)
    detections: dict[str, Any] = field(default_factory=dict)
    page: MangaPage | None = None
    script: NarrativeScript | None = None
    failed: bool = False


def find_images(folder: Path) -> list[Path]:
    """Retourne les fichiers image du dossier, triés par nom."""
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def _clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
    """Ramène `bbox` dans les bornes [0, width] x [0, height] de l'image source."""
    x1, y1, x2, y2 = bbox
    return (
        max(0.0, x1),
        max(0.0, y1),
        min(float(width), x2),
        min(float(height), y2),
    )


def _find_panel_for_text(panels: list[Panel], text_bbox: BBox) -> Panel | None:
    """Retourne le panel dont la bbox contient le centre de `text_bbox`, sinon None."""
    cx = (text_bbox[0] + text_bbox[2]) / 2
    cy = (text_bbox[1] + text_bbox[3]) / 2
    for panel in panels:
        ox1, oy1, ox2, oy2 = panel.bbox
        if ox1 <= cx <= ox2 and oy1 <= cy <= oy2:
            return panel
    return None


def _build_characters(detections: dict[str, Any]) -> list[Character]:
    """Construit les personnages depuis les clusters bruts retournés par Magiv2."""
    cluster_labels = detections.get("character_cluster_labels", [])
    return [
        Character(
            id=f"char-{cluster_id}",
            voice_id=f"voice_{cluster_id}",
            name=None,
            cluster_confidence=1.0,
        )
        for cluster_id in sorted(set(cluster_labels))
    ]


def _fail(state: _PageState, phase: str, exc: Exception) -> None:
    """Marque une page en erreur et journalise, sans interrompre les autres pages."""
    state.failed = True
    state.result.status = "ERREUR"
    state.result.error_message = f"{phase}: {exc}"
    logger.error(f"{state.image_path.name} : échec {phase} — {exc}")


def _run_structure_phase(states: list[_PageState], structure_backend: Magiv2Backend) -> None:
    """Passe 1 : structuration Magiv2 sur toutes les pages, modèle chargé une seule fois."""
    structure_backend.load()
    try:
        for state in states:
            page_start = time.perf_counter()
            try:
                image = Image.open(state.image_path).convert("RGB")
                width, height = image.size
                detections = structure_backend.detect(image)

                state.detections = detections
                state.panels = [
                    Panel(
                        id=f"panel-{i}",
                        order=i,
                        bbox=_clip_bbox(tuple(float(v) for v in bbox), width, height),
                    )
                    for i, bbox in enumerate(detections.get("panels", []))
                ]
                state.text_bboxes = [
                    _clip_bbox(tuple(float(v) for v in bbox), width, height)
                    for bbox in detections.get("texts", [])
                ]
                state.result.panels_count = len(state.panels)
                state.result.texts_detected_count = len(state.text_bboxes)
                state.result.processing_time_s += time.perf_counter() - page_start
            except Exception as exc:  # noqa: BLE001 - benchmark : une page en erreur ne doit pas arrêter le corpus
                _fail(state, "structuration", exc)
    finally:
        structure_backend.unload()


def _run_ocr_phase(states: list[_PageState], ocr_backend: MangaOCRBackend) -> None:
    """Passe 2 : OCR + description de scène + script narratif, modèle chargé une seule fois."""
    ocr_backend.load()
    try:
        for state in states:
            if state.failed:
                continue
            page_start = time.perf_counter()
            try:
                image = Image.open(state.image_path).convert("RGB")

                n_lost = 0
                for text_bbox in state.text_bboxes:
                    element = ocr_backend.recognize(image, text_bbox)
                    panel = _find_panel_for_text(state.panels, text_bbox)
                    if panel is not None:
                        panel.elements.append(element)
                    else:
                        n_lost += 1
                        cx = (text_bbox[0] + text_bbox[2]) / 2
                        cy = (text_bbox[1] + text_bbox[3]) / 2
                        logger.warning(
                            f"{state.image_path.name} : texte hors panel ignoré : "
                            f"centre=({cx:.1f}, {cy:.1f}) "
                            f"bbox={tuple(round(v, 1) for v in text_bbox)}"
                        )

                characters = _build_characters(state.detections)
                for panel in state.panels:
                    panel.scene_description = describe_panel(
                        panel, n_characters=len(characters)
                    )

                state.page = MangaPage(
                    source={"file": str(state.image_path), "page_index": 0},
                    reading_direction="right_to_left",
                    characters=characters,
                    panels=state.panels,
                )
                state.script = build_narrative_script(state.page)

                state.result.texts_lost_count = n_lost
                state.result.segments_count = len(state.script.segments)
                state.result.processing_time_s += time.perf_counter() - page_start
            except Exception as exc:  # noqa: BLE001 - benchmark : une page en erreur ne doit pas arrêter le corpus
                _fail(state, "OCR", exc)
    finally:
        ocr_backend.unload()


def _assemble_audio_loaded(
    script: NarrativeScript, tts_backend: KokoroBackend, output_path: Path
) -> float:
    """Synthétise `script` via un backend TTS déjà chargé, retourne la durée en secondes.

    Réplique pipeline.audio_assembler.assemble_audio() sans les appels
    load()/unload() : dans ce script, Kokoro est chargé une seule fois pour
    tout le corpus.
    """
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=_SILENCE_BETWEEN_SEGMENTS_MS)

    for index, segment in enumerate(script.segments):
        if segment.kind == "scene_description":
            continue
        text_stripped = segment.text.strip()
        lang = _detect_lang(text_stripped, segment.kind)
        audio_bytes = tts_backend.synthesize(text_stripped, segment.voice_id, lang=lang)
        audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        if index > 0:
            combined += silence
        combined += audio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_path, format="opus")
    return len(combined) / 1000.0


def _run_audio_phase(
    states: list[_PageState], tts_backend: KokoroBackend, audio_dir: Path
) -> None:
    """Passe 3 : synthèse + assemblage audio, modèle chargé une seule fois."""
    tts_backend.load()
    try:
        for state in states:
            if state.failed:
                continue
            page_start = time.perf_counter()
            try:
                assert state.script is not None
                output_path = audio_dir / f"{state.image_path.stem}.opus"
                duration_s = _assemble_audio_loaded(state.script, tts_backend, output_path)

                transcript_path = audio_dir / "transcripts" / f"{state.image_path.stem}.txt"
                save_transcript(state.script, transcript_path)

                state.result.audio_duration_s = duration_s
                state.result.processing_time_s += time.perf_counter() - page_start
            except Exception as exc:  # noqa: BLE001 - benchmark : une page en erreur ne doit pas arrêter le corpus
                _fail(state, "synthèse audio", exc)
    finally:
        tts_backend.unload()


def write_csv(results: list[PageResult], output_path: Path) -> None:
    """Écrit les métriques par page dans un CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "page",
                "panels_count",
                "texts_detected_count",
                "texts_lost_count",
                "segments_count",
                "audio_duration_s",
                "processing_time_s",
                "status",
                "error_message",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "page": result.page,
                    "panels_count": result.panels_count,
                    "texts_detected_count": result.texts_detected_count,
                    "texts_lost_count": result.texts_lost_count,
                    "segments_count": result.segments_count,
                    "audio_duration_s": f"{result.audio_duration_s:.2f}",
                    "processing_time_s": f"{result.processing_time_s:.2f}",
                    "status": result.status,
                    "error_message": result.error_message,
                }
            )


def print_summary(results: list[PageResult], total_wall_time_s: float) -> None:
    """Affiche un résumé console : totaux et moyennes du corpus."""
    n_total = len(results)
    ok_results = [r for r in results if r.status == "OK"]
    error_results = [r for r in results if r.status != "OK"]
    n_ok = len(ok_results)

    print("\n=== Résumé benchmark corpus ===")
    print(f"Pages : {n_total} ({n_ok} OK, {len(error_results)} ERREUR)")

    if error_results:
        print("\nPages en erreur :")
        for result in error_results:
            print(f"  - {result.page} : {result.error_message}")

    if ok_results:
        n = len(ok_results)
        sum_panels = sum(r.panels_count for r in ok_results)
        sum_texts = sum(r.texts_detected_count for r in ok_results)
        sum_lost = sum(r.texts_lost_count for r in ok_results)
        sum_segments = sum(r.segments_count for r in ok_results)
        sum_audio = sum(r.audio_duration_s for r in ok_results)
        sum_processing = sum(r.processing_time_s for r in ok_results)

        print("\nSur les pages OK :")
        print(f"  Panels détectés      : {sum_panels} (moyenne {sum_panels / n:.1f}/page)")
        print(f"  Textes détectés      : {sum_texts} (moyenne {sum_texts / n:.1f}/page)")
        print(f"  Textes hors panel    : {sum_lost} (moyenne {sum_lost / n:.1f}/page)")
        print(f"  Segments narratifs   : {sum_segments} (moyenne {sum_segments / n:.1f}/page)")
        print(f"  Durée audio totale   : {sum_audio:.1f}s (moyenne {sum_audio / n:.1f}s/page)")
        print(
            f"  Temps de traitement  : {sum_processing:.1f}s "
            f"(moyenne {sum_processing / n:.1f}s/page, hors chargement modèles)"
        )

    print(f"\nTemps total du run (chargements modèles inclus) : {total_wall_time_s:.1f}s")


def run_benchmark(folder: Path, output_path: Path, audio_dir: Path) -> None:
    """Exécute le benchmark corpus complet en 3 passes séquentielles."""
    images = find_images(folder)
    if not images:
        print(f"Aucune image trouvée dans {folder}", file=sys.stderr)
        raise SystemExit(1)

    states = [_PageState(image_path=p, result=PageResult(page=p.name)) for p in images]

    total_start = time.perf_counter()

    logger.info(f"Passe 1/3 : structuration Magiv2 sur {len(states)} page(s)")
    _run_structure_phase(states, Magiv2Backend())

    logger.info("Passe 2/3 : OCR manga-ocr + script narratif")
    _run_ocr_phase(states, MangaOCRBackend())

    logger.info("Passe 3/3 : synthèse audio Kokoro")
    _run_audio_phase(states, KokoroBackend(), audio_dir)

    total_elapsed = time.perf_counter() - total_start

    results = [state.result for state in states]
    write_csv(results, output_path)
    print(f"Résultat écrit dans {output_path}")
    print_summary(results, total_elapsed)


def main() -> None:
    """Point d'entrée CLI du benchmark corpus complet."""
    parser = argparse.ArgumentParser(
        description="Benchmark pipeline bout-en-bout sur un corpus complet de pages manga."
    )
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=Path("data/manga_jpg"),
        help="Dossier contenant les images manga (défaut : data/manga_jpg)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/outputs/benchmark_corpus.csv"),
        help="Fichier CSV de sortie (défaut : data/outputs/benchmark_corpus.csv)",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data/outputs/benchmark"),
        help="Dossier des fichiers audio générés (défaut : data/outputs/benchmark)",
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"Erreur : {args.folder} n'est pas un dossier valide.", file=sys.stderr)
        raise SystemExit(1)

    run_benchmark(args.folder, args.output, args.audio_dir)


if __name__ == "__main__":
    main()
