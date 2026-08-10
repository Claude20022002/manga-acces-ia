"""Assemblage audio d'une planche depuis un NarrativeScript, via un backend TTS."""

from __future__ import annotations

import io
import re
import time
from pathlib import Path

from loguru import logger
from pydub import AudioSegment

from manga_access.backends.base import TTSBackend
from manga_access.schemas.narrative_script import NarrativeScript
from manga_access.schemas.timeline import Timeline, TimelineSegment

_SILENCE_BETWEEN_SEGMENTS_MS = 300
_JAPANESE_CHAR_PATTERN = re.compile("[\\u3040-\\u309f\\u30a0-\\u30ff\\u4e00-\\u9fff]")


def _detect_lang(text: str, kind: str) -> str:
    """Détecte la langue de synthèse à passer à `TTSBackend.synthesize()`.

    Priorité : présence de caractères japonais (hiragana/katakana/kanji) ->
    "ja" ; sinon texte de `kind == "scene_description"` (généré par
    `describe_panel()`, toujours en français) -> "fr-fr" ; sinon -> "en-us"
    (défaut de `TTSBackend.synthesize`).
    """
    if _JAPANESE_CHAR_PATTERN.search(text):
        return "ja"
    if kind == "scene_description":
        return "fr-fr"
    return "en-us"


def assemble_audio(script: NarrativeScript, tts_backend: TTSBackend, output_path: Path) -> Timeline:
    """Synthétise et assemble tous les segments de `script` en un fichier .opus.

    Charge `tts_backend`, synthétise chaque segment dans l'ordre (300ms de
    silence entre segments consécutifs), exporte le résultat concaténé vers
    `output_path` au format Opus (FFmpeg via pydub), puis décharge le backend.
    Retourne la Timeline correspondante (`start_ms`/`end_ms` de chaque
    segment audible, alignés sur le fichier .opus exporté) — `start_ms` est
    le point où la parole du segment commence réellement, après insertion
    du silence inter-segments.
    """
    start = time.perf_counter()
    tts_backend.load()

    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=_SILENCE_BETWEEN_SEGMENTS_MS)
    timeline_segments: list[TimelineSegment] = []

    for segment in script.segments:
        text_stripped = segment.text.strip()
        if not text_stripped:
            logger.warning(f"Segment ignoré (texte vide) : {segment.id!r}")
            continue
        lang = _detect_lang(text_stripped, segment.kind)
        audio_bytes = tts_backend.synthesize(text_stripped, segment.voice_id, lang=lang)
        audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))

        if len(combined) > 0:
            combined += silence
        start_ms = len(combined)
        combined += audio
        end_ms = len(combined)

        timeline_segments.append(
            TimelineSegment(
                id=segment.id,
                kind=segment.kind,
                text=text_stripped,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )

    tts_backend.unload()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_path, format="opus")

    elapsed = time.perf_counter() - start
    logger.info(
        f"{len(timeline_segments)} segment(s) assemblé(s) en {elapsed:.2f}s -> {output_path}"
    )

    return Timeline(source=script.source, segments=timeline_segments)


def save_transcript(script: NarrativeScript, output_path: Path) -> None:
    """Sauvegarde le transcript textuel du script narratif dans un fichier .txt."""
    lines = []
    for segment in script.segments:
        prefix = f"[{segment.kind.upper()}]"
        lines.append(f"{prefix} {segment.text}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_timeline(timeline: Timeline, output_path: Path) -> None:
    """Sauvegarde la timeline JSON, sibling du .opus produit par assemble_audio()."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(timeline.to_json(), encoding="utf-8")
