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

_SILENCE_BETWEEN_SEGMENTS_MS = 300
_JAPANESE_CHAR_PATTERN = re.compile("[\\u3040-\\u309f\\u30a0-\\u30ff\\u4e00-\\u9fff]")


def _detect_lang(text: str, kind: str) -> str:
    """Détecte la langue de synthèse à passer à `TTSBackend.synthesize()`.

    Priorité : présence de caractères japonais (hiragana/katakana/kanji) ->
    "ja" ; sinon texte de `kind == "scene_description"` (généré par
    `describe_panel()`, toujours en français) -> "fr" ; sinon -> "en-us"
    (défaut de `TTSBackend.synthesize`).
    """
    if _JAPANESE_CHAR_PATTERN.search(text):
        return "ja"
    if kind == "scene_description":
        return "fr"
    return "en-us"


def assemble_audio(script: NarrativeScript, tts_backend: TTSBackend, output_path: Path) -> None:
    """Synthétise et assemble tous les segments de `script` en un fichier .opus.

    Charge `tts_backend`, synthétise chaque segment dans l'ordre (300ms de
    silence entre segments consécutifs), exporte le résultat concaténé vers
    `output_path` au format Opus (FFmpeg via pydub), puis décharge le backend.
    """
    start = time.perf_counter()
    tts_backend.load()

    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=_SILENCE_BETWEEN_SEGMENTS_MS)
    synthesized_count = 0

    for index, segment in enumerate(script.segments):
        text_stripped = segment.text.strip()
        if not text_stripped:
            logger.warning(f"Segment ignoré (texte vide) : {segment.id!r}")
            continue
        if segment.kind == "scene_description":
            continue
        lang = _detect_lang(text_stripped, segment.kind)
        audio_bytes = tts_backend.synthesize(text_stripped, segment.voice_id, lang=lang)
        audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        synthesized_count += 1

        if index > 0:
            combined += silence
        combined += audio

    tts_backend.unload()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_path, format="opus")

    elapsed = time.perf_counter() - start
    logger.info(
        f"{synthesized_count} segment(s) assemblé(s) en {elapsed:.2f}s -> {output_path}"
    )


def save_transcript(script: NarrativeScript, output_path: Path) -> None:
    """Sauvegarde le transcript textuel du script narratif dans un fichier .txt."""
    lines = []
    for segment in script.segments:
        if segment.kind == "scene_description":
            continue
        prefix = f"[{segment.kind.upper()}]"
        lines.append(f"{prefix} {segment.text}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
