"""Assemblage audio d'une planche depuis un NarrativeScript, via un backend TTS."""

from __future__ import annotations

import io
import re
import time
import unicodedata
from pathlib import Path

from loguru import logger
from pydub import AudioSegment

from manga_access.backends.base import TTSBackend
from manga_access.pipeline.narration_builder import enrich_script
from manga_access.schemas.narrative_script import NarrativeScript
from manga_access.schemas.timeline import Timeline, TimelineSegment

_SILENCE_BETWEEN_SEGMENTS_MS = 300
_JAPANESE_CHAR_PATTERN = re.compile("[\\u3040-\\u309f\\u30a0-\\u30ff\\u4e00-\\u9fff]")


def _detect_lang(text: str, kind: str) -> str:
    """Détecte la langue de synthèse à passer à `TTSBackend.synthesize()`.

    Priorité : présence de caractères japonais (hiragana/katakana/kanji) ->
    "ja" ; sinon texte d'un `kind` narratif généré en français ->
    "fr-fr" (`scene_description` par `describe_panel()`/Qwen3-VL,
    `narration` par les segments préfixe/suffixe insérés par
    `enrich_script()` — le check japonais ci-dessus a déjà exclu tout texte
    japonais à ce stade, donc pas besoin de le revérifier) ; sinon -> "en-us"
    (défaut de `TTSBackend.synthesize`).
    """
    if _JAPANESE_CHAR_PATTERN.search(text):
        return "ja"
    if kind in ("scene_description", "narration"):
        return "fr-fr"
    return "en-us"


def _voice_for_lang(voice_id: str, lang: str) -> str:
    """Sélectionne une voix Kokoro compatible avec la langue.

    voice_id est déjà une voix japonaise valide pour les segments ja —
    narrative_builder.py assigne désormais narrateur/inconnu/personnages
    sur le pool japonais en amont, pas de substitution ici. Le français
    (descriptions de scène) reste forcé sur une voix dédiée faute de pool
    de voix par personnage en fr ; l'anglais (fallback par défaut) garde
    voice_id tel quel.
    """
    if lang == "ja":
        return voice_id
    if lang == "fr-fr":
        return "ff_siwis"
    return voice_id


def _clean_japanese_text(text: str) -> str:
    """Nettoie le texte japonais OCR avant synthèse TTS.

    Normalise les points de suspension pleine chasse ('．．．' -> '、') et
    réduit les répétitions de ponctuation ('！！' -> '！', '？？' -> '？')
    avant passage au G2P de Kokoro (misaki) — cas réels observés dans
    data/outputs/benchmark_v6/transcripts/2-1.txt. Normalise d'abord en
    NFKC les lettres/chiffres pleine chasse (fullwidth latin, ex.
    "Ｗｏｒｄｏｗｓ") — sans ça, Kokoro les épelle lettre par lettre au lieu de
    lire le mot, et le texte inutilement long déclenche la troncature à 510
    phonèmes. Caractère par caractère (`ch.isalnum()`), pas sur toute la
    chaîne : ponctuation pleine chasse ("．", "！", "？") a aussi une forme
    NFKC (ASCII), ce qui casserait le nettoyage ci-dessous si on
    normalisait tout d'un coup (il cherche spécifiquement ces caractères
    pleine chasse).
    """
    text = "".join(
        unicodedata.normalize("NFKC", ch) if ch.isalnum() else ch for ch in text
    )
    text = text.replace("．", "。")
    text = re.sub(r"[。、]{2,}", "、", text)
    text = re.sub(r"！{2,}", "！", text)
    text = re.sub(r"？{2,}", "？", text)
    return text.strip()


def assemble_audio(
    script: NarrativeScript,
    tts_backend: TTSBackend,
    output_path: Path,
    include_scene_descriptions: bool = False,
    narration_lang: str | None = None,
) -> Timeline:
    """Synthétise et assemble tous les segments de `script` en un fichier .opus.

    Charge `tts_backend`, synthétise chaque segment dans l'ordre (300ms de
    silence entre segments consécutifs), exporte le résultat concaténé vers
    `output_path` au format Opus (FFmpeg via pydub), puis décharge le backend.
    Retourne la Timeline correspondante (`start_ms`/`end_ms` de chaque
    segment audible, alignés sur le fichier .opus exporté) — `start_ms` est
    le point où la parole du segment commence réellement, après insertion
    du silence inter-segments.

    Si `narration_lang` est fourni (ex. "fr"), `script` est enrichi via
    `enrich_script()` (narration_builder.py) avant toute synthèse : le texte
    enrichi (préfixes narratifs contextuels) est ce qui est synthétisé et ce
    qui apparaît dans la Timeline retournée. `enrich_script()` mute `script`
    en place, donc l'appelant voit aussi le texte enrichi sur son objet
    `script` d'origine après cet appel (ex. pour un `save_transcript()`
    ultérieur sur le même script). `include_scene_descriptions` (défaut
    False) contrôle si les segments `scene_description` sont synthétisés :
    par défaut ils sont ignorés comme un segment à texte vide (ni audio, ni
    entrée dans la Timeline retournée) — descriptions par règles jugées trop
    répétitives à l'écoute ("Deux personnages détectés..."). Ils restent
    cependant dans `script.segments` (jamais retirés, seulement sautés dans
    la boucle de synthèse), donc un `save_transcript()` fait sur ce même
    script les conserve. À True, comportement historique restauré (toujours
    synthétisés).
    """
    if narration_lang is not None:
        script = enrich_script(script, lang=narration_lang)

    start = time.perf_counter()
    tts_backend.load()

    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=_SILENCE_BETWEEN_SEGMENTS_MS)
    timeline_segments: list[TimelineSegment] = []

    for segment in script.segments:
        if segment.kind == "scene_description" and not include_scene_descriptions:
            continue
        text_stripped = segment.text.strip()
        if not text_stripped:
            logger.warning(f"Segment ignoré (texte vide) : {segment.id!r}")
            continue
        lang = _detect_lang(text_stripped, segment.kind)
        voice = _voice_for_lang(segment.voice_id, lang)
        if lang == "ja":
            text_to_synth = _clean_japanese_text(text_stripped)
            if not _JAPANESE_CHAR_PATTERN.search(text_to_synth):
                logger.warning(
                    "Segment ignoré (ja détecté mais aucun caractère japonais après "
                    f"nettoyage) : {segment.id!r} texte nettoyé={text_to_synth!r}"
                )
                continue
        else:
            text_to_synth = text_stripped
        audio_bytes = tts_backend.synthesize(text_to_synth, voice, lang=lang)
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
                page_index=segment.page_index,
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
