"""Traduction des dialogues/pensées japonais vers narration_lang (Phase 3 du roadmap original)."""

from __future__ import annotations

from loguru import logger

from manga_access.backends.base import TranslationBackend
from manga_access.pipeline.japanese_text import JAPANESE_CHAR_PATTERN
from manga_access.schemas.narrative_script import NarrativeScript

_TRANSLATABLE_KINDS = ("dialogue", "thought")


def translate_dialogues(
    script: NarrativeScript, backend: TranslationBackend, target_lang: str
) -> NarrativeScript:
    """Traduit en place les segments dialogue/thought japonais de `script` vers `target_lang`.

    Même convention que `enrich_script()` (narration_builder.py) : mute
    `script.segments` en place, retourne `script`. Ne touche que
    `kind in ("dialogue", "thought")` dont le texte contient au moins un
    caractère japonais (`JAPANESE_CHAR_PATTERN`) — sfx (onomatopée, pas
    vraiment "traduisible" au sens utile), narration (déjà générée dans la
    langue cible par narration_builder.py) et scene_description (toujours
    en français, indépendant de `target_lang`) ne sont jamais modifiés.

    Limite connue (cf. QwenVLBackend.translate() et
    docs/sessions/2026-08-11-qwen-translate-smoketest.md) : la traduction
    par LLM peut produire des contresens occasionnels sur les tournures
    japonaises familières/elliptiques typiques des bulles de manga (mesuré :
    1 cas sur 10 phrases réelles du corpus lors du smoketest, sur les deux
    langues cibles fr/en — négation + particule de citation mal
    interprétée, sens inversé). C'est un problème ouvert de la traduction
    par LLM sur ce registre, pas quelque chose que ce module peut détecter
    ou corriger automatiquement. Acceptable pour un MVP : le contenu reste
    globalement fidèle, l'erreur est l'exception, pas la norme.

    `backend.translate()` ne lève jamais d'exception (contrat de
    `TranslationBackend`, cf. base.py) : sur échec interne il retourne déjà
    le texte japonais d'origine inchangé. Un warning est loggé si le
    résultat contient encore du japonais après l'appel (traduction non
    aboutie) — le segment est alors resynthétisé tel quel en amont
    (`_detect_lang` dans audio_assembler.py redétecte correctement "ja"
    grâce au japonais resté en place), pas de segment cassé.
    """
    for segment in script.segments:
        if segment.kind not in _TRANSLATABLE_KINDS:
            continue
        if not JAPANESE_CHAR_PATTERN.search(segment.text):
            continue

        translated = backend.translate(segment.text, target_lang)
        if JAPANESE_CHAR_PATTERN.search(translated):
            logger.warning(f"Traduction indisponible, texte japonais conservé : {segment.id!r}")
        segment.text = translated

    return script
