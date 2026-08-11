"""Enrichissement du NarrativeScript avec des préfixes narratifs contextuels."""

from __future__ import annotations

import zlib
from typing import Literal

from manga_access.pipeline.narrative_builder import VOICE_UNKNOWN
from manga_access.schemas.narrative_script import NarrativeScript, NarrativeSegment

_SUPPORTED_LANGS = ("fr", "ja")

_SFX_PREFIX_VARIATIONS_FR = (
    "On entend : {texte}",
    "Soudain, on entend : {texte}",
    "Un bruit retentit : {texte}",
    "Dans le silence, on perçoit : {texte}",
)


def _infer_gender(voice_id: str) -> Literal["female", "male", "unknown"]:
    """Déduit le genre d'une voix Kokoro depuis son id ("{lang}{genre}_{nom}").

    Convention Kokoro (jf_alpha, jm_kumo, af_bella, am_adam...) : 2e caractère
    'f'/'m'. Retourne "unknown" si l'id ne suit pas ce format (fallback sûr,
    pas d'exception).
    """
    if len(voice_id) >= 2 and voice_id[1] in ("f", "m"):
        return "female" if voice_id[1] == "f" else "male"
    return "unknown"


def _dialogue_text(segment: NarrativeSegment, lang: str) -> str:
    """Préfixe/suffixe un dialogue selon le genre du speaker (ou VOICE_UNKNOWN)."""
    if segment.voice_id == VOICE_UNKNOWN:
        if lang == "fr":
            return f"Une voix dit : {segment.text}"
        return f"声が言った：{segment.text}"  # ja

    gender = _infer_gender(segment.voice_id)
    if lang == "fr":
        prefix = {"female": "Elle dit", "male": "Il dit", "unknown": "Le personnage dit"}[gender]
        return f"{prefix} : {segment.text}"
    return f"{segment.text}、と言った"  # ja, pas de distinction de genre demandée


def _sfx_text(segment: NarrativeSegment, lang: str) -> str:
    """Préfixe un SFX ; variation choisie déterministiquement par crc32(id) % n."""
    if lang != "fr":
        return segment.text  # pas de gabarit ja spécifié
    index = zlib.crc32(segment.id.encode()) % len(_SFX_PREFIX_VARIATIONS_FR)
    return _SFX_PREFIX_VARIATIONS_FR[index].format(texte=segment.text)


def enrich_script(script: NarrativeScript, lang: str = "fr") -> NarrativeScript:
    """Enrichit en place le texte de chaque segment avec un préfixe narratif contextuel.

    `lang` sélectionne le gabarit de préfixe ("fr" ou "ja") — il ne détecte ni
    ne traduit rien : l'appelant garantit que `lang` correspond à la langue
    réelle du contenu des segments concernés (dialogue/sfx). narration et
    scene_description ne sont jamais modifiés (déjà narratifs ; scene_description
    est toujours généré en français par Qwen3-VL, quel que soit `lang`).
    thought est laissé inchangé (hors périmètre).

    Retourne `script` (même instance, muté en place) — à passer directement
    à `assemble_audio()`.
    """
    if lang not in _SUPPORTED_LANGS:
        raise ValueError(f"lang non supporté : {lang!r} (attendu : {_SUPPORTED_LANGS})")

    for segment in script.segments:
        if segment.kind == "dialogue":
            segment.text = _dialogue_text(segment, lang)
        elif segment.kind == "sfx":
            segment.text = _sfx_text(segment, lang)
        # narration, scene_description, thought : inchangés

    return script
