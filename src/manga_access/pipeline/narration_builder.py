"""Enrichissement du NarrativeScript : insertion de segments de narration
contextuels (voix du narrateur) autour de chaque dialogue/sfx concerné.
"""

from __future__ import annotations

import zlib
from typing import Literal

from manga_access.pipeline.narrative_builder import VOICE_NARRATOR, VOICE_UNKNOWN
from manga_access.schemas.narrative_script import NarrativeScript, NarrativeSegment

_SUPPORTED_LANGS = ("fr", "ja")

_NARRATOR_VOICE_BY_LANG = {"fr": "ff_siwis", "ja": VOICE_NARRATOR}

_SFX_PREFIX_VARIATIONS_FR = (
    "On entend :",
    "Soudain, on entend :",
    "Un bruit retentit :",
    "Dans le silence, on perçoit :",
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


def _narrator_segment(text: str, lang: str, source: NarrativeSegment) -> NarrativeSegment:
    """Construit le segment de narration (voix du narrateur) associé à `source`.

    Segment indépendant, jamais fusionné dans le texte de `source` : c'est ce
    qui permet à assemble_audio() de synthétiser le préfixe/suffixe avec la
    voix du narrateur et `source` avec sa propre voix (personnage ou
    narrateur selon son kind d'origine).
    """
    return NarrativeSegment(
        id=f"{source.id}-narrator",
        panel_id=source.panel_id,
        kind="narration",
        voice_id=_NARRATOR_VOICE_BY_LANG[lang],
        text=text,
    )


def _has_following_dialogue_or_sfx_in_panel(
    segments: list[NarrativeSegment], index: int
) -> bool:
    """Vrai si un segment dialogue/sfx suit `segments[index]` dans le même panel.

    Cherche dans `segments` (liste d'origine de `script`, avant insertion des
    segments de narration) à partir de `index + 1` : seuls les segments dont
    `panel_id` correspond à celui de `segments[index]` sont considérés.
    """
    panel_id = segments[index].panel_id
    return any(
        later.panel_id == panel_id and later.kind in ("dialogue", "sfx")
        for later in segments[index + 1 :]
    )


def _dialogue_segments(
    segment: NarrativeSegment,
    lang: str,
    same_speaker_as_previous: bool,
    has_following_dialogue_or_sfx: bool,
) -> list[NarrativeSegment]:
    """Entoure un dialogue (inchangé) d'un segment de narration contextuel.

    `same_speaker_as_previous` (comparaison par voice_id — cf. docstring de
    enrich_script pour la limite connue) et `has_following_dialogue_or_sfx`
    (aucun dialogue/sfx après dans le même panel) suppriment chacun le
    segment de narration : un même personnage qui enchaîne plusieurs
    répliques ne se voit annoncé qu'une fois, et une réplique qui clôt son
    panel (rien de vocal après elle) n'est pas non plus annoncée. fr et ja
    speaker inconnu : narration avant (préfixe). ja speaker connu : narration
    après (suffixe) — "{texte}、と言った" est une proposition rapportée qui
    suit la citation en japonais, gabarit conservé tel quel depuis la
    Phase D plutôt que d'inventer une formulation en préfixe. fr speaker
    connu et nommé (character_bank, Phase 9) : préfixe "{nom} dit :" plutôt
    que le préfixe genré générique.
    """
    if same_speaker_as_previous or not has_following_dialogue_or_sfx:
        return [segment]

    if segment.voice_id == VOICE_UNKNOWN:
        prefix = "Une voix dit :" if lang == "fr" else "声が言った："
        return [_narrator_segment(prefix, lang, segment), segment]

    if lang == "ja":
        suffix = _narrator_segment("、と言った", lang, segment)
        return [segment, suffix]

    if segment.character_name is not None:
        return [_narrator_segment(f"{segment.character_name} dit :", lang, segment), segment]

    gender = _infer_gender(segment.voice_id)
    prefix = {"female": "Elle dit", "male": "Il dit", "unknown": "Le personnage dit"}[gender]
    return [_narrator_segment(f"{prefix} :", lang, segment), segment]


def _sfx_segments(segment: NarrativeSegment, lang: str) -> list[NarrativeSegment]:
    """Précède un SFX (inchangé) d'une narration en fr ; inchangé en ja (pas de gabarit)."""
    if lang != "fr":
        return [segment]
    index = zlib.crc32(segment.id.encode()) % len(_SFX_PREFIX_VARIATIONS_FR)
    prefix = _SFX_PREFIX_VARIATIONS_FR[index]
    return [_narrator_segment(prefix, lang, segment), segment]


def enrich_script(script: NarrativeScript, lang: str = "fr") -> NarrativeScript:
    """Insère des segments de narration contextuels autour des dialogues/sfx.

    `lang` sélectionne le gabarit ("fr" ou "ja") — il ne détecte ni ne
    traduit rien : l'appelant garantit que `lang` correspond à la langue
    réelle du contenu des segments concernés. Les segments dialogue/sfx
    d'origine ne sont jamais modifiés (ni texte, ni voice_id) : seuls de
    nouveaux segments `kind="narration"`, portant la voix du narrateur
    (`ff_siwis` en fr, VOICE_NARRATOR en ja), sont insérés autour d'eux —
    c'est ce qui permet à assemble_audio() de synthétiser le préfixe/suffixe
    avec la voix du narrateur et le dialogue/sfx avec sa propre voix.
    Un dialogue n'est annoncé que lors d'un changement de locuteur : la
    comparaison se fait par `voice_id` (NarrativeSegment n'a pas de
    speaker_id — narrative_builder.py le résout déjà en voix concrète en
    amont) contre le `voice_id` du dialogue précédent dans le script
    (narration/sfx/scene_description intercalés n'interrompent pas le
    suivi). Limite connue : le pool ne compte que 3 voix japonaises
    (cluster_id % 3), donc deux personnages distincts dont les cluster_id
    tombent dans la même classe modulo 3 partagent un voice_id et seraient
    à tort traités comme le même locuteur — risque jugé acceptable pour
    l'instant. Un dialogue n'est également annoncé que s'il reste au moins
    un autre dialogue/sfx après lui dans le même panel : une réplique
    isolée, ou la dernière voix d'un panel, n'est jamais préfixée/suffixée
    — y compris un panel qui n'a qu'une seule réplique. SFX est toujours
    annoncé, à chaque occurrence (pas de suivi de locuteur, pas de condition
    de suite). narration, scene_description et thought ne sont jamais
    modifiés ni entourés (scene_description est toujours généré en français
    par Qwen3-VL, quel que soit `lang` ; thought est hors périmètre).

    Retourne `script` (même instance ; `script.segments` est remplacé par
    une nouvelle liste, mais les segments d'origine non transformés y sont
    conservés par référence) — à passer directement à `assemble_audio()`.
    """
    if lang not in _SUPPORTED_LANGS:
        raise ValueError(f"lang non supporté : {lang!r} (attendu : {_SUPPORTED_LANGS})")

    original_segments = script.segments
    enriched_segments: list[NarrativeSegment] = []
    last_dialogue_voice_id: str | None = None
    for index, segment in enumerate(original_segments):
        if segment.kind == "dialogue":
            same_speaker = segment.voice_id == last_dialogue_voice_id
            has_more = _has_following_dialogue_or_sfx_in_panel(original_segments, index)
            enriched_segments.extend(
                _dialogue_segments(segment, lang, same_speaker, has_more)
            )
            last_dialogue_voice_id = segment.voice_id
        elif segment.kind == "sfx":
            enriched_segments.extend(_sfx_segments(segment, lang))
        else:
            enriched_segments.append(segment)

    script.segments = enriched_segments
    return script
