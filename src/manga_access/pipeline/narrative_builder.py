"""Assemblage du script narratif depuis un MangaPage, prêt pour la synthèse TTS."""

from __future__ import annotations

import zlib

from manga_access.schemas.manga_page import MangaPage, Panel, TextElement
from manga_access.schemas.narrative_script import NarrativeScript, NarrativeSegment

VOICE_NARRATOR = "jf_tebukuro"
VOICE_UNKNOWN = "jf_nezumi"

_CHARACTER_VOICE_POOL = ("jf_alpha", "jf_gongitsune", "jm_kumo")


def _voice_for_character(character_id: str, name: str | None) -> str:
    """Détermine la voix TTS japonaise d'un personnage.

    Si `name` est connu (identification par character_bank, Phase 9), la
    voix est dérivée d'un hash stable du nom plutôt que du cluster_id : le
    cluster_id est local à chaque page (PageProcessor/ChapterProcessor), donc
    un même personnage nommé changerait sinon de voix d'une page à l'autre
    malgré un nom annoncé stable. Sans nom, fallback sur l'ancien schéma
    (cluster_id % 3, character_id au format "char-{cluster_id}").
    """
    if name is not None:
        index = zlib.crc32(name.encode()) % len(_CHARACTER_VOICE_POOL)
        return _CHARACTER_VOICE_POOL[index]
    cluster_id = int(character_id.rsplit("-", 1)[-1])
    return _CHARACTER_VOICE_POOL[cluster_id % len(_CHARACTER_VOICE_POOL)]


def _voice_for_element(element: TextElement, character_voices: dict[str, str]) -> str:
    """Détermine la voix TTS d'un élément de texte selon son type et son locuteur."""
    if element.type in ("dialogue", "thought"):
        if element.speaker_id is not None and element.speaker_id in character_voices:
            return character_voices[element.speaker_id]
        return VOICE_UNKNOWN
    return VOICE_NARRATOR


def _text_for_element(element: TextElement) -> str:
    """Formate le texte d'un élément selon son type (SFX entre crochets)."""
    if element.type == "sfx":
        return f"[{element.text_original}]"
    return element.text_original


def _segments_for_panel(
    panel: Panel,
    character_voices: dict[str, str],
    character_names: dict[str, str],
    last_scene_description: str | None,
) -> tuple[list[NarrativeSegment], str | None]:
    """Construit les segments narratifs d'un panel : description de scène (si nouvelle) puis éléments.

    La description de scène n'est émise que si elle diffère de
    `last_scene_description` : évite de répéter le même paragraphe à
    chaque panel quand un vision_backend (Qwen3-VL) décrit la planche
    entière et applique la même description à tous ses panels
    (cf. PageProcessor.process()). Retourne le texte de scène courant pour
    que l'appelant le propage au panel suivant.
    """
    segments: list[NarrativeSegment] = []

    if panel.scene_description is not None and panel.scene_description != last_scene_description:
        segments.append(
            NarrativeSegment(
                id=f"seg-{panel.id}-scene",
                panel_id=panel.id,
                kind="scene_description",
                voice_id=VOICE_NARRATOR,
                text=panel.scene_description,
            )
        )
        last_scene_description = panel.scene_description

    # TODO(Phase 3) : panel.elements n'est pas trié spatialement, c'est l'ordre
    # de détection Magiv2 brut — pas garanti top-to-bottom / droite-à-gauche.
    for element in panel.elements:
        character_name = (
            character_names.get(element.speaker_id) if element.speaker_id is not None else None
        )
        segments.append(
            NarrativeSegment(
                id=f"seg-{element.id}",
                panel_id=panel.id,
                kind=element.type,
                voice_id=_voice_for_element(element, character_voices),
                text=_text_for_element(element),
                source_element_id=element.id,
                character_name=character_name,
            )
        )

    return segments, last_scene_description


def build_narrative_script(page: MangaPage) -> NarrativeScript:
    """Assemble le script narratif ordonné d'une planche depuis son MangaPage.

    Parcourt les panels dans l'ordre de `Panel.order` (cohérent avec
    `reading_direction`), et pour chacun : un segment de description de scène
    s'il est renseigné, puis un segment par élément de texte détecté.
    """
    character_voices = {
        character.id: _voice_for_character(character.id, character.name)
        for character in page.characters
    }
    character_names = {
        character.id: character.name for character in page.characters if character.name is not None
    }
    ordered_panels = sorted(page.panels, key=lambda panel: panel.order)

    segments: list[NarrativeSegment] = []
    last_scene_description: str | None = None
    for panel in ordered_panels:
        panel_segments, last_scene_description = _segments_for_panel(
            panel, character_voices, character_names, last_scene_description
        )
        segments.extend(panel_segments)

    return NarrativeScript(source=page.source, segments=segments)
