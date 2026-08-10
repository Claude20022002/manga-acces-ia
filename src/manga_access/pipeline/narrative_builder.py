"""Assemblage du script narratif depuis un MangaPage, prêt pour la synthèse TTS."""

from __future__ import annotations

from manga_access.schemas.manga_page import MangaPage, Panel, TextElement
from manga_access.schemas.narrative_script import NarrativeScript, NarrativeSegment

VOICE_NARRATOR = "af_sky"
VOICE_UNKNOWN = "af_bella"


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
        segments.append(
            NarrativeSegment(
                id=f"seg-{element.id}",
                panel_id=panel.id,
                kind=element.type,
                voice_id=_voice_for_element(element, character_voices),
                text=_text_for_element(element),
                source_element_id=element.id,
            )
        )

    return segments, last_scene_description


def build_narrative_script(page: MangaPage) -> NarrativeScript:
    """Assemble le script narratif ordonné d'une planche depuis son MangaPage.

    Parcourt les panels dans l'ordre de `Panel.order` (cohérent avec
    `reading_direction`), et pour chacun : un segment de description de scène
    s'il est renseigné, puis un segment par élément de texte détecté.
    """
    character_voices = {character.id: character.voice_id for character in page.characters}
    ordered_panels = sorted(page.panels, key=lambda panel: panel.order)

    segments: list[NarrativeSegment] = []
    last_scene_description: str | None = None
    for panel in ordered_panels:
        panel_segments, last_scene_description = _segments_for_panel(
            panel, character_voices, last_scene_description
        )
        segments.extend(panel_segments)

    return NarrativeScript(source=page.source, segments=segments)
