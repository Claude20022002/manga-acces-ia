"""Tests de l'assemblage du script narratif depuis MangaPage."""

from __future__ import annotations

from manga_access.pipeline.narrative_builder import (
    VOICE_NARRATOR,
    VOICE_UNKNOWN,
    _voice_for_character,
    build_narrative_script,
)
from manga_access.schemas.manga_page import Character, MangaPage, Panel, TextElement
from manga_access.schemas.narrative_script import NarrativeScript


def _make_element(
    id_: str, type_: str, text: str, speaker_id: str | None = None
) -> TextElement:
    """Construit un TextElement minimal pour les tests."""
    return TextElement(
        id=id_,
        type=type_,  # type: ignore[arg-type]
        bbox=(0.0, 0.0, 10.0, 10.0),
        text_original=text,
        speaker_id=speaker_id,
        confidence=1.0,
    )


def _make_panel(
    id_: str,
    order: int,
    elements: list[TextElement] | None = None,
    scene_description: str | None = None,
) -> Panel:
    """Construit un Panel minimal pour les tests."""
    return Panel(
        id=id_,
        order=order,
        bbox=(0.0, 0.0, 100.0, 100.0),
        elements=elements or [],
        scene_description=scene_description,
    )


def _make_page(panels: list[Panel], characters: list[Character] | None = None) -> MangaPage:
    """Construit un MangaPage minimal pour les tests."""
    return MangaPage(
        source={"file": "test.jpg", "page_index": 0},
        reading_direction="right_to_left",
        characters=characters or [],
        panels=panels,
    )


def test_empty_page() -> None:
    """Un MangaPage sans panels produit un NarrativeScript sans segments."""
    page = _make_page(panels=[])

    script = build_narrative_script(page)

    assert script.segments == []


def test_dialogue_with_speaker() -> None:
    """Un dialogue avec speaker_id résolu utilise la voix du pool japonais (cluster_id % 3)."""
    element = _make_element("text-1", "dialogue", "おはよう", speaker_id="char-1")
    panel = _make_panel("panel-0", order=0, elements=[element])
    character = Character(id="char-1", voice_id="voice-a", cluster_confidence=1.0)
    page = _make_page([panel], characters=[character])

    script = build_narrative_script(page)

    assert script.segments[0].voice_id == "jf_gongitsune"


def test_voice_for_character_pool_by_cluster_id() -> None:
    """cluster_id % 3 sélectionne déterministiquement dans le pool japonais."""
    assert _voice_for_character("char-0") == "jf_alpha"
    assert _voice_for_character("char-1") == "jf_gongitsune"
    assert _voice_for_character("char-2") == "jm_kumo"
    assert _voice_for_character("char-3") == "jf_alpha"


def test_dialogue_unknown_speaker() -> None:
    """Un dialogue sans speaker_id résolu retombe sur VOICE_UNKNOWN."""
    element = _make_element("text-1", "dialogue", "おはよう", speaker_id=None)
    panel = _make_panel("panel-0", order=0, elements=[element])
    page = _make_page([panel])

    script = build_narrative_script(page)

    assert script.segments[0].voice_id == VOICE_UNKNOWN


def test_sfx_formatting() -> None:
    """Un SFX est formaté entre crochets."""
    element = _make_element("text-1", "sfx", "ドン")
    panel = _make_panel("panel-0", order=0, elements=[element])
    page = _make_page([panel])

    script = build_narrative_script(page)

    assert script.segments[0].text == "[ドン]"


def test_narration_voice() -> None:
    """Une narration utilise VOICE_NARRATOR."""
    element = _make_element("text-1", "narration", "静かな夜だった。")
    panel = _make_panel("panel-0", order=0, elements=[element])
    page = _make_page([panel])

    script = build_narrative_script(page)

    assert script.segments[0].voice_id == VOICE_NARRATOR


def test_scene_description_skipped_when_none() -> None:
    """Un panel sans scene_description ne produit pas de segment scene_description."""
    panel = _make_panel("panel-0", order=0, elements=[], scene_description=None)
    page = _make_page([panel])

    script = build_narrative_script(page)

    assert script.segments == []


def test_panel_order() -> None:
    """Les panels sont ordonnés par Panel.order, indépendamment de leur ordre dans la liste."""
    panel_b = _make_panel(
        "panel-b", order=1, elements=[_make_element("text-b", "narration", "second")]
    )
    panel_a = _make_panel(
        "panel-a", order=0, elements=[_make_element("text-a", "narration", "first")]
    )
    page = _make_page([panel_b, panel_a])  # inséré dans le mauvais ordre

    script = build_narrative_script(page)

    assert [segment.panel_id for segment in script.segments] == ["panel-a", "panel-b"]


def test_roundtrip_json() -> None:
    """to_json() puis from_json() doit reproduire un NarrativeScript identique."""
    element = _make_element("text-1", "dialogue", "おはよう", speaker_id="char-1")
    panel = _make_panel("panel-0", order=0, elements=[element], scene_description="Une rue calme.")
    character = Character(id="char-1", voice_id="voice-a", cluster_confidence=1.0)
    page = _make_page([panel], characters=[character])
    original = build_narrative_script(page)

    serialized = original.to_json()
    restored = NarrativeScript.from_json(serialized)

    assert restored == original
