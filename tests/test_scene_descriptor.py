"""Tests de la génération de description de scène par règles (sans VLM)."""

from __future__ import annotations

from manga_access.pipeline.scene_descriptor import describe_panel
from manga_access.schemas.manga_page import Panel, TextElement


def _make_element(id_: str, type_: str) -> TextElement:
    """Construit un TextElement minimal pour les tests."""
    return TextElement(
        id=id_,
        type=type_,  # type: ignore[arg-type]
        bbox=(0.0, 0.0, 10.0, 10.0),
        text_original="peu importe",
        confidence=1.0,
    )


def _make_panel(elements: list[TextElement] | None = None) -> Panel:
    """Construit un Panel minimal pour les tests."""
    return Panel(
        id="panel-0",
        order=0,
        bbox=(0.0, 0.0, 100.0, 100.0),
        elements=elements or [],
    )


def test_no_elements_no_characters_returns_none() -> None:
    """Exemple du cahier des charges : 0 élément, 0 personnage -> None."""
    panel = _make_panel(elements=[])

    assert describe_panel(panel, n_characters=0) is None


def test_no_elements_with_characters() -> None:
    """Exemple du cahier des charges : 0 élément, 2 personnages."""
    panel = _make_panel(elements=[])

    result = describe_panel(panel, n_characters=2)

    assert result == "2 personnages détectés. Aucun texte."


def test_single_dialogue_single_character() -> None:
    """Exemple du cahier des charges : 1 dialogue, 1 personnage."""
    panel = _make_panel(elements=[_make_element("text-1", "dialogue")])

    result = describe_panel(panel, n_characters=1)

    assert result == "1 personnage détecté. 1 dialogue."


def test_mixed_types_multiple_characters() -> None:
    """Exemple du cahier des charges : 2 dialogues, 3 personnages, 1 sfx."""
    panel = _make_panel(
        elements=[
            _make_element("text-1", "dialogue"),
            _make_element("text-2", "dialogue"),
            _make_element("text-3", "sfx"),
        ]
    )

    result = describe_panel(panel, n_characters=3)

    assert result == "3 personnages détectés. 2 dialogues, 1 effet sonore."


def test_elements_without_characters() -> None:
    """0 personnage sur la page mais des éléments dans le panel -> pas de phrase personnages."""
    panel = _make_panel(elements=[_make_element("text-1", "dialogue")])

    result = describe_panel(panel, n_characters=0)

    assert result == "1 dialogue."


def test_narration_pluralization() -> None:
    """Pluriel correct pour narration (1 vs 2)."""
    singular = _make_panel(elements=[_make_element("text-1", "narration")])
    plural = _make_panel(
        elements=[_make_element("text-1", "narration"), _make_element("text-2", "narration")]
    )

    assert describe_panel(singular, n_characters=0) == "1 narration."
    assert describe_panel(plural, n_characters=0) == "2 narrations."


def test_sfx_pluralization() -> None:
    """Pluriel correct pour sfx (1 effet sonore vs 2 effets sonores)."""
    singular = _make_panel(elements=[_make_element("text-1", "sfx")])
    plural = _make_panel(
        elements=[_make_element("text-1", "sfx"), _make_element("text-2", "sfx")]
    )

    assert describe_panel(singular, n_characters=0) == "1 effet sonore."
    assert describe_panel(plural, n_characters=0) == "2 effets sonores."


def test_thought_pluralization() -> None:
    """Pluriel correct pour thought (1 pensée vs 2 pensées)."""
    singular = _make_panel(elements=[_make_element("text-1", "thought")])
    plural = _make_panel(
        elements=[_make_element("text-1", "thought"), _make_element("text-2", "thought")]
    )

    assert describe_panel(singular, n_characters=0) == "1 pensée."
    assert describe_panel(plural, n_characters=0) == "2 pensées."


def test_all_types_mixed_uses_canonical_order() -> None:
    """Panel avec les 4 types mélangés : compte correct et ordre canonique dialogue/narration/sfx/thought."""
    panel = _make_panel(
        elements=[
            _make_element("text-1", "sfx"),
            _make_element("text-2", "thought"),
            _make_element("text-3", "dialogue"),
            _make_element("text-4", "narration"),
        ]
    )

    result = describe_panel(panel, n_characters=0)

    assert result == "1 dialogue, 1 narration, 1 effet sonore, 1 pensée."
