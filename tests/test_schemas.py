"""Tests du schéma d'échange MangaPage JSON v1."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from manga_access.schemas.manga_page import Character, MangaPage, Panel, TextElement


def _make_full_manga_page() -> MangaPage:
    """Construit un MangaPage complet avec tous les champs optionnels renseignés."""
    element = TextElement(
        id="text-1",
        type="dialogue",
        bbox=(10.0, 20.0, 100.0, 60.0),
        text_original="こんにちは",
        speaker_id="char-1",
        confidence=0.95,
    )
    panel = Panel(
        id="panel-1",
        order=0,
        bbox=(0.0, 0.0, 500.0, 400.0),
        elements=[element],
        scene_description="Deux personnages dans une rue.",
    )
    character = Character(
        id="char-1",
        voice_id="voice-a",
        name="Taro",
        cluster_confidence=0.88,
    )
    return MangaPage(
        source={"file": "volume1/page001.jpg", "page_number": 1},
        reading_direction="right_to_left",
        characters=[character],
        panels=[panel],
    )


def test_manga_page_roundtrip() -> None:
    """to_json() puis from_json() doit reproduire un MangaPage identique."""
    original = _make_full_manga_page()

    serialized = original.to_json()
    restored = MangaPage.from_json(serialized)

    assert restored == original


def test_text_element_type_rejects_invalid_literal() -> None:
    """Un type hors de la liste autorisée doit lever ValidationError."""
    with pytest.raises(ValidationError):
        TextElement(
            id="text-1",
            type="monologue",  # invalide : hors Literal["dialogue","narration","sfx","thought"]
            bbox=(0.0, 0.0, 10.0, 10.0),
            text_original="...",
            confidence=0.5,
        )


def test_confidence_out_of_range_rejected() -> None:
    """confidence hors [0.0, 1.0] doit lever ValidationError, sur TextElement et Character."""
    with pytest.raises(ValidationError):
        TextElement(
            id="text-1",
            type="dialogue",
            bbox=(0.0, 0.0, 10.0, 10.0),
            text_original="...",
            confidence=1.5,
        )

    with pytest.raises(ValidationError):
        Character(id="char-1", voice_id="voice-a", cluster_confidence=-0.1)


def test_optional_fields_accept_none() -> None:
    """Les champs Optional (speaker_id, name, scene_description) acceptent None."""
    element = TextElement(
        id="text-1",
        type="narration",
        bbox=(0.0, 0.0, 10.0, 10.0),
        text_original="...",
        speaker_id=None,
        confidence=0.5,
    )
    assert element.speaker_id is None

    panel = Panel(
        id="panel-1",
        order=0,
        bbox=(0.0, 0.0, 10.0, 10.0),
        scene_description=None,
    )
    assert panel.scene_description is None

    character = Character(
        id="char-1",
        voice_id="voice-a",
        name=None,
        cluster_confidence=0.5,
    )
    assert character.name is None
