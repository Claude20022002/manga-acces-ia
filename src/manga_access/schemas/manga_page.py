"""Schéma d'échange inter-modules MangaPage JSON v1.

Ce module définit le contrat unique de communication entre les backends
du pipeline (OCR, structuration, TTS, etc.). Aucun module métier ne doit
échanger de données via un format différent.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]
"""Boîte englobante (x1, y1, x2, y2) en pixels, origine en haut à gauche de l'image source."""


class TextElement(BaseModel):
    """Un élément de texte détecté dans une case (bulle, narration, onomatopée...)."""

    id: str
    type: Literal["dialogue", "narration", "sfx", "thought"]
    bbox: BBox
    text_original: str
    speaker_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class Panel(BaseModel):
    """Une case de la planche, avec ses éléments de texte."""

    id: str
    order: int
    bbox: BBox
    elements: list[TextElement] = Field(default_factory=list)
    scene_description: str | None = None


class Character(BaseModel):
    """Un personnage identifié sur la planche, associé à une voix TTS."""

    id: str
    voice_id: str
    name: str | None = None
    cluster_confidence: float = Field(ge=0.0, le=1.0)


class MangaPage(BaseModel):
    """Représentation complète d'une planche de manga après traitement.

    C'est le seul format d'échange autorisé entre les backends du pipeline.
    """

    schema_version: Literal["1.0"] = "1.0"
    source: dict[str, Any]  # métadonnées libres (chemin fichier, page...), non structurées à ce niveau
    reading_direction: Literal["right_to_left", "left_to_right"]
    characters: list[Character] = Field(default_factory=list)
    panels: list[Panel] = Field(default_factory=list)

    def to_json(self) -> str:
        """Sérialise la planche en JSON (UTF-8, indenté)."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> MangaPage:
        """Désérialise une planche depuis une chaîne JSON."""
        return cls.model_validate_json(data)
