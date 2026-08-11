"""Schéma de la timeline audio générée après synthèse TTS d'un NarrativeScript."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TimelineSegment(BaseModel):
    """Un segment audio synthétisé, avec ses bornes temporelles dans le fichier assemblé."""

    id: str
    kind: Literal["dialogue", "narration", "sfx", "thought", "scene_description"]
    text: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    page_index: int = 0


class Timeline(BaseModel):
    """Timeline ordonnée des segments audibles d'une planche, alignée sur le .opus produit.

    Assemblée par pipeline/audio_assembler.py::assemble_audio(), en parallèle
    de l'export du fichier audio. Les segments "scene_description" sont
    synthétisés comme les autres (déduplication en amont dans
    narrative_builder.py pour éviter les répétitions quand un vision_backend
    applique la même description à tous les panels d'une planche).
    """

    schema_version: Literal["1.0"] = "1.0"
    source: dict[str, Any]
    segments: list[TimelineSegment] = Field(default_factory=list)

    def to_json(self) -> str:
        """Sérialise la timeline en JSON (UTF-8, indenté)."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> Timeline:
        """Désérialise une timeline depuis une chaîne JSON."""
        return cls.model_validate_json(data)
