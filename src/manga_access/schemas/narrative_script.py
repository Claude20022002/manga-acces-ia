"""Schéma du script narratif assemblé depuis un MangaPage, prêt pour la synthèse TTS."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class NarrativeSegment(BaseModel):
    """Un segment de texte à synthétiser, associé à une voix TTS."""

    id: str
    panel_id: str
    kind: Literal["scene_description", "dialogue", "narration", "sfx", "thought"]
    voice_id: str
    text: str
    source_element_id: str | None = None


class NarrativeScript(BaseModel):
    """Script narratif ordonné d'une planche, prêt pour la synthèse TTS.

    Assemblé depuis un MangaPage validé (cf. build_narrative_script dans
    pipeline/narrative_builder.py). Chaque segment correspond à un futur
    appel TTSBackend.synthesize(text, voice_id).
    """

    schema_version: Literal["1.0"] = "1.0"
    source: dict[str, Any]
    segments: list[NarrativeSegment] = Field(default_factory=list)

    def to_json(self) -> str:
        """Sérialise le script en JSON (UTF-8, indenté)."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> NarrativeScript:
        """Désérialise un script depuis une chaîne JSON."""
        return cls.model_validate_json(data)
