"""Interfaces abstraites des backends du pipeline.

Chaque backend concret (OCR, TTS, etc.) doit implémenter une de ces
interfaces. La logique métier ne doit jamais importer un modèle
directement — toujours passer par ces contrats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from manga_access.schemas.manga_page import BBox, TextElement


class OCRBackend(ABC):
    """Contrat pour un backend de reconnaissance de texte japonais."""

    @abstractmethod
    def recognize(self, image: Image.Image, bbox: BBox) -> TextElement:
        """Reconnaît le texte contenu dans la région `bbox` de `image`.

        L'implémentation est responsable de recadrer l'image selon `bbox`
        avant reconnaissance et de renseigner `confidence` dans le
        `TextElement` retourné.
        """
        raise NotImplementedError


class TTSBackend(ABC):
    """Contrat pour un backend de synthèse vocale."""

    @abstractmethod
    def synthesize(self, text: str, voice_id: str) -> bytes:
        """Synthétise `text` avec la voix `voice_id` et retourne l'audio brut."""
        raise NotImplementedError
