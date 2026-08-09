"""Tests des interfaces abstraites des backends."""

from __future__ import annotations

import pytest
from PIL import Image

from manga_access.backends.base import OCRBackend, TTSBackend
from manga_access.schemas.manga_page import BBox, TextElement


def test_ocr_backend_is_abstract() -> None:
    """OCRBackend ne peut pas être instanciée directement."""
    with pytest.raises(TypeError):
        OCRBackend()  # type: ignore[abstract]


def test_tts_backend_is_abstract() -> None:
    """TTSBackend ne peut pas être instanciée directement."""
    with pytest.raises(TypeError):
        TTSBackend()  # type: ignore[abstract]


class _FullBackend(OCRBackend, TTSBackend):
    """Double héritage implémentant les deux méthodes abstraites (recognize + synthesize)."""

    def recognize(self, image: Image.Image, bbox: BBox) -> TextElement:
        return TextElement(
            id="text-1",
            type="dialogue",
            bbox=bbox,
            text_original="stub",
            confidence=1.0,
        )

    def synthesize(self, text: str, voice_id: str) -> bytes:
        return b"stub-audio"


class _PartialBackend(OCRBackend, TTSBackend):
    """Double héritage n'implémentant que recognize, pas synthesize."""

    def recognize(self, image: Image.Image, bbox: BBox) -> TextElement:
        return TextElement(
            id="text-1",
            type="dialogue",
            bbox=bbox,
            text_original="stub",
            confidence=1.0,
        )


def test_concrete_subclass_implementing_both_methods_is_instantiable() -> None:
    """Une sous-classe implémentant recognize et synthesize s'instancie sans erreur."""
    backend = _FullBackend()
    assert isinstance(backend, OCRBackend)
    assert isinstance(backend, TTSBackend)


def test_partial_implementation_is_not_instantiable() -> None:
    """Une sous-classe n'implémentant qu'une des deux méthodes reste abstraite."""
    with pytest.raises(TypeError):
        _PartialBackend()
