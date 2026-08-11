"""Tests des interfaces abstraites des backends."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image

from manga_access.backends.base import OCRBackend, StructureBackend, TTSBackend
from manga_access.schemas.manga_page import BBox, TextElement


def test_ocr_backend_is_abstract() -> None:
    """OCRBackend ne peut pas être instanciée directement."""
    with pytest.raises(TypeError):
        OCRBackend()  # type: ignore[abstract]


def test_structure_backend_is_abstract() -> None:
    """StructureBackend ne peut pas être instanciée directement."""
    with pytest.raises(TypeError):
        StructureBackend()  # type: ignore[abstract]


def test_tts_backend_is_abstract() -> None:
    """TTSBackend ne peut pas être instanciée directement."""
    with pytest.raises(TypeError):
        TTSBackend()  # type: ignore[abstract]


class _FullBackend(OCRBackend, TTSBackend):
    """Double héritage implémentant toutes les méthodes abstraites requises."""

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

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
    """Implémente load/unload/recognize (OCRBackend) mais pas synthesize (TTSBackend)."""

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def recognize(self, image: Image.Image, bbox: BBox) -> TextElement:
        return TextElement(
            id="text-1",
            type="dialogue",
            bbox=bbox,
            text_original="stub",
            confidence=1.0,
        )


class _FullStructureBackend(StructureBackend):
    """Implémente les quatre méthodes abstraites de StructureBackend."""

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        return {"panels": [], "texts": []}

    def detect_chapter(
        self, images: list[np.ndarray], character_bank: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return [{"panels": [], "texts": []} for _ in images]


class _PartialStructureBackend(StructureBackend):
    """Implémente load/detect mais pas unload."""

    def load(self) -> None:
        pass

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        return {"panels": [], "texts": []}


def test_concrete_subclass_implementing_both_methods_is_instantiable() -> None:
    """Une sous-classe implémentant load/unload/recognize et synthesize s'instancie sans erreur."""
    backend = _FullBackend()
    assert isinstance(backend, OCRBackend)
    assert isinstance(backend, TTSBackend)


def test_partial_implementation_is_not_instantiable() -> None:
    """Une sous-classe n'implémentant pas synthesize reste abstraite."""
    with pytest.raises(TypeError):
        _PartialBackend()


def test_structure_backend_full_implementation_is_instantiable() -> None:
    """Une sous-classe implémentant load/unload/detect s'instancie sans erreur."""
    backend = _FullStructureBackend()
    assert isinstance(backend, StructureBackend)


def test_structure_backend_partial_implementation_is_not_instantiable() -> None:
    """Une sous-classe n'implémentant pas unload reste abstraite."""
    with pytest.raises(TypeError):
        _PartialStructureBackend()
