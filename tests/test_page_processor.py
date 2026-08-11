"""Tests de PageProcessor (orchestration mono-page structure -> OCR -> MangaPage).

Aucune couverture n'existait avant Phase 9 pour ce module. Ajoutée en
accompagnement du refactor qui extrait _build_panels/_run_ocr_for_page/
_build_characters, réutilisés par ChapterProcessor (cf.
test_chapter_processor.py) : ces tests figent le comportement observable de
PageProcessor.process() pour garantir la non-régression du refactor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from manga_access.backends.base import (
    OCRBackend,
    SceneDescriptionBackend,
    StructureBackend,
)
from manga_access.pipeline.page_processor import PageProcessor
from manga_access.schemas.manga_page import BBox, TextElement


class _FakeStructureBackend(StructureBackend):
    """Backend de structure factice : détections fixes, compte load/unload."""

    def __init__(self, detections: dict[str, Any]) -> None:
        self._detections = detections
        self.load_calls = 0
        self.unload_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def unload(self) -> None:
        self.unload_calls += 1

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        return self._detections

    def detect_chapter(
        self, images: list[np.ndarray], character_bank: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("non utilisé par PageProcessor")


class _FakeOCRBackend(OCRBackend):
    """Backend OCR factice : un TextElement déterministe par bbox reconnue."""

    def __init__(self) -> None:
        self.load_calls = 0
        self.unload_calls = 0
        self._next_id = 0

    def load(self) -> None:
        self.load_calls += 1

    def unload(self) -> None:
        self.unload_calls += 1

    def recognize(self, image: Image.Image, bbox: BBox) -> TextElement:
        element = TextElement(
            id=f"text-{self._next_id}",
            type="dialogue",
            bbox=bbox,
            text_original="こんにちは",
            confidence=1.0,
        )
        self._next_id += 1
        return element


class _FakeVisionBackend(SceneDescriptionBackend):
    """Backend vision factice : description fixe, compte load/unload."""

    def __init__(self, description: str = "Une rue calme.") -> None:
        self._description = description
        self.load_calls = 0
        self.unload_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def unload(self) -> None:
        self.unload_calls += 1

    def describe(self, image: Image.Image, bbox: BBox) -> str | None:
        return self._description


def _make_detections() -> dict[str, Any]:
    """Détections brutes minimales : un panel, un texte associé à un personnage (cluster 0)."""
    return {
        "panels": [[0, 0, 50, 50]],
        "texts": [[5, 5, 20, 20]],
        "text_character_associations": [(0, 0)],
        "character_cluster_labels": [0],
    }


def _write_test_image(tmp_path: Path) -> Path:
    image_path = tmp_path / "page-001.png"
    Image.new("RGB", (100, 100), color="white").save(image_path)
    return image_path


def test_process_builds_panels_and_attaches_text(tmp_path: Path) -> None:
    """process() construit les panels détectés et y attache le texte reconnu."""
    image_path = _write_test_image(tmp_path)
    structure_backend = _FakeStructureBackend(_make_detections())
    ocr_backend = _FakeOCRBackend()
    processor = PageProcessor(structure_backend, ocr_backend)

    page = processor.process(image_path)

    assert len(page.panels) == 1
    assert len(page.panels[0].elements) == 1
    assert page.panels[0].elements[0].text_original == "こんにちは"


def test_process_resolves_speaker_id_from_character_cluster(tmp_path: Path) -> None:
    """Le speaker_id de l'élément texte est résolu depuis l'association texte<->personnage."""
    image_path = _write_test_image(tmp_path)
    structure_backend = _FakeStructureBackend(_make_detections())
    processor = PageProcessor(structure_backend, _FakeOCRBackend())

    page = processor.process(image_path)

    assert page.panels[0].elements[0].speaker_id == "char-0"


def test_process_builds_character_without_name(tmp_path: Path) -> None:
    """Sans character_bank (chemin PageProcessor mono-page), Character.name reste None."""
    image_path = _write_test_image(tmp_path)
    structure_backend = _FakeStructureBackend(_make_detections())
    processor = PageProcessor(structure_backend, _FakeOCRBackend())

    page = processor.process(image_path)

    assert len(page.characters) == 1
    assert page.characters[0].id == "char-0"
    assert page.characters[0].name is None


def test_process_loads_and_unloads_each_backend_once(tmp_path: Path) -> None:
    """Chaque backend (structure, OCR) est chargé puis déchargé exactement une fois."""
    image_path = _write_test_image(tmp_path)
    structure_backend = _FakeStructureBackend(_make_detections())
    ocr_backend = _FakeOCRBackend()
    processor = PageProcessor(structure_backend, ocr_backend)

    processor.process(image_path)

    assert structure_backend.load_calls == structure_backend.unload_calls == 1
    assert ocr_backend.load_calls == ocr_backend.unload_calls == 1


def test_process_without_vision_backend_uses_rule_based_scene_description(tmp_path: Path) -> None:
    """Sans vision_backend, panel.scene_description vient de describe_panel() (règles)."""
    image_path = _write_test_image(tmp_path)
    structure_backend = _FakeStructureBackend(_make_detections())
    processor = PageProcessor(structure_backend, _FakeOCRBackend())

    page = processor.process(image_path)

    assert page.panels[0].scene_description is not None


def test_process_with_vision_backend_uses_vlm_description(tmp_path: Path) -> None:
    """Avec vision_backend, panel.scene_description vient de describe() (Qwen3-VL), chargé/déchargé."""
    image_path = _write_test_image(tmp_path)
    structure_backend = _FakeStructureBackend(_make_detections())
    vision_backend = _FakeVisionBackend(description="Une scène décrite par le VLM.")
    processor = PageProcessor(structure_backend, _FakeOCRBackend(), vision_backend)

    page = processor.process(image_path)

    assert page.panels[0].scene_description == "Une scène décrite par le VLM."
    assert vision_backend.load_calls == vision_backend.unload_calls == 1
