"""Tests de ChapterProcessor (structure chapitre-entier -> OCR -> vision, une passe chacune)."""

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
from manga_access.pipeline.chapter_processor import ChapterProcessor
from manga_access.schemas.manga_page import BBox, TextElement


class _FakeChapterStructureBackend(StructureBackend):
    """Backend de structure factice : detect_chapter renvoie des détections fixes par page."""

    def __init__(self, detections_per_page: list[dict[str, Any]]) -> None:
        self._detections_per_page = detections_per_page
        self.load_calls = 0
        self.unload_calls = 0
        self.detect_chapter_calls: list[tuple[int, dict[str, Any] | None]] = []

    def load(self) -> None:
        self.load_calls += 1

    def unload(self) -> None:
        self.unload_calls += 1

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        raise NotImplementedError("non utilisé par ChapterProcessor")

    def detect_chapter(
        self, images: list[np.ndarray], character_bank: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.detect_chapter_calls.append((len(images), character_bank))
        return self._detections_per_page


class _FakeOCRBackend(OCRBackend):
    """Backend OCR factice : un TextElement déterministe par bbox, compte load/unload."""

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
            id=f"text-{self._next_id}", type="dialogue", bbox=bbox, text_original="test", confidence=1.0
        )
        self._next_id += 1
        return element


class _FakeVisionBackend(SceneDescriptionBackend):
    """Backend vision factice : description fixe, compte load/unload."""

    def __init__(self) -> None:
        self.load_calls = 0
        self.unload_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def unload(self) -> None:
        self.unload_calls += 1

    def describe(self, image: Image.Image, bbox: BBox) -> str | None:
        return "Description VLM."


def _page_detections(character_name: str | None) -> dict[str, Any]:
    return {
        "panels": [[0, 0, 50, 50]],
        "texts": [[5, 5, 20, 20]],
        "text_character_associations": [(0, 0)],
        "character_cluster_labels": [0],
        "characters": [[0, 0, 10, 10]],
        "character_names": [character_name],
    }


def _write_pages(tmp_path: Path, count: int) -> list[Path]:
    paths = []
    for i in range(count):
        path = tmp_path / f"page-{i:03d}.png"
        Image.new("RGB", (100, 100), color="white").save(path)
        paths.append(path)
    return paths


def test_process_calls_detect_chapter_once_with_all_pages(tmp_path: Path) -> None:
    """detect_chapter() reçoit toutes les pages en un seul appel, pas un par page."""
    image_paths = _write_pages(tmp_path, count=3)
    structure_backend = _FakeChapterStructureBackend(
        [_page_detections("Naruto") for _ in range(3)]
    )
    processor = ChapterProcessor(structure_backend, _FakeOCRBackend())

    processor.process(image_paths)

    assert len(structure_backend.detect_chapter_calls) == 1
    assert structure_backend.detect_chapter_calls[0][0] == 3
    assert structure_backend.load_calls == structure_backend.unload_calls == 1


def test_process_propagates_character_bank_to_detect_chapter(tmp_path: Path) -> None:
    """La character_bank passée à process() est transmise telle quelle à detect_chapter()."""
    image_paths = _write_pages(tmp_path, count=1)
    structure_backend = _FakeChapterStructureBackend([_page_detections("Naruto")])
    bank = {"images": [np.zeros((5, 5, 3), dtype=np.uint8)], "names": ["Naruto"]}
    processor = ChapterProcessor(structure_backend, _FakeOCRBackend())

    processor.process(image_paths, character_bank=bank)

    assert structure_backend.detect_chapter_calls[0][1] is bank


def test_process_assigns_consistent_name_across_pages(tmp_path: Path) -> None:
    """Le même personnage nommé sur deux pages garde le même Character.name sur les deux MangaPage."""
    image_paths = _write_pages(tmp_path, count=2)
    structure_backend = _FakeChapterStructureBackend(
        [_page_detections("Naruto"), _page_detections("Naruto")]
    )
    processor = ChapterProcessor(structure_backend, _FakeOCRBackend())

    pages = processor.process(image_paths)

    assert len(pages) == 2
    assert pages[0].characters[0].name == "Naruto"
    assert pages[1].characters[0].name == "Naruto"


def test_process_unmatched_character_has_no_name(tmp_path: Path) -> None:
    """Un personnage non matché contre la character_bank ("Other") donne Character.name=None."""
    image_paths = _write_pages(tmp_path, count=1)
    structure_backend = _FakeChapterStructureBackend([_page_detections("Other")])
    processor = ChapterProcessor(structure_backend, _FakeOCRBackend())

    pages = processor.process(image_paths)

    assert pages[0].characters[0].name is None


def test_process_ocr_backend_loaded_and_unloaded_once_for_whole_chapter(tmp_path: Path) -> None:
    """ocr_backend est chargé/déchargé une seule fois pour tout le chapitre (pas par page)."""
    image_paths = _write_pages(tmp_path, count=3)
    structure_backend = _FakeChapterStructureBackend(
        [_page_detections("Naruto") for _ in range(3)]
    )
    ocr_backend = _FakeOCRBackend()
    processor = ChapterProcessor(structure_backend, ocr_backend)

    processor.process(image_paths)

    assert ocr_backend.load_calls == ocr_backend.unload_calls == 1


def test_process_attaches_ocr_text_to_each_page(tmp_path: Path) -> None:
    """Chaque page reçoit son propre texte reconnu par l'OCR (pas seulement la première)."""
    image_paths = _write_pages(tmp_path, count=2)
    structure_backend = _FakeChapterStructureBackend(
        [_page_detections("Naruto"), _page_detections("Naruto")]
    )
    processor = ChapterProcessor(structure_backend, _FakeOCRBackend())

    pages = processor.process(image_paths)

    assert len(pages[0].panels[0].elements) == 1
    assert len(pages[1].panels[0].elements) == 1


def test_process_vision_backend_loaded_and_unloaded_once_for_whole_chapter(tmp_path: Path) -> None:
    """vision_backend, quand fourni, est chargé/déchargé une seule fois pour tout le chapitre."""
    image_paths = _write_pages(tmp_path, count=3)
    structure_backend = _FakeChapterStructureBackend(
        [_page_detections("Naruto") for _ in range(3)]
    )
    vision_backend = _FakeVisionBackend()
    processor = ChapterProcessor(structure_backend, _FakeOCRBackend(), vision_backend)

    pages = processor.process(image_paths)

    assert vision_backend.load_calls == vision_backend.unload_calls == 1
    assert all(panel.scene_description == "Description VLM." for page in pages for panel in page.panels)


def test_process_without_vision_backend_uses_rule_based_description(tmp_path: Path) -> None:
    """Sans vision_backend, chaque panel reçoit une description par règles (describe_panel)."""
    image_paths = _write_pages(tmp_path, count=1)
    structure_backend = _FakeChapterStructureBackend([_page_detections("Naruto")])
    processor = ChapterProcessor(structure_backend, _FakeOCRBackend())

    pages = processor.process(image_paths)

    assert pages[0].panels[0].scene_description is not None
