"""Orchestration chapitre-entier : structuration (+ identification nominative) -> OCR -> MangaPage[].

Contrairement à PageProcessor (mono-page), s'appuie sur
`StructureBackend.detect_chapter` pour traiter toutes les pages du chapitre
en un seul appel modèle — seule façon d'obtenir une identification
nominative des personnages cohérente entre les pages via une character_bank
(cf. docstring de `detect_chapter`).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from manga_access.backends.base import (
    OCRBackend,
    SceneDescriptionBackend,
    StructureBackend,
)
from manga_access.pipeline.page_processor import (
    _build_characters,
    _build_panels,
    _full_image_bbox,
    _run_ocr_for_page,
)
from manga_access.pipeline.scene_descriptor import describe_panel
from manga_access.schemas.manga_page import MangaPage, Panel


class ChapterProcessor:
    """Orchestration d'un chapitre entier : structure -> OCR -> vision, chacun en une seule passe.

    Charge chaque backend une seule fois pour tout le chapitre (contrainte
    RAM du projet : un seul modèle lourd en mémoire à la fois), à l'inverse
    de PageProcessor qui recharge chaque backend par page.
    """

    def __init__(
        self,
        structure_backend: StructureBackend,
        ocr_backend: OCRBackend,
        vision_backend: SceneDescriptionBackend | None = None,
    ) -> None:
        self._structure_backend = structure_backend
        self._ocr_backend = ocr_backend
        self._vision_backend = vision_backend

    def process(
        self, image_paths: list[Path], character_bank: dict | None = None
    ) -> list[MangaPage]:
        """Traite toutes les planches de `image_paths`, dans l'ordre de lecture.

        `character_bank` (optionnel) = {"images": [...], "names": [...]} de
        portraits de référence, transmis à `detect_chapter` pour
        l'identification nominative des personnages.
        """
        images = [Image.open(p).convert("RGB") for p in image_paths]

        self._structure_backend.load()
        detections_per_page = self._structure_backend.detect_chapter(images, character_bank)
        self._structure_backend.unload()

        panels_per_page: list[list[Panel]] = []
        characters_per_page = []
        for image, detections in zip(images, detections_per_page):
            width, height = image.size
            panels = _build_panels(detections, width, height)
            characters = _build_characters(detections, detections.get("character_names"))
            panels_per_page.append(panels)
            characters_per_page.append(characters)

        self._ocr_backend.load()
        for image, detections, panels in zip(images, detections_per_page, panels_per_page):
            _run_ocr_for_page(image, detections, panels, self._ocr_backend)
        self._ocr_backend.unload()

        if self._vision_backend is not None:
            self._vision_backend.load()
            for image, panels in zip(images, panels_per_page):
                description = self._vision_backend.describe(image, _full_image_bbox(image))
                for panel in panels:
                    panel.scene_description = description
            self._vision_backend.unload()
        else:
            for panels, characters in zip(panels_per_page, characters_per_page):
                for panel in panels:
                    panel.scene_description = describe_panel(panel, n_characters=len(characters))

        return [
            MangaPage(
                source={"file": str(image_path), "page_index": index},
                reading_direction="right_to_left",
                characters=characters,
                panels=panels,
            )
            for index, (image_path, characters, panels) in enumerate(
                zip(image_paths, characters_per_page, panels_per_page)
            )
        ]
