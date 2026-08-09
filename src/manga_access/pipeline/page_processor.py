"""Pipeline minimal : orchestration Magiv2 (structure) -> manga-ocr (texte) -> MangaPage."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from PIL import Image

from manga_access.backends.base import OCRBackend, StructureBackend
from manga_access.schemas.manga_page import BBox, Character, MangaPage, Panel


def _find_panel_for_text(panels: list[Panel], text_bbox: BBox) -> Panel | None:
    """Retourne le panel dont la bbox contient le centre de `text_bbox`, sinon None."""
    cx = (text_bbox[0] + text_bbox[2]) / 2
    cy = (text_bbox[1] + text_bbox[3]) / 2
    for panel in panels:
        ox1, oy1, ox2, oy2 = panel.bbox
        if ox1 <= cx <= ox2 and oy1 <= cy <= oy2:
            return panel
    return None


class PageProcessor:
    """Orchestration minimale d'une planche : structure -> OCR -> MangaPage."""

    def __init__(self, structure_backend: StructureBackend, ocr_backend: OCRBackend) -> None:
        self._structure_backend = structure_backend
        self._ocr_backend = ocr_backend

    def process(self, image_path: Path) -> MangaPage:
        """Traite la planche à `image_path` et retourne un MangaPage validé."""
        image = Image.open(image_path).convert("RGB")

        self._structure_backend.load()
        detections = self._structure_backend.detect(image)
        self._structure_backend.unload()

        panels = [
            Panel(id=f"panel-{i}", order=i, bbox=tuple(float(v) for v in bbox))
            for i, bbox in enumerate(detections.get("panels", []))
        ]
        text_bboxes = [tuple(float(v) for v in bbox) for bbox in detections.get("texts", [])]

        self._ocr_backend.load()
        for text_bbox in text_bboxes:
            element = self._ocr_backend.recognize(image, text_bbox)
            panel = _find_panel_for_text(panels, text_bbox)
            if panel is not None:
                panel.elements.append(element)
            # else : centre du texte hors de tout panel détecté — élément
            # ignoré silencieusement dans cette version minimale (Phase 1).
            # À logger en Phase 2.
        self._ocr_backend.unload()

        return MangaPage(
            source={"file": str(image_path), "page_index": 0},
            reading_direction="right_to_left",
            characters=self._build_characters(detections),
            panels=panels,
        )

    @staticmethod
    def _build_characters(detections: dict) -> list[Character]:
        cluster_labels = detections.get("character_cluster_labels", [])
        return [
            Character(
                id=f"char-{cluster_id}",
                voice_id=f"voice_{cluster_id}",
                name=None,
                cluster_confidence=1.0,  # Magiv2 ne fournit pas de score de confiance par cluster
            )
            for cluster_id in sorted(set(cluster_labels))
        ]
