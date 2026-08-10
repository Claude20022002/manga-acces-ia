"""Pipeline minimal : orchestration Magiv2 (structure) -> manga-ocr (texte) -> MangaPage."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from PIL import Image

from manga_access.backends.base import OCRBackend, SceneDescriptionBackend, StructureBackend
from manga_access.pipeline.scene_descriptor import describe_panel
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


def _clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
    """Ramène `bbox` dans les bornes [0, width] x [0, height] de l'image source."""
    x1, y1, x2, y2 = bbox
    return (
        max(0.0, x1),
        max(0.0, y1),
        min(float(width), x2),
        min(float(height), y2),
    )


def _full_image_bbox(image: Image.Image) -> BBox:
    """Retourne la bbox couvrant l'image entière (pour describe() en contexte pleine page)."""
    w, h = image.size
    return (0.0, 0.0, float(w), float(h))


class PageProcessor:
    """Orchestration minimale d'une planche : structure -> OCR -> MangaPage."""

    def __init__(
        self,
        structure_backend: StructureBackend,
        ocr_backend: OCRBackend,
        vision_backend: SceneDescriptionBackend | None = None,
    ) -> None:
        self._structure_backend = structure_backend
        self._ocr_backend = ocr_backend
        self._vision_backend = vision_backend

    def process(self, image_path: Path) -> MangaPage:
        """Traite la planche à `image_path` et retourne un MangaPage validé."""
        image = Image.open(image_path).convert("RGB")

        self._structure_backend.load()
        detections = self._structure_backend.detect(image)
        self._structure_backend.unload()

        width, height = image.size
        panels = [
            Panel(
                id=f"panel-{i}",
                order=i,
                bbox=_clip_bbox(tuple(float(v) for v in bbox), width, height),
            )
            for i, bbox in enumerate(detections.get("panels", []))
        ]
        text_bboxes = [
            _clip_bbox(tuple(float(v) for v in bbox), width, height)
            for bbox in detections.get("texts", [])
        ]
        logger.info(f"Magiv2 : {len(text_bboxes)} bbox(es) texte brut(es) détectée(s)")

        text_to_char_idx = dict(detections.get("text_character_associations", []))
        character_cluster_labels = detections.get("character_cluster_labels", [])

        self._ocr_backend.load()
        for text_idx, text_bbox in enumerate(text_bboxes):
            element = self._ocr_backend.recognize(image, text_bbox)
            char_idx = text_to_char_idx.get(text_idx)
            if char_idx is not None:
                element.speaker_id = f"char-{character_cluster_labels[char_idx]}"
            panel = _find_panel_for_text(panels, text_bbox)
            if panel is not None:
                panel.elements.append(element)
            else:
                cx = (text_bbox[0] + text_bbox[2]) / 2
                cy = (text_bbox[1] + text_bbox[3]) / 2
                logger.warning(
                    f"Texte hors panel ignoré : centre=({cx:.1f}, {cy:.1f}) "
                    f"bbox={tuple(round(v, 1) for v in text_bbox)}"
                )
        self._ocr_backend.unload()

        characters = self._build_characters(detections)
        if self._vision_backend is not None:
            self._vision_backend.load()
            description = self._vision_backend.describe(image, _full_image_bbox(image))
            self._vision_backend.unload()
            for panel in panels:
                panel.scene_description = description
        else:
            for panel in panels:
                panel.scene_description = describe_panel(panel, n_characters=len(characters))

        return MangaPage(
            source={"file": str(image_path), "page_index": 0},
            reading_direction="right_to_left",
            characters=characters,
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
