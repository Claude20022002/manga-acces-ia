"""Backend OCR manga-ocr (kha-white/manga-ocr-base)."""

from __future__ import annotations

import gc
import time
import uuid
from typing import Any

from loguru import logger
from manga_ocr import MangaOcr
from PIL import Image

from manga_access.backends.base import OCRBackend
from manga_access.schemas.manga_page import BBox, TextElement

_MODEL_ID = "kha-white/manga-ocr-base"


class MangaOCRBackend(OCRBackend):
    """Backend OCR basé sur manga-ocr, contraint au CPU."""

    def __init__(self) -> None:
        self._model: Any = None
        # Any : MangaOcr n'expose pas de type public réutilisable dans ses
        # annotations (bibliothèque tierce sans stubs de types).

    def load(self) -> None:
        """Charge manga-ocr en forçant explicitement le CPU."""
        start = time.perf_counter()
        self._model = MangaOcr(pretrained_model_name_or_path=_MODEL_ID, force_cpu=True)
        elapsed = time.perf_counter() - start
        logger.info(f"manga-ocr chargé en {elapsed:.2f}s")

    def unload(self) -> None:
        """Décharge le modèle et force la libération immédiate de la RAM."""
        self._model = None
        gc.collect()

    def recognize(self, image: Image.Image, bbox: BBox) -> TextElement:
        """Reconnaît le texte dans `bbox` en recadrant `image` puis en appelant manga-ocr."""
        if self._model is None:
            raise RuntimeError("MangaOCRBackend.load() doit être appelé avant recognize().")

        cropped = image.crop(bbox)
        text = self._model(cropped)

        return TextElement(
            id=f"text-{uuid.uuid4().hex[:8]}",
            # type="dialogue" : placeholder, manga-ocr ne classifie pas
            # dialogue/narration/sfx — à réaffecter dans page_processor.py
            # à partir des infos Magiv2 (is_essential_text, contexte case).
            type="dialogue",
            bbox=bbox,
            text_original=text,
            confidence=1.0,  # manga-ocr n'expose pas de score de confiance réel
        )
