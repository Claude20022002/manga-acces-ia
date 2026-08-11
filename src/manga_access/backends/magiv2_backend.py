"""Backend de structuration Magiv2 (détection cases/bulles/personnages)."""

from __future__ import annotations

import gc
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image
from transformers import AutoModel

from manga_access.backends.base import StructureBackend

_MODEL_ID = "ragavsachdeva/magiv2"


class Magiv2Backend(StructureBackend):
    """Backend de structuration basé sur Magiv2 (ragavsachdeva/magiv2, licence NC recherche)."""

    def __init__(self) -> None:
        self._model: Any = None
        # Any : Magiv2Model est défini via trust_remote_code=True (code distant
        # HuggingFace, non importable statiquement pour un typage précis).

    def load(self) -> None:
        """Charge Magiv2 depuis HuggingFace et le force explicitement sur CPU."""
        start = time.perf_counter()
        self._model = AutoModel.from_pretrained(
            _MODEL_ID, trust_remote_code=True
        ).to("cpu").eval()
        elapsed = time.perf_counter() - start
        logger.info(f"Magiv2 chargé en {elapsed:.2f}s")

    def unload(self) -> None:
        """Décharge le modèle et force la libération immédiate de la RAM."""
        self._model = None
        gc.collect()

    def detect(self, image: np.ndarray | Image.Image) -> dict[str, Any]:
        """Détecte panels/bulles/personnages/queues sur `image` via Magiv2.

        # Any : cf. StructureBackend.detect, format brut de
        # predict_detections_and_associations (Magiv2).
        """
        if self._model is None:
            raise RuntimeError("Magiv2Backend.load() doit être appelé avant detect().")

        image_np = np.array(image) if isinstance(image, Image.Image) else image

        results = self._model.predict_detections_and_associations(
            [image_np], move_to_device_fn=None
        )
        return results[0]

    def detect_chapter(
        self,
        images: Sequence[np.ndarray | Image.Image],
        character_bank: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Détecte panels/bulles/personnages sur tout un chapitre via Magiv2, avec noms optionnels.

        `character_bank` = {"images": [...], "names": [...]} de portraits de
        référence (mêmes conventions que `detect` pour le format image).
        L'OCR intégré de Magiv2 est désactivé (do_ocr=False) : le projet
        utilise manga-ocr comme seule source d'OCR (stack décidée).
        """
        if self._model is None:
            raise RuntimeError("Magiv2Backend.load() doit être appelé avant detect_chapter().")

        images_np = [np.array(img) if isinstance(img, Image.Image) else img for img in images]
        bank = character_bank if character_bank is not None else {"images": [], "names": []}

        return self._model.do_chapter_wide_prediction(images_np, bank, do_ocr=False)
