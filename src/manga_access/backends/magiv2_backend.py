"""Backend de structuration Magiv2 (détection cases/bulles/personnages)."""

from __future__ import annotations

import gc
import time
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
