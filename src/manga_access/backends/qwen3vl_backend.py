"""Backend de description de scène Qwen3-VL (GGUF Q4_K_M via llama-cpp-python).

Validé par benchmarks/qwen_vl_smoketest.py, cf.
docs/sessions/2026-08-10-qwen-smoketest.md : chargement 2.12s, RAM pic
7110 Mo, inférence 111.62s/image — batch offline et chargement séquentiel
obligatoires (contraintes déjà en place pour Magiv2/manga-ocr/Kokoro).
"""

from __future__ import annotations

import base64
import gc
import io
import os
import time

from llama_cpp import Llama
from llama_cpp.llama_chat_format import MTMDChatHandler
from loguru import logger
from PIL import Image

from manga_access.backends.base import SceneDescriptionBackend
from manga_access.schemas.manga_page import BBox

_DEFAULT_MODEL_PATH = "models/qwen3vl/Qwen3VL-4B-Instruct-Q4_K_M.gguf"
_DEFAULT_MMPROJ_PATH = "models/qwen3vl/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"

_DEFAULT_PROMPT = (
    "Décris cette image de manga en une ou deux phrases, en français, pour "
    "une personne aveugle : action, ambiance, éléments visuels importants. "
    "Ne décris pas le texte des bulles, seulement l'image."
)

_N_CTX = 4096
_N_THREADS = 6
_MAX_TOKENS = 256


def _image_to_data_uri(image: Image.Image) -> str:
    """Encode une image PIL en data URI base64 JPEG (format attendu par llama-cpp-python)."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class QwenVLBackend(SceneDescriptionBackend):
    """Backend de description de scène basé sur Qwen3-VL-4B-Instruct (GGUF Q4_K_M)."""

    def __init__(self, model_path: str | None = None, mmproj_path: str | None = None) -> None:
        self._model_path = model_path or os.environ.get("QWEN3VL_MODEL_PATH", _DEFAULT_MODEL_PATH)
        self._mmproj_path = mmproj_path or os.environ.get(
            "QWEN3VL_MMPROJ_PATH", _DEFAULT_MMPROJ_PATH
        )
        self._llm: Llama | None = None
        self._chat_handler: MTMDChatHandler | None = None

    def load(self) -> None:
        """Charge Qwen3-VL et son projecteur vision, forcés sur CPU (n_threads=6)."""
        start = time.perf_counter()
        self._chat_handler = MTMDChatHandler(clip_model_path=self._mmproj_path)
        self._llm = Llama(
            model_path=self._model_path,
            chat_handler=self._chat_handler,
            n_ctx=_N_CTX,
            n_threads=_N_THREADS,
            verbose=False,
        )
        elapsed = time.perf_counter() - start
        logger.info(f"Qwen3-VL chargé en {elapsed:.2f}s")

    def unload(self) -> None:
        """Décharge le modèle et son projecteur, force la libération immédiate de la RAM."""
        self._llm = None
        self._chat_handler = None
        gc.collect()

    def describe(self, image: Image.Image, bbox: BBox) -> str | None:
        """Décrit la scène contenue dans `bbox` de `image` via Qwen3-VL.

        Recadre l'image sur `bbox` avant inférence (cohérent avec
        OCRBackend.recognize). Retourne None si la réponse est vide après
        nettoyage.
        """
        if self._llm is None:
            raise RuntimeError("QwenVLBackend.load() doit être appelé avant describe().")

        cropped = image.crop(bbox)
        data_uri = _image_to_data_uri(cropped)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _DEFAULT_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]
        response = self._llm.create_chat_completion(messages=messages, max_tokens=_MAX_TOKENS)
        text = response["choices"][0]["message"]["content"].strip()
        return text or None
