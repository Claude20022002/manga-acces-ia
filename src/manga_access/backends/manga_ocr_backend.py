"""Backend OCR manga-ocr (kha-white/manga-ocr-base)."""

from __future__ import annotations

import gc
import re
import time
import uuid
from typing import Any, Literal

from loguru import logger
from manga_ocr import MangaOcr
from PIL import Image

from manga_access.backends.base import OCRBackend
from manga_access.schemas.manga_page import BBox, TextElement

_MODEL_ID = "kha-white/manga-ocr-base"

_KATAKANA_PATTERN = re.compile("[゠-ヿ]")
_JAPANESE_KANA_KANJI_PATTERN = re.compile("[぀-ゟ゠-ヿ一-鿿]")
_SENTENCE_PUNCTUATION = ("。", "、", "？")  # 。、？

_PARTICLES = (
    "は", "が", "を", "に", "で", "と", "も", "の",
    "から", "まで", "より", "へ", "です", "ます", "ました",
    "だった", "ない", "ね", "よ", "わ", "な", "か", "けど", "けれど", "し",
)

_SFX_MAX_LENGTH = 8
_SFX_MIN_KATAKANA_RATIO = 0.7


def _classify_text_type(text: str) -> Literal["dialogue", "sfx"]:
    """Classe un texte OCR en 'dialogue' ou 'sfx' par heuristique texte pure.

    Ne distingue PAS narration/thought (nécessite le contexte panel/bulle
    de Magiv2, non disponible à ce niveau) — tout ce qui n'est pas
    reconnu comme onomatopée reste 'dialogue', comme avant cette heuristique.

    Règle : un texte est classé 'sfx' seulement s'il est court (<= 8
    caractères), sans ponctuation de fin de phrase (。、？), sans aucune
    particule grammaticale japonaise connue, ET majoritairement composé
    de katakana (>= 70% des caractères japonais du texte — tolérance
    conçue pour absorber le bruit OCR observé sur le corpus, ex.
    バタバタッ mal reconnu バタいタッ garde un ratio katakana suffisant).
    """
    stripped = text.strip()
    if not stripped:
        return "dialogue"
    if len(stripped) > _SFX_MAX_LENGTH:
        return "dialogue"
    if any(mark in stripped for mark in _SENTENCE_PUNCTUATION):
        return "dialogue"
    if any(particle in stripped for particle in _PARTICLES):
        return "dialogue"

    japanese_chars = _JAPANESE_KANA_KANJI_PATTERN.findall(stripped)
    if not japanese_chars:
        return "dialogue"

    katakana_ratio = len(_KATAKANA_PATTERN.findall(stripped)) / len(japanese_chars)
    if katakana_ratio >= _SFX_MIN_KATAKANA_RATIO:
        return "sfx"
    return "dialogue"


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
            type=_classify_text_type(text),
            bbox=bbox,
            text_original=text,
            confidence=1.0,  # manga-ocr n'expose pas de score de confiance réel
        )
