"""Backend TTS Kokoro (kokoro-onnx, poids ONNX quantisés int8)."""

from __future__ import annotations

import gc
import io
import os
import time

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro
from loguru import logger

from manga_access.backends.base import TTSBackend

_DEFAULT_MODEL_PATH = "models/kokoro/kokoro-v1.0.int8.onnx"
_DEFAULT_VOICES_PATH = "models/kokoro/voices-v1.0.bin"


class KokoroBackend(TTSBackend):
    """Backend TTS basé sur kokoro-onnx (ONNX Runtime, CPU par défaut)."""

    def __init__(self, model_path: str | None = None, voices_path: str | None = None) -> None:
        self._model_path = model_path or os.environ.get("KOKORO_MODEL_PATH", _DEFAULT_MODEL_PATH)
        self._voices_path = voices_path or os.environ.get("KOKORO_VOICES_PATH", _DEFAULT_VOICES_PATH)
        self._model: Kokoro | None = None

    def load(self) -> None:
        """Charge kokoro-onnx depuis les poids locaux (cf. README pour le téléchargement)."""
        start = time.perf_counter()
        self._model = Kokoro(self._model_path, self._voices_path)
        elapsed = time.perf_counter() - start
        logger.info(f"Kokoro chargé en {elapsed:.2f}s")

    def unload(self) -> None:
        """Décharge le modèle et force la libération immédiate de la RAM."""
        self._model = None
        gc.collect()

    def synthesize(self, text: str, voice_id: str, lang: str = "en-us") -> bytes:
        """Synthétise `text` avec `voice_id`/`lang` et encode le résultat en WAV.

        Le WAV (plutôt que le PCM brut) évite de transporter le sample rate
        hors bande — pydub pourra charger chaque segment directement lors de
        l'assemblage par page (Tâche 2, Phase 3).
        """
        if self._model is None:
            raise RuntimeError("KokoroBackend.load() doit être appelé avant synthesize().")

        samples, sample_rate = self._model.create(text, voice=voice_id, lang=lang)
        if len(samples) == 0:
            logger.warning(f"Kokoro : texte non phonémisable, silence retourné : {text!r}")
            samples = np.zeros(int(sample_rate * 0.1), dtype=np.float32)

        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format="WAV")
        return buffer.getvalue()
