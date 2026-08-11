"""Tests du backend TTS Kokoro (misaki/pyopenjtalk pour le japonais)."""

from __future__ import annotations

import io
import wave

import pytest
from kokoro_onnx import SAMPLE_RATE

from manga_access.backends.kokoro_backend import KokoroBackend


def test_phonemize_japanese_raises_assertion_error_on_problematic_kana() -> None:
    """JAG2P lève AssertionError (pas un tuple) sur 'ヒィッ' — limitation connue de misaki/pyopenjtalk.

    Régression : initialement mal diagnostiqué comme "JAG2P retourne un tuple
    au lieu d'une string" ; la vérification directe montre qu'il n'y a pas de
    valeur de retour anormale, JAG2P lève une exception avant tout `return`
    (misaki/ja.py:283, désaccord entre pyopenjtalk et pron2moras() sur le
    nombre de moras pour ce motif kana).
    """
    backend = KokoroBackend()

    with pytest.raises(AssertionError):
        backend._phonemize_japanese("ヒィッ")


def test_synthesize_falls_back_to_silence_on_problematic_kana() -> None:
    """synthesize() ne plante plus sur 'ヒィッ' : silence renvoyé au lieu de laisser l'AssertionError remonter."""
    backend = KokoroBackend()
    backend._model = object()  # jamais déréférencé : l'échec a lieu dans _phonemize_japanese(), avant self._model.create()

    audio_bytes = backend.synthesize("ヒィッ", voice_id="jf_alpha", lang="ja")

    with wave.open(io.BytesIO(audio_bytes)) as wav_file:
        assert wav_file.getframerate() == SAMPLE_RATE
        assert wav_file.getnframes() == int(SAMPLE_RATE * 0.1)
