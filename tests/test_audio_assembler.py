"""Tests de l'assemblage audio depuis un NarrativeScript (backend TTS mocké)."""

from __future__ import annotations

import io
import wave
from pathlib import Path

from pydub import AudioSegment

from manga_access.backends.base import TTSBackend
from manga_access.pipeline.audio_assembler import assemble_audio
from manga_access.schemas.narrative_script import NarrativeScript, NarrativeSegment


def _make_silent_wav_bytes(duration_ms: int = 100, sample_rate: int = 24000) -> bytes:
    """Construit un WAV PCM 16 bits mono valide, silencieux, de `duration_ms`."""
    n_frames = int(sample_rate * duration_ms / 1000)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * n_frames)
    return buffer.getvalue()


class _FakeTTSBackend(TTSBackend):
    """Backend TTS factice : compte load/unload, retourne un WAV silencieux fixe."""

    def __init__(self) -> None:
        self.load_calls = 0
        self.unload_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def unload(self) -> None:
        self.unload_calls += 1

    def synthesize(self, text: str, voice_id: str, lang: str = "en-us") -> bytes:
        return _make_silent_wav_bytes()


def _make_segment(id_: str) -> NarrativeSegment:
    """Construit un NarrativeSegment minimal pour les tests."""
    return NarrativeSegment(
        id=id_,
        panel_id="panel-0",
        kind="dialogue",
        voice_id="af_sky",
        text="dummy",
    )


def _make_script(n_segments: int) -> NarrativeScript:
    """Construit un NarrativeScript avec `n_segments` segments factices."""
    return NarrativeScript(
        source={"file": "test.jpg", "page_index": 0},
        segments=[_make_segment(f"seg-{i}") for i in range(n_segments)],
    )


def test_empty_script(tmp_path: Path) -> None:
    """Un script sans segments produit tout de même un fichier de sortie."""
    output_path = tmp_path / "empty.opus"
    assemble_audio(_make_script(0), _FakeTTSBackend(), output_path)

    assert output_path.exists()
    # Pas de segment synthétisé -> flux quasi vide, taille très inférieure à
    # un export contenant un vrai segment audio (cf. test_single_segment).
    assert output_path.stat().st_size < 1000


def test_single_segment(tmp_path: Path) -> None:
    """Un script à un segment produit un fichier audio valide de la bonne durée."""
    output_path = tmp_path / "single.opus"
    assemble_audio(_make_script(1), _FakeTTSBackend(), output_path)

    assert output_path.exists()
    audio = AudioSegment.from_file(output_path)
    assert len(audio) == 100


def test_multiple_segments(tmp_path: Path) -> None:
    """3 segments -> durée = 3x100ms de contenu + 2x300ms de silence entre segments."""
    output_path = tmp_path / "multi.opus"
    assemble_audio(_make_script(3), _FakeTTSBackend(), output_path)

    audio = AudioSegment.from_file(output_path)
    assert len(audio) == 3 * 100 + 2 * 300


def test_output_directory_created(tmp_path: Path) -> None:
    """Le dossier parent de output_path est créé automatiquement s'il n'existe pas."""
    output_path = tmp_path / "nested" / "dir" / "out.opus"
    assert not output_path.parent.exists()

    assemble_audio(_make_script(1), _FakeTTSBackend(), output_path)

    assert output_path.exists()


def test_load_unload_called(tmp_path: Path) -> None:
    """load() et unload() sont appelés exactement une fois chacun."""
    backend = _FakeTTSBackend()
    assemble_audio(_make_script(2), backend, tmp_path / "out.opus")

    assert backend.load_calls == 1
    assert backend.unload_calls == 1
