"""Tests de l'assemblage audio depuis un NarrativeScript (backend TTS mocké)."""

from __future__ import annotations

import io
import wave
from pathlib import Path

from pydub import AudioSegment

from manga_access.backends.base import TTSBackend
from manga_access.pipeline.audio_assembler import _detect_lang, assemble_audio, save_timeline
from manga_access.schemas.narrative_script import NarrativeScript, NarrativeSegment
from manga_access.schemas.timeline import Timeline, TimelineSegment


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
        self.synthesize_calls: list[tuple[str, str, str]] = []

    def load(self) -> None:
        self.load_calls += 1

    def unload(self) -> None:
        self.unload_calls += 1

    def synthesize(self, text: str, voice_id: str, lang: str = "en-us") -> bytes:
        self.synthesize_calls.append((text, voice_id, lang))
        return _make_silent_wav_bytes()


def _make_segment(
    id_: str, text: str = "dummy", kind: str = "dialogue"
) -> NarrativeSegment:
    """Construit un NarrativeSegment minimal pour les tests."""
    return NarrativeSegment(
        id=id_,
        panel_id="panel-0",
        kind=kind,
        voice_id="af_sky",
        text=text,
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


def test_detect_lang_japanese_hiragana() -> None:
    """Texte contenant du hiragana -> 'ja'."""
    assert _detect_lang("こんにちは", kind="dialogue") == "ja"


def test_detect_lang_japanese_katakana() -> None:
    """Texte contenant du katakana -> 'ja'."""
    assert _detect_lang("カタカナ", kind="dialogue") == "ja"


def test_detect_lang_japanese_kanji() -> None:
    """Texte contenant du kanji -> 'ja'."""
    assert _detect_lang("東京", kind="dialogue") == "ja"


def test_detect_lang_scene_description_french() -> None:
    """Texte latin de kind='scene_description' -> 'fr'."""
    assert _detect_lang("2 personnages détectés.", kind="scene_description") == "fr"


def test_detect_lang_default_english() -> None:
    """Texte latin d'un autre kind -> 'en-us' par défaut."""
    assert _detect_lang("Hello there!", kind="dialogue") == "en-us"


def test_detect_lang_japanese_wins_over_scene_description() -> None:
    """Priorité : japonais détecté même si kind='scene_description' -> 'ja'."""
    assert _detect_lang("こんにちは", kind="scene_description") == "ja"


def test_assemble_audio_passes_detected_lang(tmp_path: Path) -> None:
    """assemble_audio() transmet le lang détecté à synthesize() par segment."""
    script = NarrativeScript(
        source={"file": "test.jpg", "page_index": 0},
        segments=[
            _make_segment("seg-ja", text="こんにちは", kind="dialogue"),
            _make_segment("seg-en", text="Hello there", kind="dialogue"),
        ],
    )
    backend = _FakeTTSBackend()
    assemble_audio(script, backend, tmp_path / "out.opus")

    langs = [call[2] for call in backend.synthesize_calls]
    assert langs == ["ja", "en-us"]


def test_assemble_audio_returns_timeline_with_correct_boundaries(tmp_path: Path) -> None:
    """3 segments dialogue -> bornes (0,100), (400,500), (800,900) (100ms audio + 300ms silence)."""
    output_path = tmp_path / "multi.opus"
    timeline = assemble_audio(_make_script(3), _FakeTTSBackend(), output_path)

    assert [(s.start_ms, s.end_ms) for s in timeline.segments] == [
        (0, 100), (400, 500), (800, 900),
    ]
    assert [s.id for s in timeline.segments] == ["seg-0", "seg-1", "seg-2"]
    assert all(s.kind == "dialogue" and s.text == "dummy" for s in timeline.segments)


def test_timeline_excludes_scene_description_and_empty_segments(tmp_path: Path) -> None:
    """scene_description et texte vide n'ont pas d'intervalle audio -> absents de la timeline."""
    script = NarrativeScript(
        source={"file": "test.jpg", "page_index": 0},
        segments=[
            _make_segment("seg-desc", text="4 personnages détectés.", kind="scene_description"),
            _make_segment("seg-empty", text="   ", kind="dialogue"),
            _make_segment("seg-real", text="Bonjour", kind="dialogue"),
        ],
    )
    timeline = assemble_audio(script, _FakeTTSBackend(), tmp_path / "out.opus")

    assert [s.id for s in timeline.segments] == ["seg-real"]


def test_timeline_no_leading_silence_after_skipped_segment(tmp_path: Path) -> None:
    """Un segment scene_description initial (ignoré) n'insère pas 300ms de silence avant le premier segment réel."""
    script = NarrativeScript(
        source={"file": "test.jpg", "page_index": 0},
        segments=[
            _make_segment("seg-desc", text="Description.", kind="scene_description"),
            _make_segment("seg-real", text="Bonjour", kind="dialogue"),
        ],
    )
    timeline = assemble_audio(script, _FakeTTSBackend(), tmp_path / "out.opus")

    assert timeline.segments[0].start_ms == 0
    assert timeline.segments[0].end_ms == 100


def test_save_timeline_writes_json(tmp_path: Path) -> None:
    """save_timeline() écrit un JSON relisible via Timeline.from_json()."""
    timeline = Timeline(
        source={"file": "test.jpg", "page_index": 0},
        segments=[
            TimelineSegment(id="seg-0", kind="dialogue", text="Bonjour", start_ms=0, end_ms=100),
        ],
    )
    output_path = tmp_path / "out.timeline.json"
    save_timeline(timeline, output_path)

    assert output_path.exists()
    restored = Timeline.from_json(output_path.read_text(encoding="utf-8"))
    assert restored == timeline


def test_timeline_roundtrip_json() -> None:
    """to_json() puis from_json() doit reproduire une Timeline identique."""
    original = Timeline(
        source={"file": "test.jpg", "page_index": 0},
        segments=[
            TimelineSegment(id="seg-0", kind="dialogue", text="Bonjour", start_ms=0, end_ms=100),
            TimelineSegment(id="seg-1", kind="sfx", text="[ドン]", start_ms=400, end_ms=500),
        ],
    )

    serialized = original.to_json()
    restored = Timeline.from_json(serialized)

    assert restored == original
