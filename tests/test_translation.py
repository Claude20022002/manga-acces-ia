"""Tests de translate_dialogues (traduction des dialogues/pensées japonais, Phase 3)."""

from __future__ import annotations

from manga_access.backends.base import TranslationBackend
from manga_access.pipeline.translation import translate_dialogues
from manga_access.schemas.narrative_script import NarrativeScript, NarrativeSegment


class _FakeTranslationBackend(TranslationBackend):
    """Backend de traduction factice : retourne une traduction fixe, journalise les appels."""

    def __init__(self, translation: str = "translated") -> None:
        self._translation = translation
        self.load_calls = 0
        self.unload_calls = 0
        self.translate_calls: list[tuple[str, str]] = []

    def load(self) -> None:
        self.load_calls += 1

    def unload(self) -> None:
        self.unload_calls += 1

    def translate(self, text: str, target_lang: str) -> str:
        self.translate_calls.append((text, target_lang))
        return self._translation


def _make_segment(id_: str, kind: str, text: str) -> NarrativeSegment:
    """Construit un NarrativeSegment minimal pour les tests."""
    return NarrativeSegment(
        id=id_,
        panel_id="panel-0",
        kind=kind,  # type: ignore[arg-type]
        voice_id="jf_alpha",
        text=text,
    )


def _make_script(segments: list[NarrativeSegment]) -> NarrativeScript:
    """Construit un NarrativeScript minimal pour les tests."""
    return NarrativeScript(source={"file": "test.jpg"}, segments=segments)


def test_dialogue_with_japanese_is_translated() -> None:
    """Un dialogue japonais est envoyé à backend.translate() et son texte remplacé."""
    segment = _make_segment("seg-1", "dialogue", "おはよう")
    script = _make_script([segment])
    backend = _FakeTranslationBackend(translation="Good morning")

    translate_dialogues(script, backend, "en")

    assert segment.text == "Good morning"
    assert backend.translate_calls == [("おはよう", "en")]


def test_thought_with_japanese_is_translated() -> None:
    """kind='thought' est traité comme 'dialogue' (aussi traduit)."""
    segment = _make_segment("seg-1", "thought", "本当にいいのかな…")
    script = _make_script([segment])
    backend = _FakeTranslationBackend(translation="Is this really okay...")

    translate_dialogues(script, backend, "en")

    assert segment.text == "Is this really okay..."
    assert len(backend.translate_calls) == 1


def test_dialogue_without_japanese_is_not_translated() -> None:
    """Un dialogue déjà sans caractère japonais (ex. déjà traduit) n'est pas renvoyé au backend."""
    segment = _make_segment("seg-1", "dialogue", "Hello there")
    script = _make_script([segment])
    backend = _FakeTranslationBackend()

    translate_dialogues(script, backend, "en")

    assert segment.text == "Hello there"
    assert backend.translate_calls == []


def test_sfx_is_never_translated() -> None:
    """Un sfx japonais (onomatopée) n'est jamais envoyé au backend, même avec du japonais."""
    segment = _make_segment("seg-1", "sfx", "ドン")
    script = _make_script([segment])
    backend = _FakeTranslationBackend()

    translate_dialogues(script, backend, "en")

    assert segment.text == "ドン"
    assert backend.translate_calls == []


def test_narration_is_never_translated() -> None:
    """Une narration (déjà générée dans la langue cible par narration_builder.py) n'est pas touchée."""
    segment = _make_segment("seg-1", "narration", "声が言った：")
    script = _make_script([segment])
    backend = _FakeTranslationBackend()

    translate_dialogues(script, backend, "en")

    assert segment.text == "声が言った："
    assert backend.translate_calls == []


def test_scene_description_is_never_translated() -> None:
    """scene_description (toujours en français) n'est jamais envoyé au backend."""
    segment = _make_segment("seg-1", "scene_description", "日本語のテスト")
    script = _make_script([segment])
    backend = _FakeTranslationBackend()

    translate_dialogues(script, backend, "en")

    assert segment.text == "日本語のテスト"
    assert backend.translate_calls == []


def test_translation_still_containing_japanese_logs_but_keeps_result() -> None:
    """Si le backend renvoie un texte encore japonais (échec dégradé), il est quand même assigné."""
    segment = _make_segment("seg-1", "dialogue", "おはよう")
    script = _make_script([segment])
    backend = _FakeTranslationBackend(translation="おはよう")  # simule un échec de backend.translate()

    translate_dialogues(script, backend, "en")

    assert segment.text == "おはよう"


def test_multiple_segments_only_eligible_ones_translated() -> None:
    """Dans un script mixte, seuls les dialogues/thoughts japonais sont envoyés au backend."""
    dialogue_ja = _make_segment("seg-1", "dialogue", "こんにちは")
    dialogue_en = _make_segment("seg-2", "dialogue", "Already English")
    sfx = _make_segment("seg-3", "sfx", "ガチャ")
    script = _make_script([dialogue_ja, dialogue_en, sfx])
    backend = _FakeTranslationBackend(translation="Hello")

    translate_dialogues(script, backend, "en")

    assert dialogue_ja.text == "Hello"
    assert dialogue_en.text == "Already English"
    assert sfx.text == "ガチャ"
    assert backend.translate_calls == [("こんにちは", "en")]


def test_returns_same_script_instance() -> None:
    """translate_dialogues() retourne la même instance NarrativeScript (mutation en place)."""
    script = _make_script([_make_segment("seg-1", "dialogue", "おはよう")])
    backend = _FakeTranslationBackend()

    result = translate_dialogues(script, backend, "en")

    assert result is script
