"""Tests de l'assemblage audio depuis un NarrativeScript (backend TTS mocké)."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import pytest
from pydub import AudioSegment

from manga_access.backends.base import TTSBackend
from manga_access.pipeline.audio_assembler import (
    _clean_japanese_text,
    _detect_lang,
    _voice_for_lang,
    assemble_audio,
    save_timeline,
)
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
    id_: str, text: str = "dummy", kind: str = "dialogue", page_index: int = 0
) -> NarrativeSegment:
    """Construit un NarrativeSegment minimal pour les tests."""
    return NarrativeSegment(
        id=id_,
        panel_id="panel-0",
        kind=kind,
        voice_id="af_sky",
        text=text,
        page_index=page_index,
    )


def _make_script(n_segments: int) -> NarrativeScript:
    """Construit un NarrativeScript avec `n_segments` segments factices."""
    return NarrativeScript(
        source={"file": "test.jpg", "page_index": 0},
        segments=[_make_segment(f"seg-{i}") for i in range(n_segments)],
    )


def test_clean_japanese_text_normalizes_fullwidth_latin_words() -> None:
    """Texte OCR en fullwidth latin (pleine chasse) -> mot ASCII normal, pas épelé lettre par lettre."""
    assert _clean_japanese_text("Ｗｏｒｄｏｗｓ") == "Wordows"


def test_clean_japanese_text_normalizes_fullwidth_latin_acronym() -> None:
    """Cas rapporté : un acronyme entièrement en fullwidth se normalise en ASCII standard."""
    assert _clean_japanese_text("ＦＩＲＳＴＣＯＮＴＡＣＴ") == "FIRSTCONTACT"


def test_clean_japanese_text_still_normalizes_ellipsis_after_nfkc() -> None:
    """La normalisation NFKC n'empêche pas le nettoyage existant (points de suspension pleine chasse)."""
    assert _clean_japanese_text("．．．") == "、"


def test_clean_japanese_text_still_dedupes_punctuation_after_nfkc() -> None:
    """La normalisation NFKC n'empêche pas la déduplication de ponctuation existante."""
    assert _clean_japanese_text("！！") == "！"
    assert _clean_japanese_text("？？") == "？"


def test_ja_segment_with_real_japanese_text_is_synthesized_normally(tmp_path: Path) -> None:
    """Un segment japonais légitime (kana réels après nettoyage) n'est pas sauté par le garde-fou."""
    output_path = tmp_path / "ja.opus"
    script = NarrativeScript(
        source={"file": "test.jpg", "page_index": 0},
        segments=[_make_segment("seg-1", text="おはよう")],
    )
    backend = _FakeTTSBackend()

    timeline = assemble_audio(script, backend, output_path)

    assert len(timeline.segments) == 1
    assert len(backend.synthesize_calls) == 1


def test_ja_segment_with_no_japanese_chars_after_cleaning_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """lang détecté 'ja' mais texte nettoyé sans caractère japonais -> segment sauté, pas synthétisé.

    _clean_japanese_text() actuel ne supprime jamais un caractère japonais
    (NFKC le recompose au pire) donc ce cas n'est pas atteignable avec
    l'implémentation réelle aujourd'hui — on le simule via monkeypatch pour
    tester le garde-fou lui-même (protection contre un futur changement de
    _clean_japanese_text, ou un cas d'OCR non anticipé).
    """
    monkeypatch.setattr(
        "manga_access.pipeline.audio_assembler._clean_japanese_text",
        lambda text: "Letter FF43",
    )
    output_path = tmp_path / "skip.opus"
    script = NarrativeScript(
        source={"file": "test.jpg", "page_index": 0},
        # texte brut avec un vrai caractère japonais pour déclencher lang="ja" via _detect_lang
        segments=[_make_segment("seg-1", text="レターFF43")],
    )
    backend = _FakeTTSBackend()

    timeline = assemble_audio(script, backend, output_path)

    assert timeline.segments == []
    assert backend.synthesize_calls == []


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
    """Texte latin de kind='scene_description' -> 'fr-fr'."""
    assert _detect_lang("2 personnages détectés.", kind="scene_description") == "fr-fr"


def test_detect_lang_default_english() -> None:
    """Texte latin d'un autre kind -> 'en-us' par défaut."""
    assert _detect_lang("Hello there!", kind="dialogue") == "en-us"


def test_detect_lang_japanese_wins_over_scene_description() -> None:
    """Priorité : japonais détecté même si kind='scene_description' -> 'ja'."""
    assert _detect_lang("こんにちは", kind="scene_description") == "ja"


def test_detect_lang_narration_french() -> None:
    """Texte latin de kind='narration' -> 'fr-fr' (préfixes/suffixes insérés par enrich_script())."""
    assert _detect_lang("Elle dit :", kind="narration") == "fr-fr"


def test_detect_lang_japanese_wins_over_narration() -> None:
    """Priorité : japonais détecté même si kind='narration' -> 'ja'."""
    assert _detect_lang("こんにちは", kind="narration") == "ja"


def test_detect_lang_default_lang_override_french() -> None:
    """default_lang="fr-fr" : un dialogue sans japonais (ex. traduit) est détecté "fr-fr", pas "en-us".

    Bug corrigé (Phase 3, traduction) : avant ce fix, un dialogue traduit en
    français retombait sur "en-us" en dur, donnant à Kokoro le mauvais
    phonémiseur espeak pour du texte français.
    """
    assert _detect_lang("Bonjour", kind="dialogue", default_lang="fr-fr") == "fr-fr"


def test_detect_lang_default_lang_defaults_to_en_us() -> None:
    """Sans default_lang explicite, comportement historique préservé ("en-us")."""
    assert _detect_lang("Hello there!", kind="dialogue") == "en-us"


def test_detect_lang_japanese_wins_over_default_lang_override() -> None:
    """Priorité : japonais détecté même avec default_lang="fr-fr" -> "ja" quand même."""
    assert _detect_lang("こんにちは", kind="dialogue", default_lang="fr-fr") == "ja"


def test_detect_lang_scene_description_stays_french_regardless_of_default_lang() -> None:
    """scene_description reste "fr-fr" même avec default_lang="en-us" (règle indépendante)."""
    assert (
        _detect_lang("2 personnages détectés.", kind="scene_description", default_lang="en-us")
        == "fr-fr"
    )


def test_voice_for_lang_japanese_keeps_voice_id() -> None:
    """lang='ja' -> voice_id d'entrée inchangé (déjà une voix japonaise assignée en amont)."""
    assert _voice_for_lang("jf_alpha", "ja") == "jf_alpha"


def test_voice_for_lang_french_overrides_voice_id() -> None:
    """lang='fr-fr' -> 'ff_siwis', quel que soit le voice_id d'entrée."""
    assert _voice_for_lang("af_bella", "fr-fr") == "ff_siwis"


def test_voice_for_lang_default_keeps_voice_id() -> None:
    """lang='en-us' -> voice_id d'entrée inchangé."""
    assert _voice_for_lang("af_sky", "en-us") == "af_sky"


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


def test_assemble_audio_translated_dialogue_detected_as_narration_lang(tmp_path: Path) -> None:
    """Un dialogue déjà traduit (sans japonais) est détecté dans narration_lang, pas "en-us" en dur.

    Simule le résultat de translate_dialogues() (Phase 3) : le texte du
    segment est déjà en français au moment où assemble_audio() le reçoit
    (la traduction se fait en amont, dans demo.py, pas ici).
    """
    script = NarrativeScript(
        source={"file": "test.jpg", "page_index": 0},
        segments=[_make_segment("seg-1", text="Bonjour", kind="dialogue")],
    )
    backend = _FakeTTSBackend()

    assemble_audio(script, backend, tmp_path / "out.opus", narration_lang="fr")

    assert backend.synthesize_calls[0][2] == "fr-fr"


def test_assemble_audio_returns_timeline_with_correct_boundaries(tmp_path: Path) -> None:
    """3 segments dialogue -> bornes (0,100), (400,500), (800,900) (100ms audio + 300ms silence)."""
    output_path = tmp_path / "multi.opus"
    timeline = assemble_audio(_make_script(3), _FakeTTSBackend(), output_path)

    assert [(s.start_ms, s.end_ms) for s in timeline.segments] == [
        (0, 100), (400, 500), (800, 900),
    ]
    assert [s.id for s in timeline.segments] == ["seg-0", "seg-1", "seg-2"]
    assert all(s.kind == "dialogue" and s.text == "dummy" for s in timeline.segments)


def test_timeline_segment_page_index_propagated_from_narrative_segment(tmp_path: Path) -> None:
    """TimelineSegment.page_index reprend NarrativeSegment.page_index de son segment d'origine."""
    output_path = tmp_path / "paged.opus"
    script = NarrativeScript(
        source={"file": "test.jpg", "page_index": 0},
        segments=[
            _make_segment("seg-0", page_index=0),
            _make_segment("seg-1", page_index=2),
        ],
    )

    timeline = assemble_audio(script, _FakeTTSBackend(), output_path)

    assert [s.page_index for s in timeline.segments] == [0, 2]


def test_timeline_excludes_only_empty_segments(tmp_path: Path) -> None:
    """Avec include_scene_descriptions=True, seul le texte vide n'a pas d'intervalle audio."""
    script = NarrativeScript(
        source={"file": "test.jpg", "page_index": 0},
        segments=[
            _make_segment("seg-desc", text="4 personnages détectés.", kind="scene_description"),
            _make_segment("seg-empty", text="   ", kind="dialogue"),
            _make_segment("seg-real", text="Bonjour", kind="dialogue"),
        ],
    )
    timeline = assemble_audio(
        script, _FakeTTSBackend(), tmp_path / "out.opus", include_scene_descriptions=True
    )

    assert [s.id for s in timeline.segments] == ["seg-desc", "seg-real"]


def test_timeline_no_leading_silence_after_skipped_segment(tmp_path: Path) -> None:
    """Un premier segment ignoré (texte vide) n'insère pas 300ms de silence avant le suivant."""
    script = NarrativeScript(
        source={"file": "test.jpg", "page_index": 0},
        segments=[
            _make_segment("seg-empty", text="   ", kind="dialogue"),
            _make_segment("seg-real", text="Bonjour", kind="dialogue"),
        ],
    )
    timeline = assemble_audio(script, _FakeTTSBackend(), tmp_path / "out.opus")

    assert timeline.segments[0].id == "seg-real"
    assert timeline.segments[0].start_ms == 0
    assert timeline.segments[0].end_ms == 100


def test_include_scene_descriptions_default_false_skips_synthesis(tmp_path: Path) -> None:
    """Par défaut (include_scene_descriptions non précisé), scene_description n'est plus synthétisé."""
    script = NarrativeScript(
        source={"file": "test.jpg", "page_index": 0},
        segments=[_make_segment("seg-desc", text="Une rue calme.", kind="scene_description")],
    )
    backend = _FakeTTSBackend()
    timeline = assemble_audio(script, backend, tmp_path / "out.opus")

    assert backend.synthesize_calls == []
    assert timeline.segments == []


def test_include_scene_descriptions_true_still_synthesizes(tmp_path: Path) -> None:
    """include_scene_descriptions=True (opt-in) restaure la synthèse des scene_description."""
    script = NarrativeScript(
        source={"file": "test.jpg", "page_index": 0},
        segments=[_make_segment("seg-desc", text="Une rue calme.", kind="scene_description")],
    )
    backend = _FakeTTSBackend()
    timeline = assemble_audio(
        script, backend, tmp_path / "out.opus", include_scene_descriptions=True
    )

    assert len(backend.synthesize_calls) == 1
    assert [s.id for s in timeline.segments] == ["seg-desc"]


def test_include_scene_descriptions_false_excludes_from_timeline_and_synthesis(
    tmp_path: Path,
) -> None:
    """include_scene_descriptions=False (explicite, redondant avec le défaut) : même comportement."""
    script = NarrativeScript(
        source={"file": "test.jpg", "page_index": 0},
        segments=[
            _make_segment("seg-desc", text="Une rue calme.", kind="scene_description"),
            _make_segment("seg-real", text="Bonjour", kind="dialogue"),
        ],
    )
    backend = _FakeTTSBackend()
    timeline = assemble_audio(
        script, backend, tmp_path / "out.opus", include_scene_descriptions=False
    )

    assert [s.id for s in timeline.segments] == ["seg-real"]
    assert all(call[0] != "Une rue calme." for call in backend.synthesize_calls)


def test_scene_description_excluded_from_audio_but_kept_for_transcript(tmp_path: Path) -> None:
    """Exclu de l'audio par défaut, un scene_description reste dans script.segments pour le transcript."""
    script = NarrativeScript(
        source={"file": "test.jpg", "page_index": 0},
        segments=[_make_segment("seg-desc", text="Deux personnages détectés.", kind="scene_description")],
    )
    backend = _FakeTTSBackend()
    assemble_audio(script, backend, tmp_path / "out.opus")

    assert backend.synthesize_calls == []  # rien synthétisé
    assert script.segments[0].text == "Deux personnages détectés."  # toujours là pour save_transcript()


def test_narration_lang_none_leaves_text_unchanged(tmp_path: Path) -> None:
    """narration_lang=None (défaut) : aucun enrichissement, texte synthétisé tel quel."""
    script = NarrativeScript(
        source={"file": "test.jpg", "page_index": 0},
        segments=[_make_segment("seg-1", text="Bonjour", kind="dialogue")],
    )
    backend = _FakeTTSBackend()
    assemble_audio(script, backend, tmp_path / "out.opus")

    assert backend.synthesize_calls[0][0] == "Bonjour"


def test_narration_lang_enriches_text_before_synthesis(tmp_path: Path) -> None:
    """narration_lang="fr" : un segment narration (voix narrateur) est synthétisé avant le dialogue.

    Un sfx compagnon (même panel) est nécessaire : narration_builder.py
    n'annonce plus un dialogue qui clôt son panel (rien de vocal après lui).
    """
    script = NarrativeScript(
        source={"file": "test.jpg", "page_index": 0},
        segments=[
            _make_segment("seg-1", text="Bonjour", kind="dialogue"),
            _make_segment("seg-sfx", text="[ドン]", kind="sfx"),
        ],
    )
    backend = _FakeTTSBackend()
    timeline = assemble_audio(script, backend, tmp_path / "out.opus", narration_lang="fr")

    assert backend.synthesize_calls[0] == ("Elle dit :", "ff_siwis", "fr-fr")
    assert backend.synthesize_calls[1][0] == "Bonjour"
    assert [s.kind for s in timeline.segments[:2]] == ["narration", "dialogue"]
    assert [s.text for s in timeline.segments[:2]] == ["Elle dit :", "Bonjour"]


def test_narration_lang_mutates_caller_script_in_place(tmp_path: Path) -> None:
    """enrich_script() étant appelé en place, le script de l'appelant a aussi le segment narration."""
    script = NarrativeScript(
        source={"file": "test.jpg", "page_index": 0},
        segments=[
            _make_segment("seg-1", text="Bonjour", kind="dialogue"),
            _make_segment("seg-sfx", text="[ドン]", kind="sfx"),
        ],
    )
    assemble_audio(script, _FakeTTSBackend(), tmp_path / "out.opus", narration_lang="fr")

    assert [s.kind for s in script.segments[:2]] == ["narration", "dialogue"]
    assert script.segments[0].text == "Elle dit :"
    assert script.segments[1].text == "Bonjour"


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
