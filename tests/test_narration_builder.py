"""Tests de l'enrichissement narratif du NarrativeScript (préfixes contextuels)."""

from __future__ import annotations

import pytest

from manga_access.pipeline.narration_builder import (
    _SFX_PREFIX_VARIATIONS_FR,
    enrich_script,
)
from manga_access.pipeline.narrative_builder import VOICE_NARRATOR, VOICE_UNKNOWN
from manga_access.schemas.narrative_script import NarrativeScript, NarrativeSegment


def _make_segment(
    id_: str, kind: str, text: str, voice_id: str = VOICE_NARRATOR
) -> NarrativeSegment:
    """Construit un NarrativeSegment minimal pour les tests."""
    return NarrativeSegment(
        id=id_,
        panel_id="panel-0",
        kind=kind,  # type: ignore[arg-type]
        voice_id=voice_id,
        text=text,
    )


def _make_script(segments: list[NarrativeSegment]) -> NarrativeScript:
    """Construit un NarrativeScript minimal pour les tests."""
    return NarrativeScript(source={"file": "test.jpg"}, segments=segments)


def test_dialogue_known_speaker_female_fr() -> None:
    """Un dialogue connu avec une voix jf_* devient 'Elle dit : {texte}' en fr."""
    segment = _make_segment("seg-1", "dialogue", "おはよう", voice_id="jf_alpha")
    script = _make_script([segment])

    enrich_script(script, lang="fr")

    assert script.segments[0].text == "Elle dit : おはよう"


def test_dialogue_known_speaker_male_fr() -> None:
    """Un dialogue connu avec une voix jm_* devient 'Il dit : {texte}' en fr."""
    segment = _make_segment("seg-1", "dialogue", "おはよう", voice_id="jm_kumo")
    script = _make_script([segment])

    enrich_script(script, lang="fr")

    assert script.segments[0].text == "Il dit : おはよう"


def test_dialogue_known_speaker_unknown_gender_fr() -> None:
    """Une voix qui ne suit pas la convention f/m devient 'Le personnage dit :' en fr."""
    segment = _make_segment("seg-1", "dialogue", "おはよう", voice_id="voice-custom")
    script = _make_script([segment])

    enrich_script(script, lang="fr")

    assert script.segments[0].text == "Le personnage dit : おはよう"


def test_dialogue_known_speaker_ja() -> None:
    """Un dialogue connu en ja devient '{texte}、と言った', sans distinction de genre."""
    segment = _make_segment("seg-1", "dialogue", "おはよう", voice_id="jf_alpha")
    script = _make_script([segment])

    enrich_script(script, lang="ja")

    assert script.segments[0].text == "おはよう、と言った"


def test_dialogue_unknown_speaker_fr() -> None:
    """Un dialogue avec VOICE_UNKNOWN devient 'Une voix dit : {texte}' en fr."""
    segment = _make_segment("seg-1", "dialogue", "おはよう", voice_id=VOICE_UNKNOWN)
    script = _make_script([segment])

    enrich_script(script, lang="fr")

    assert script.segments[0].text == "Une voix dit : おはよう"


def test_dialogue_unknown_speaker_ja() -> None:
    """Un dialogue avec VOICE_UNKNOWN devient '声が言った：{texte}' en ja."""
    segment = _make_segment("seg-1", "dialogue", "おはよう", voice_id=VOICE_UNKNOWN)
    script = _make_script([segment])

    enrich_script(script, lang="ja")

    assert script.segments[0].text == "声が言った：おはよう"


def test_sfx_deterministic_same_id_same_variation() -> None:
    """Le même segment.id sélectionne toujours la même variation SFX (déterminisme)."""
    segment_a = _make_segment("seg-text-0", "sfx", "[ドン]")
    segment_b = _make_segment("seg-text-0", "sfx", "[ドン]")

    enrich_script(_make_script([segment_a]), lang="fr")
    enrich_script(_make_script([segment_b]), lang="fr")

    assert segment_a.text == segment_b.text


def test_sfx_variation_selection_by_crc32_id() -> None:
    """crc32(id) % 4 sélectionne déterministiquement la variation SFX (table exhaustive)."""
    ids_by_bucket = ["seg-text-1", "seg-text-5", "seg-text-0", "seg-text-4"]
    segments = [_make_segment(id_, "sfx", "[ドン]") for id_ in ids_by_bucket]
    script = _make_script(segments)

    enrich_script(script, lang="fr")

    for segment, variation in zip(script.segments, _SFX_PREFIX_VARIATIONS_FR, strict=True):
        assert segment.text == variation.format(texte="[ドン]")


def test_sfx_ja_unchanged() -> None:
    """Un SFX en ja reste inchangé (aucun gabarit ja spécifié)."""
    segment = _make_segment("seg-1", "sfx", "[ドン]")
    script = _make_script([segment])

    enrich_script(script, lang="ja")

    assert script.segments[0].text == "[ドン]"


def test_narration_unchanged_fr() -> None:
    """Une narration reste inchangée en fr (déjà narrative)."""
    segment = _make_segment("seg-1", "narration", "静かな夜だった。")
    script = _make_script([segment])

    enrich_script(script, lang="fr")

    assert script.segments[0].text == "静かな夜だった。"


def test_narration_unchanged_ja() -> None:
    """Une narration reste inchangée en ja."""
    segment = _make_segment("seg-1", "narration", "静かな夜だった。")
    script = _make_script([segment])

    enrich_script(script, lang="ja")

    assert script.segments[0].text == "静かな夜だった。"


def test_scene_description_unchanged_even_in_ja() -> None:
    """scene_description reste inchangé même avec lang='ja' (toujours généré en fr par Qwen3-VL)."""
    segment = _make_segment("seg-1", "scene_description", "Une rue calme sous la pluie.")
    script = _make_script([segment])

    enrich_script(script, lang="ja")

    assert script.segments[0].text == "Une rue calme sous la pluie."


def test_thought_unchanged() -> None:
    """thought reste inchangé (hors périmètre de cette phase)."""
    segment = _make_segment("seg-1", "thought", "本当にいいのかな…", voice_id="jf_alpha")
    script = _make_script([segment])

    enrich_script(script, lang="fr")

    assert script.segments[0].text == "本当にいいのかな…"


def test_unsupported_lang_raises() -> None:
    """Un lang non supporté ('en') lève ValueError."""
    segment = _make_segment("seg-1", "narration", "texte")
    script = _make_script([segment])

    with pytest.raises(ValueError, match="en"):
        enrich_script(script, lang="en")


def test_enrich_script_returns_same_instance() -> None:
    """enrich_script() mute en place et retourne la même instance NarrativeScript."""
    segment = _make_segment("seg-1", "narration", "texte")
    script = _make_script([segment])

    result = enrich_script(script, lang="fr")

    assert result is script
