"""Tests de l'enrichissement narratif du NarrativeScript (segments de narration contextuels)."""

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
    """Dialogue connu, voix jf_* : un segment narration 'Elle dit :' est inséré avant, en ff_siwis."""
    dialogue = _make_segment("seg-1", "dialogue", "おはよう", voice_id="jf_alpha")
    script = _make_script([dialogue])

    enrich_script(script, lang="fr")

    assert [s.kind for s in script.segments] == ["narration", "dialogue"]
    assert script.segments[0].text == "Elle dit :"
    assert script.segments[0].voice_id == "ff_siwis"
    assert script.segments[1] is dialogue
    assert dialogue.text == "おはよう"
    assert dialogue.voice_id == "jf_alpha"


def test_dialogue_known_speaker_male_fr() -> None:
    """Dialogue connu, voix jm_* : le segment narration inséré dit 'Il dit :'."""
    dialogue = _make_segment("seg-1", "dialogue", "おはよう", voice_id="jm_kumo")
    script = _make_script([dialogue])

    enrich_script(script, lang="fr")

    assert script.segments[0].text == "Il dit :"


def test_dialogue_known_speaker_unknown_gender_fr() -> None:
    """Une voix qui ne suit pas la convention f/m donne 'Le personnage dit :'."""
    dialogue = _make_segment("seg-1", "dialogue", "おはよう", voice_id="voice-custom")
    script = _make_script([dialogue])

    enrich_script(script, lang="fr")

    assert script.segments[0].text == "Le personnage dit :"


def test_dialogue_known_speaker_ja_suffix_after() -> None:
    """ja, speaker connu : le dialogue reste inchangé, suivi d'un segment narration '、と言った'."""
    dialogue = _make_segment("seg-1", "dialogue", "おはよう", voice_id="jf_alpha")
    script = _make_script([dialogue])

    enrich_script(script, lang="ja")

    assert [s.kind for s in script.segments] == ["dialogue", "narration"]
    assert script.segments[0] is dialogue
    assert dialogue.text == "おはよう"
    assert script.segments[1].text == "、と言った"
    assert script.segments[1].voice_id == VOICE_NARRATOR


def test_dialogue_unknown_speaker_fr() -> None:
    """VOICE_UNKNOWN, fr : segment narration 'Une voix dit :' inséré avant, dialogue inchangé."""
    dialogue = _make_segment("seg-1", "dialogue", "おはよう", voice_id=VOICE_UNKNOWN)
    script = _make_script([dialogue])

    enrich_script(script, lang="fr")

    assert [s.kind for s in script.segments] == ["narration", "dialogue"]
    assert script.segments[0].text == "Une voix dit :"
    assert script.segments[0].voice_id == "ff_siwis"
    assert script.segments[1] is dialogue
    assert dialogue.voice_id == VOICE_UNKNOWN


def test_dialogue_unknown_speaker_ja() -> None:
    """VOICE_UNKNOWN, ja : segment narration '声が言った：' inséré avant, dialogue inchangé."""
    dialogue = _make_segment("seg-1", "dialogue", "おはよう", voice_id=VOICE_UNKNOWN)
    script = _make_script([dialogue])

    enrich_script(script, lang="ja")

    assert [s.kind for s in script.segments] == ["narration", "dialogue"]
    assert script.segments[0].text == "声が言った："
    assert script.segments[0].voice_id == VOICE_NARRATOR


def test_dialogue_same_speaker_consecutive_no_second_prefix() -> None:
    """Deux dialogues consécutifs du même voice_id : un seul segment narration, sur le premier."""
    dialogue_1 = _make_segment("seg-1", "dialogue", "おはよう", voice_id="jf_alpha")
    dialogue_2 = _make_segment("seg-2", "dialogue", "元気？", voice_id="jf_alpha")
    script = _make_script([dialogue_1, dialogue_2])

    enrich_script(script, lang="fr")

    assert [s.kind for s in script.segments] == ["narration", "dialogue", "dialogue"]
    assert script.segments[1] is dialogue_1
    assert script.segments[2] is dialogue_2


def test_dialogue_different_speaker_gets_prefix_each_time() -> None:
    """Deux dialogues de voice_id différents : chacun garde son segment narration."""
    dialogue_1 = _make_segment("seg-1", "dialogue", "おはよう", voice_id="jf_alpha")
    dialogue_2 = _make_segment("seg-2", "dialogue", "元気？", voice_id="jm_kumo")
    script = _make_script([dialogue_1, dialogue_2])

    enrich_script(script, lang="fr")

    assert [s.kind for s in script.segments] == ["narration", "dialogue", "narration", "dialogue"]
    assert script.segments[0].text == "Elle dit :"
    assert script.segments[2].text == "Il dit :"


def test_dialogue_same_speaker_across_intervening_sfx() -> None:
    """Un sfx intercalé n'interrompt pas le suivi du dernier locuteur de dialogue."""
    dialogue_1 = _make_segment("seg-1", "dialogue", "おはよう", voice_id="jf_alpha")
    sfx = _make_segment("seg-sfx", "sfx", "[ドン]")
    dialogue_2 = _make_segment("seg-2", "dialogue", "元気？", voice_id="jf_alpha")
    script = _make_script([dialogue_1, sfx, dialogue_2])

    enrich_script(script, lang="fr")

    assert [s.kind for s in script.segments] == [
        "narration", "dialogue", "narration", "sfx", "dialogue",
    ]


def test_dialogue_same_unknown_speaker_consecutive_no_second_prefix() -> None:
    """Deux dialogues consécutifs à VOICE_UNKNOWN : un seul 'Une voix dit :', sur le premier."""
    dialogue_1 = _make_segment("seg-1", "dialogue", "おはよう", voice_id=VOICE_UNKNOWN)
    dialogue_2 = _make_segment("seg-2", "dialogue", "元気？", voice_id=VOICE_UNKNOWN)
    script = _make_script([dialogue_1, dialogue_2])

    enrich_script(script, lang="fr")

    assert [s.kind for s in script.segments] == ["narration", "dialogue", "dialogue"]
    assert script.segments[0].text == "Une voix dit :"


def test_dialogue_ja_same_speaker_consecutive_no_second_suffix() -> None:
    """ja, même speaker : seul le premier dialogue reçoit le suffixe '、と言った'."""
    dialogue_1 = _make_segment("seg-1", "dialogue", "おはよう", voice_id="jf_alpha")
    dialogue_2 = _make_segment("seg-2", "dialogue", "元気？", voice_id="jf_alpha")
    script = _make_script([dialogue_1, dialogue_2])

    enrich_script(script, lang="ja")

    assert [s.kind for s in script.segments] == ["dialogue", "narration", "dialogue"]


def test_sfx_deterministic_same_id_same_variation() -> None:
    """Le même segment.id sélectionne toujours la même variation SFX (déterminisme)."""
    sfx_a = _make_segment("seg-text-0", "sfx", "[ドン]")
    sfx_b = _make_segment("seg-text-0", "sfx", "[ドン]")

    enrich_script(_make_script([sfx_a]), lang="fr")
    enrich_script(_make_script([sfx_b]), lang="fr")

    assert sfx_a.text == sfx_b.text == "[ドン]"  # segment sfx d'origine inchangé dans les deux cas


def test_sfx_variation_selection_by_crc32_id() -> None:
    """crc32(id) % 4 sélectionne déterministiquement la variation SFX (table exhaustive)."""
    ids_by_bucket = ["seg-text-1", "seg-text-5", "seg-text-0", "seg-text-4"]
    sfx_segments = [_make_segment(id_, "sfx", "[ドン]") for id_ in ids_by_bucket]
    script = _make_script(sfx_segments)

    enrich_script(script, lang="fr")

    narrator_texts = [script.segments[i].text for i in range(0, 8, 2)]
    assert narrator_texts == list(_SFX_PREFIX_VARIATIONS_FR)


def test_sfx_fr_inserts_narration_before_unchanged_sfx() -> None:
    """SFX fr : un segment narration est inséré avant le sfx, qui reste inchangé."""
    sfx = _make_segment("seg-1", "sfx", "[ドン]")
    script = _make_script([sfx])

    enrich_script(script, lang="fr")

    assert [s.kind for s in script.segments] == ["narration", "sfx"]
    assert script.segments[0].voice_id == "ff_siwis"
    assert script.segments[1] is sfx
    assert sfx.text == "[ドン]"


def test_sfx_always_announced_even_consecutive() -> None:
    """Deux sfx consécutifs gardent chacun leur segment narration (pas de suivi de locuteur pour sfx)."""
    sfx_1 = _make_segment("seg-1", "sfx", "[ドン]")
    sfx_2 = _make_segment("seg-2", "sfx", "[ガチャ]")
    script = _make_script([sfx_1, sfx_2])

    enrich_script(script, lang="fr")

    assert [s.kind for s in script.segments] == ["narration", "sfx", "narration", "sfx"]


def test_sfx_ja_unchanged() -> None:
    """Un SFX en ja reste seul, sans segment narration ajouté (aucun gabarit ja spécifié)."""
    sfx = _make_segment("seg-1", "sfx", "[ドン]")
    script = _make_script([sfx])

    enrich_script(script, lang="ja")

    assert script.segments == [sfx]


def test_narration_unchanged_fr() -> None:
    """Une narration reste inchangée en fr, aucun segment supplémentaire n'est ajouté."""
    narration = _make_segment("seg-1", "narration", "静かな夜だった。")
    script = _make_script([narration])

    enrich_script(script, lang="fr")

    assert script.segments == [narration]


def test_narration_unchanged_ja() -> None:
    """Une narration reste inchangée en ja."""
    narration = _make_segment("seg-1", "narration", "静かな夜だった。")
    script = _make_script([narration])

    enrich_script(script, lang="ja")

    assert script.segments == [narration]


def test_scene_description_unchanged_even_in_ja() -> None:
    """scene_description reste inchangé même avec lang='ja' (toujours généré en fr par Qwen3-VL)."""
    scene = _make_segment("seg-1", "scene_description", "Une rue calme sous la pluie.")
    script = _make_script([scene])

    enrich_script(script, lang="ja")

    assert script.segments == [scene]


def test_thought_unchanged() -> None:
    """thought reste inchangé (hors périmètre de cette phase)."""
    thought = _make_segment("seg-1", "thought", "本当にいいのかな…", voice_id="jf_alpha")
    script = _make_script([thought])

    enrich_script(script, lang="fr")

    assert script.segments == [thought]


def test_unsupported_lang_raises() -> None:
    """Un lang non supporté ('en') lève ValueError."""
    script = _make_script([_make_segment("seg-1", "narration", "texte")])

    with pytest.raises(ValueError, match="en"):
        enrich_script(script, lang="en")


def test_enrich_script_returns_same_instance() -> None:
    """enrich_script() retourne la même instance NarrativeScript (segments remplacés en place)."""
    script = _make_script([_make_segment("seg-1", "narration", "texte")])

    result = enrich_script(script, lang="fr")

    assert result is script


def test_mixed_script_preserves_order_and_inserts_only_where_needed() -> None:
    """Un script mixte garde son ordre global ; seuls dialogue/sfx gagnent un segment narration."""
    narration = _make_segment("seg-n", "narration", "Le vent soufflait.")
    dialogue = _make_segment("seg-d", "dialogue", "おはよう", voice_id="jm_kumo")
    sfx = _make_segment("seg-s", "sfx", "[ドン]")
    script = _make_script([narration, dialogue, sfx])

    enrich_script(script, lang="fr")

    assert [s.kind for s in script.segments] == [
        "narration",  # narration d'origine, inchangée
        "narration",  # préfixe inséré avant dialogue
        "dialogue",
        "narration",  # préfixe inséré avant sfx
        "sfx",
    ]
    assert script.segments[0] is narration
