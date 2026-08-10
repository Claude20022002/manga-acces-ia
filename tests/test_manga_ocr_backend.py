"""Tests de l'heuristique de classification dialogue/sfx (_classify_text_type)."""

from __future__ import annotations

from manga_access.backends.manga_ocr_backend import _classify_text_type


def test_classify_dialogue_with_particles() -> None:
    """Texte avec particules grammaticales (は, です) -> dialogue."""
    assert _classify_text_type("これは重要です") == "dialogue"


def test_classify_sfx_known_corpus_case_batabata() -> None:
    """バタバタッ (cas corpus documenté, katakana pur) -> sfx."""
    assert _classify_text_type("バタバタッ") == "sfx"


def test_classify_sfx_ocr_corrupted_variant() -> None:
    """バタいタッ (variante corrompue par l'OCR, un hiragana en plus) -> sfx quand même."""
    assert _classify_text_type("バタいタッ") == "sfx"


def test_classify_sfx_known_corpus_case_po() -> None:
    """ポッ (cas corpus documenté) -> sfx."""
    assert _classify_text_type("ポッ") == "sfx"


def test_classify_sfx_generic_onomatopoeia() -> None:
    """ドン (onomatopée générique courte) -> sfx."""
    assert _classify_text_type("ドン") == "sfx"


def test_classify_empty_string_defaults_to_dialogue() -> None:
    """Chaîne vide -> dialogue (défaut sûr)."""
    assert _classify_text_type("") == "dialogue"


def test_classify_whitespace_only_defaults_to_dialogue() -> None:
    """Chaîne composée uniquement d'espaces -> dialogue (défaut sûr)."""
    assert _classify_text_type("   ") == "dialogue"


def test_classify_short_dialogue_with_sentence_punctuation() -> None:
    """え？ : dialogue très court mais avec ponctuation de phrase -> dialogue."""
    assert _classify_text_type("え？") == "dialogue"


def test_classify_narration_like_with_particle_and_punctuation() -> None:
    """時は流れた。: particule は et ponctuation 。 -> dialogue."""
    assert _classify_text_type("時は流れた。") == "dialogue"


def test_classify_kanji_only_without_particle() -> None:
    """深夜 : kanji seul, aucune particule, ratio katakana nul -> dialogue."""
    assert _classify_text_type("深夜") == "dialogue"


def test_classify_latin_text_without_japanese_chars() -> None:
    """Texte latin sans caractère japonais -> dialogue."""
    assert _classify_text_type("Hello there") == "dialogue"


def test_classify_hiragana_onomatopoeia_known_limitation() -> None:
    """わあわあ : onomatopée en hiragana -> dialogue (faux négatif assumé,
    documenté comme limite connue de l'heuristique ; capté ici par la
    particule finale 'わ', pas seulement par l'absence de katakana)."""
    assert _classify_text_type("わあわあ") == "dialogue"


def test_classify_katakana_loanword_known_limitation() -> None:
    """オーケー ('okay') : emprunt katakana sans particule -> sfx (faux
    positif assumé, documenté comme limite connue de l'heuristique)."""
    assert _classify_text_type("オーケー") == "sfx"
