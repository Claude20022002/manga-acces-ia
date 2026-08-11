"""Détection de caractères japonais, partagée entre audio_assembler.py et translation.py."""

from __future__ import annotations

import re

JAPANESE_CHAR_PATTERN = re.compile("[\\u3040-\\u309f\\u30a0-\\u30ff\\u4e00-\\u9fff]")
"""Hiragana, katakana, kanji — pas la ponctuation CJK (。、！？ etc.).

Extrait depuis audio_assembler.py (Phase 3, traduction des dialogues) :
`translate_dialogues()` (pipeline/translation.py) a besoin exactement du
même test "contient du japonais" que `_detect_lang()` (audio_assembler.py),
pour ne traduire que ce qui en a besoin — une divergence entre les deux
casserait silencieusement la détection (même leçon que le partage de
`find_images`, Phase 9).
"""
