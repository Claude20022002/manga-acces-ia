"""Tests de load_character_bank (chargement runtime d'une character_bank écrite sur disque)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from manga_access.pipeline.character_bank import load_character_bank


def _make_reference_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color="blue").save(path)


def test_load_character_bank_returns_ndarrays_aligned_with_names(tmp_path: Path) -> None:
    """Charge les images en np.ndarray, alignées 1:1 avec les noms."""
    _make_reference_image(tmp_path / "naruto.jpg")
    _make_reference_image(tmp_path / "sakura.jpg")
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(
        json.dumps({"names": ["Naruto", "Sakura"], "image_paths": ["naruto.jpg", "sakura.jpg"]}),
        encoding="utf-8",
    )

    bank = load_character_bank(bank_path)

    assert bank["names"] == ["Naruto", "Sakura"]
    assert len(bank["images"]) == 2
    assert all(isinstance(img, np.ndarray) for img in bank["images"])


def test_load_character_bank_resolves_relative_paths_from_json_dir(tmp_path: Path) -> None:
    """Les chemins relatifs du JSON sont résolus relativement au dossier du JSON, pas au cwd."""
    subdir = tmp_path / "banks"
    subdir.mkdir()
    _make_reference_image(tmp_path / "naruto.jpg")
    bank_path = subdir / "bank.json"
    bank_path.write_text(
        json.dumps({"names": ["Naruto"], "image_paths": ["../naruto.jpg"]}), encoding="utf-8"
    )

    bank = load_character_bank(bank_path)

    assert len(bank["images"]) == 1


def test_load_character_bank_mismatched_lengths_raises(tmp_path: Path) -> None:
    """Un nombre de noms différent du nombre d'images lève ValueError."""
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(
        json.dumps({"names": ["Naruto", "Sakura"], "image_paths": ["naruto.jpg"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_character_bank(bank_path)
