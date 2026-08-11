"""Tests du script scripts/create_character_bank.py (CLI de construction de character_bank)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "create_character_bank.py"


def _load_script_module() -> ModuleType:
    """Charge scripts/create_character_bank.py comme module (scripts/ n'est pas un package)."""
    spec = importlib.util.spec_from_file_location("create_character_bank", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE = _load_script_module()
build_character_bank = _MODULE.build_character_bank


def _make_reference_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color="red").save(path)


def test_build_character_bank_writes_expected_json(tmp_path: Path) -> None:
    """Un mapping valide produit un JSON {"names": [...], "image_paths": [...]} aligné."""
    images_dir = tmp_path / "refs"
    _make_reference_image(images_dir / "naruto.jpg")
    _make_reference_image(images_dir / "sakura.jpg")
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps({"Naruto": "naruto.jpg", "Sakura": "sakura.jpg"}), encoding="utf-8"
    )
    output_path = tmp_path / "banks" / "naruto_manga.json"

    build_character_bank(images_dir, mapping_path, output_path)

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(result["names"]) == {"Naruto", "Sakura"}
    assert len(result["image_paths"]) == 2
    assert all(Path(p).is_file() for p in result["image_paths"])


def test_build_character_bank_missing_image_raises(tmp_path: Path) -> None:
    """Une image référencée mais absente lève ValueError plutôt que d'écrire une banque invalide."""
    images_dir = tmp_path / "refs"
    images_dir.mkdir()
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps({"Naruto": "missing.jpg"}), encoding="utf-8")
    output_path = tmp_path / "out.json"

    with pytest.raises(ValueError, match="introuvable"):
        build_character_bank(images_dir, mapping_path, output_path)

    assert not output_path.exists()


def test_build_character_bank_empty_mapping_raises(tmp_path: Path) -> None:
    """Un mapping vide lève ValueError."""
    images_dir = tmp_path / "refs"
    images_dir.mkdir()
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps({}), encoding="utf-8")
    output_path = tmp_path / "out.json"

    with pytest.raises(ValueError, match="vide"):
        build_character_bank(images_dir, mapping_path, output_path)
