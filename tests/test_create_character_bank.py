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
main = _MODULE.main


def _make_reference_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color="red").save(path)


def test_build_character_bank_writes_expected_json(tmp_path: Path) -> None:
    """Des paires (image, nom) valides produisent un JSON {"names": [...], "image_paths": [...]} aligné."""
    naruto_path = tmp_path / "naruto.jpg"
    sakura_path = tmp_path / "sakura.jpg"
    _make_reference_image(naruto_path)
    _make_reference_image(sakura_path)
    output_path = tmp_path / "banks" / "naruto_manga.json"

    build_character_bank([(str(naruto_path), "Naruto"), (str(sakura_path), "Sakura")], output_path)

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["names"] == ["Naruto", "Sakura"]
    assert len(result["image_paths"]) == 2
    assert all(Path(p).is_file() for p in result["image_paths"])


def test_build_character_bank_stores_absolute_paths(tmp_path: Path) -> None:
    """Les chemins écrits sont résolus en absolu (valides quel que soit le dossier de chargement)."""
    naruto_path = tmp_path / "naruto.jpg"
    _make_reference_image(naruto_path)
    output_path = tmp_path / "banks" / "naruto_manga.json"

    build_character_bank([(str(naruto_path), "Naruto")], output_path)

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert Path(result["image_paths"][0]).is_absolute()


def test_build_character_bank_missing_image_raises(tmp_path: Path) -> None:
    """Une image référencée mais absente lève ValueError plutôt que d'écrire une banque invalide."""
    output_path = tmp_path / "out.json"

    with pytest.raises(ValueError, match="introuvable"):
        build_character_bank([("missing.jpg", "Naruto")], output_path)

    assert not output_path.exists()


def test_build_character_bank_empty_entries_raises(tmp_path: Path) -> None:
    """Une liste d'entrées vide lève ValueError."""
    output_path = tmp_path / "out.json"

    with pytest.raises(ValueError, match="aucune paire"):
        build_character_bank([], output_path)


def test_main_odd_number_of_positional_args_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Un nombre impair d'arguments positionnels (paire incomplète) fait échouer le CLI."""
    naruto_path = tmp_path / "naruto.jpg"
    _make_reference_image(naruto_path)
    output_path = tmp_path / "out.json"
    monkeypatch.setattr(
        "sys.argv", ["create_character_bank.py", "--output", str(output_path), str(naruto_path), "Naruto", "orphan.jpg"]
    )

    with pytest.raises(SystemExit):
        main()

    assert not output_path.exists()


def test_main_writes_bank_from_positional_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Le CLI accepte des paires positionnelles <image> <nom> et écrit la character_bank."""
    naruto_path = tmp_path / "naruto.jpg"
    sasuke_path = tmp_path / "sasuke.jpg"
    _make_reference_image(naruto_path)
    _make_reference_image(sasuke_path)
    output_path = tmp_path / "banks" / "mon_manga.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "create_character_bank.py",
            "--output",
            str(output_path),
            str(naruto_path),
            "Naruto",
            str(sasuke_path),
            "Sasuke",
        ],
    )

    main()

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["names"] == ["Naruto", "Sasuke"]
